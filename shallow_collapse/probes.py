import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict
from typing import Dict, Any, List, Optional
import copy
import torch
import torch.nn.functional as F
from torch_scatter import scatter

PLACEHOLDER_LAYER_ID = -1

class NCProbe():
    """
    Probe NC metrics when training in a mini-batch fashion.
    """
    def __init__(self, context) -> None:
        self.context = context
        self.num_classes = self.context["num_classes"]
        self.reset_state()

    def reset_state(self):
        self.layerwise_global_sum = {}
        self.layerwise_global_mean = {}
        self.layerwise_class_means = {}
        self.layerwise_class_sums = {}
        self.layerwise_class_cov = {}
        self.class_sizes = torch.zeros((self.num_classes)).to(self.context["device"])

    def print_state(self):
        logger.info("layerwise_global_sum: {}".format(self.layerwise_global_sum))
        logger.info("layerwise_global_mean: {}".format(self.layerwise_global_mean))
        logger.info("layerwise_class_means: {}".format(self.layerwise_class_means))
        logger.info("layerwise_class_sums: {}".format(self.layerwise_class_sums))
        logger.info("layerwise_class_cov: {}".format(self.layerwise_class_cov))
        logger.info("class_sizes: {}".format(self.class_sizes))

    def compute_class_sizes(self, training_data):
        for _, labels in training_data.nc_train_loader:
            labels = labels.to(self.context["device"])
            class_count = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).sum(dim = 0)
            self.class_sizes += class_count


    def _initialize_placeholders(self, layer_idx, feat_size):
        """
        Args:
            layer_idx: index of the layer
            feat_shape: int
        """
        if layer_idx not in self.layerwise_class_sums:
            self.layerwise_class_sums[layer_idx] = torch.zeros((self.num_classes, feat_size)).to(self.context["device"])
        if layer_idx not in self.layerwise_class_cov:
            self.layerwise_class_cov[layer_idx] = {
                "S_W": torch.zeros((feat_size, feat_size)).to(self.context["device"]),
                "S_B": torch.zeros((feat_size, feat_size)).to(self.context["device"])
            }

    def _track_sums(self, features, labels):
        for layer_idx, feat in features.items():
            self._initialize_placeholders(layer_idx=layer_idx, feat_size=feat.shape[1])
            # accumulate the feature values per class
            class_sum = scatter(feat, labels, dim=0, reduce="sum")
            unique_labels = torch.unique(labels)
            # the output of scatter uses zeros as placeholder values
            # corresponding to labels which are not present in this batch.
            # To have a one-one correspondence between unique labels and
            # the class_sim, we need this additional filter operation.
            class_sum = class_sum[unique_labels]
            self.layerwise_class_sums[layer_idx][unique_labels] += class_sum

    def _compute_means(self):
        self.layerwise_class_means = {}
        for layer_idx in self.layerwise_class_sums.keys():
            self.layerwise_global_sum[layer_idx] = torch.sum(self.layerwise_class_sums[layer_idx], dim=0)
            self.layerwise_global_mean[layer_idx] = self.layerwise_global_sum[layer_idx]/torch.sum(self.class_sizes)
            # print(self.layerwise_class_sums[layer_idx].shape, self.class_sizes.shape)
            self.layerwise_class_means[layer_idx] = torch.div(self.layerwise_class_sums[layer_idx], self.class_sizes.unsqueeze(1))


    def _track_cov(self, features, labels):
        """
        Track Within-class feature cov as between-class can be computed
        directly from the already computed means
        """

        for layer_idx, feat in features.items():
            expanded_class_means = self.layerwise_class_means[layer_idx][labels]
            z_W = feat - expanded_class_means
            # S_W of shape d x d
            S_W = z_W.t() @ z_W
            self.layerwise_class_cov[layer_idx]["S_W"] += S_W

    def _compute_cov(self):
        """
        Perform normalization for S_W and compute S_B
        """
        num_datapoints = torch.sum(self.class_sizes)
        for layer_idx in self.layerwise_class_cov.keys():
            self.layerwise_class_cov[layer_idx]["S_W"] /= num_datapoints
            class_means = self.layerwise_class_means[layer_idx]
            global_mean = self.layerwise_global_mean[layer_idx]
            z_B = class_means - global_mean
            # S_B : d x d
            S_B = z_B.t() @ z_B
            S_B /= self.num_classes
            self.layerwise_class_cov[layer_idx]["S_B"] = S_B

    def compute_nc1(self):
        nc1_metrics = {}
        for layer_idx in self.layerwise_class_cov.keys():
            S_W = self.layerwise_class_cov[layer_idx]["S_W"]
            S_B = self.layerwise_class_cov[layer_idx]["S_B"]
            try:
                nc1 = torch.trace(S_W @ torch.linalg.pinv(S_B)) / self.num_classes
            except Exception as e:
                nc1 = torch.ones(1).to(self.context["device"])[0]
                logger.error("Exception raised while computing NC1: {}".format(str(e)))

            nc1_hat = torch.trace(S_W)/torch.trace(S_B)
            nc1_metrics[layer_idx] = {}
            nc1_metrics[layer_idx]["trace_S_W_pinv_S_B"] = torch.log10(nc1).detach().cpu().numpy()
            nc1_metrics[layer_idx]["trace_S_W_div_S_B"] = torch.log10(nc1_hat).detach().cpu().numpy()
            nc1_metrics[layer_idx]["trace_S_W"] = torch.log10(torch.trace(S_W)).detach().cpu().numpy()
            nc1_metrics[layer_idx]["trace_S_B"] = torch.log10(torch.trace(S_B)).detach().cpu().numpy()
        return nc1_metrics

    @staticmethod
    def compute_kernel_nc1(K, N, class_sizes):
        """
        Make sure that the kernel matrix is ordered in blocks. The default
        implementation of nngp and ntk lim kernels takes care of it.

        We do not assume that the classes are balanced. Thus, a 1D tensor
        `class_sizes` is required.
        """
        class_sizes = class_sizes.cpu().numpy()
        num_classes = class_sizes.shape[0]
        assert N == torch.sum(class_sizes)
        Tr_Sigma_W = 0
        Tr_Sigma_B = 0
        x_idx = 0
        class_block_sums = OrderedDict()
        for c in range(num_classes):
            class_size = class_sizes[c]
            block_sum = torch.sum( K[x_idx : x_idx + class_size, x_idx : x_idx + class_size] )
            class_block_sums[c] = block_sum
            x_idx += class_size

        Tr_Sigma_G = torch.sum(K)/(N**2)
        Tr_Sigma_tilde_T = torch.sum(torch.diag(K)) / N
        Tr_Sigma_tilde_B = 0
        for c in range(num_classes):
            Tr_Sigma_tilde_B += class_block_sums[c]/(class_sizes[c]**2)
        Tr_Sigma_tilde_B /= num_classes

        Tr_Sigma_W = Tr_Sigma_tilde_T - Tr_Sigma_tilde_B
        Tr_Sigma_B = Tr_Sigma_tilde_B - Tr_Sigma_G

        logger.info("Tr_Sigma_W: {}".format(Tr_Sigma_W))
        logger.info("Tr_Sigma_B: {}".format(Tr_Sigma_B))
        return Tr_Sigma_W/Tr_Sigma_B

    @torch.no_grad()
    def capture(self, model, training_data, layer_type="activation"):
        """
        Given a model and the training data, do forward passes
        in a mini-batch fashion and track the NC metrics.
        Args:
            model: torch model
            training_data: torch data loader for convenient mini-batching
            layer_type: Should be one of ["pre", "post"] to choose either
                pre-activations or post-activations as features
        """
        if layer_type not in ["affine", "activation"]:
            raise ValueError("layer_type should be one of ['affine', 'activation']")
        device = self.context["device"]
        self.compute_class_sizes(training_data=training_data)
        # one-pass to compute class means
        for data, labels in training_data.train_loader:
            model.zero_grad()
            data, labels = data.to(device), labels.to(device)
            _ = model(data)
            features = model.activation_features if layer_type == "activation" else model.affine_features
            self._track_sums(features=features, labels=labels.type(torch.int64))
        self._compute_means()
        # second-pass to compute covariance matrices
        for data, labels in training_data.train_loader:
            model.zero_grad()
            data, labels = data.to(device), labels.to(device)
            _ = model(data)
            features = model.activation_features if layer_type == "activation" else model.affine_features
            self._track_cov(features=features, labels=labels.type(torch.int64))
        self._compute_cov()
        nc_metrics = self.compute_nc1()
        self.reset_state()
        return nc_metrics

class DataNCProbe(NCProbe):
    """
    Probe NC metrics of data
    """
    def __init__(self, context) -> None:
        super().__init__(context)

    @torch.no_grad()
    def capture(self, training_data):
        """
        Given the training data, iterate in a mini-batch fashion
        and track the NC metrics of the data itself.
        Args:
            training_data: torch data loader for convenient mini-batching
        """
        device = self.context["device"]
        # one-pass to compute class means
        for data, labels in training_data.train_loader:
            data, labels = data.to(device), labels.to(device)
            features = {PLACEHOLDER_LAYER_ID : data}
            self._track_sums(features=features, labels=labels)
        self._compute_means()
        # second-pass to compute covariance matrices
        for data, labels in training_data.train_loader:
            data, labels = data.to(device), labels.to(device)
            features = {PLACEHOLDER_LAYER_ID : data}
            self._track_cov(features=features, labels=labels)
        self._compute_cov()
        nc_metrics = self.compute_nc1()
        return nc_metrics

class NTKNCProbe(NCProbe):
    """
    Compute NC metrics of the NTK features.
    NTK features = grad of output per input
    """
    def __init__(self, context) -> None:
        super().__init__(context)

    def _get_ntk_feat(self, model_copy):
        device = self.context["device"]
        ntk_feat = []
        for param in model_copy.parameters():
            ntk_feat.append(param.grad.view(-1))
        ntk_feat = torch.cat(ntk_feat).to(device)
        ntk_feat = ntk_feat.unsqueeze(0)
        logger.debug("ntk_feat shape: {}".format(ntk_feat.shape))
        return ntk_feat

    def capture(self, model, training_data):
        device = self.context["device"]
        model_copy = copy.deepcopy(model)
        # one-pass to compute class means
        ntk_feat_matrix = torch.zeros(0).to(device)
        for data, labels in training_data.train_loader:
            data, labels = data.to(device), labels.to(device)
            assert data.shape[0] == self.context["batch_size"]
            for x_idx in range(data.shape[0]):
                model_copy.zero_grad()
                x = data[x_idx, : ]
                pred = model_copy(x)
                logger.debug("shape of pred for single input: {}".format(pred.shape))
                pred.backward(retain_graph=True)
                ntk_feat = self._get_ntk_feat(model_copy=model_copy)
                ntk_feat_matrix = torch.cat([ntk_feat_matrix, ntk_feat], 0)
                logger.info("ntk_feat_matrix shape: {}".format(ntk_feat_matrix.shape))
            features = {PLACEHOLDER_LAYER_ID : ntk_feat_matrix}
            self._track_sums(features=features, labels=labels)
        self._compute_means()

        # second-pass to compute covariance matrices
        ntk_feat_matrix = torch.zeros(0).to(device)
        for data, labels in training_data.train_loader:
            data, labels = data.to(device), labels.to(device)
            assert data.shape[0] == self.context["batch_size"]
            for x_idx in range(data.shape[0]):
                model_copy.zero_grad()
                x = data[x_idx, : ]
                pred = model_copy(x)
                pred.backward(retain_graph=True)
                ntk_feat = self._get_ntk_feat(model_copy=model_copy)
                ntk_feat_matrix = torch.cat([ntk_feat_matrix, ntk_feat], 0)
            features = {PLACEHOLDER_LAYER_ID : ntk_feat_matrix}
            self._track_cov(features=features, labels=labels)
        self._compute_cov()
        nc_metrics = self.compute_nc1()
        return nc_metrics


class KernelProbe():
    def __init__(self, context) -> None:
        self.context = context

    def _nngp_relu_kernel_helper(self, nngp_kernel):
        N = self.context["N"]
        nngp_relu_kernel = torch.zeros_like(nngp_kernel)
        for i in range(N):
            for j in range(N):
                K_ii = nngp_kernel[i, i]
                K_ij = nngp_kernel[i, j]
                K_jj = nngp_kernel[j, j]
                theta = torch.arccos( K_ij/(torch.sqrt(K_ii*K_jj) + 1e-6) )
                val = (torch.sqrt(K_ii*K_jj)*( torch.sin(theta) + (torch.pi - theta)*torch.cos(theta) )) /(2*torch.pi)
                nngp_relu_kernel[i,j] = val
        return nngp_relu_kernel

    def _nngp_relu_derivative_kernel_helper(self, nngp_kernel):
        N = self.context["N"]
        nngp_relu_derivative_kernel = torch.zeros_like(nngp_kernel)
        for i in range(N):
            for j in range(N):
                K_ii = nngp_kernel[i, i]
                K_ij = nngp_kernel[i, j]
                K_jj = nngp_kernel[j, j]
                theta = torch.arccos( K_ij/(torch.sqrt(K_ii*K_jj) + 1e-6) )
                val = (torch.pi - theta) /(2*torch.pi)
                nngp_relu_derivative_kernel[i,j] = val
        return nngp_relu_derivative_kernel

    def compute_lim_nngp_kernels(self, training_data):
        """
        We use kaiming_normal init for weights and
        standard normal init for bias with:
        \sigma_w^2 = 2,
        \sigma_b^2 = 1.

        The recursive formulation is adapted from: https://arxiv.org/pdf/1711.00165.pdf
        """
        sigma_w_sq = 2
        sigma_b_sq = 1
        L = self.context["L"]
        X = training_data.X[training_data.perm_inv].to(self.context["device"])
        self.nngp_kernels = {}
        self.nngp_relu_kernels = {}
        # base case
        self.nngp_kernels[0] = sigma_b_sq + (sigma_w_sq/self.context["in_features"])*(X @ X.t())
        self.nngp_relu_kernels[0] = self._nngp_relu_kernel_helper(nngp_kernel=self.nngp_kernels[0])
        # Recursive formulation for subsequent layers.
        # note that we include the final layer as well
        for l in range(1, L):
            self.nngp_kernels[l] = sigma_b_sq + sigma_w_sq*self.nngp_relu_kernels[l-1]
            self.nngp_relu_kernels[l] = self._nngp_relu_kernel_helper(nngp_kernel=self.nngp_kernels[l])

    def compute_lim_ntk_kernels(self, training_data):
        """
        We use kaiming_normal init for weights and
        standard normal init for bias with:
        \sigma_w^2 = 2,
        \sigma_b^2 = 1.
        """
        if not hasattr(self, "nngp_kernels"):
            logger.warning("lim NNGP kernels have not been computed. \
                           Computing them before proceeding.")
            self.compute_lim_nngp_kernels(training_data=training_data)
        L = self.context["L"]
        self.ntk_kernels = {}
        self.nngp_relu_derivative_kernels = {}
        # base case
        self.ntk_kernels[0] = self.nngp_kernels[0]
        self.nngp_relu_derivative_kernels[0] = self._nngp_relu_derivative_kernel_helper(nngp_kernel=self.nngp_kernels[0])
        # Recursive formulation for subsequent layers.
        # note that we include the final layer as well
        for l in range(1, L):
            self.ntk_kernels[l] = self.nngp_kernels[l] + self.ntk_kernels[l-1] * self.nngp_relu_derivative_kernels[l-1]
            self.nngp_relu_derivative_kernels[l] = self._nngp_relu_derivative_kernel_helper(nngp_kernel=self.nngp_kernels[l])

    def compute_emp_nngp_kernels(self, model, training_data):
        self.emp_nngp_affine_kernels = OrderedDict()
        self.emp_nngp_activation_kernels = OrderedDict()
        model.zero_grad()
        X = training_data.X[training_data.perm_inv].to(self.context["device"])
        _ = model(X)
        # affine feature kernels
        for l in range(self.context["L"]):
            features = model.affine_features[l]
            # features = F.normalize(features, p=2, dim=1)
            emp_nngp_kernel = features @ features.t()
            self.emp_nngp_affine_kernels[l] = emp_nngp_kernel
        # activation feature kernels
        for l in range(self.context["L"]):
            features = model.activation_features[l]
            # features = F.normalize(features, p=2, dim=1)
            emp_nngp_kernel = features @ features.t()
            self.emp_nngp_activation_kernels[l] = emp_nngp_kernel

    def _get_ntk_feat(self, model_copy):
        device = self.context["device"]
        ntk_feat = []
        for param in model_copy.parameters():
            ntk_feat.append(param.grad.view(-1))
        ntk_feat = torch.cat(ntk_feat).to(device)
        ntk_feat = ntk_feat.unsqueeze(0)
        logger.debug("ntk_feat shape: {}".format(ntk_feat.shape))
        return ntk_feat

    def compute_emp_ntk_kernel(self, model, training_data):
        assert self.context["batch_size"] == self.context["N"]
        device = self.context["device"]
        model_copy = copy.deepcopy(model)
        ntk_feat_matrix = torch.zeros(0).to(device)
        for data, labels in training_data.train_loader:
            data, labels = data.to(device), labels.to(device)
            assert data.shape[0] == self.context["batch_size"]
            for x_idx in range(data.shape[0]):
                model_copy.zero_grad()
                x = data[x_idx, : ]
                pred = model_copy(x)
                logger.debug("shape of pred for single input: {}".format(pred.shape))
                pred.backward(retain_graph=True)
                ntk_feat = self._get_ntk_feat(model_copy=model_copy)
                ntk_feat_matrix = torch.cat([ntk_feat_matrix, ntk_feat], 0)
                # logger.info("ntk_feat_matrix shape: {}".format(ntk_feat_matrix.shape))
        features = ntk_feat_matrix[training_data.perm_inv]
        # features = F.normalize(features, p=2, dim=1)
        self.emp_ntk_kernel = features @ features.t()
