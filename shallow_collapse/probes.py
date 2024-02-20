import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict
from typing import Dict, Any, List, Optional
import copy
import torch
import torch.nn.functional as F
from torch_scatter import scatter

PLACEHOLDER_LAYER_ID = -1
EPSILON = 1e-10

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

    def print_state(self):
        logger.info("layerwise_global_sum: {}".format(self.layerwise_global_sum))
        logger.info("layerwise_global_mean: {}".format(self.layerwise_global_mean))
        logger.info("layerwise_class_means: {}".format(self.layerwise_class_means))
        logger.info("layerwise_class_sums: {}".format(self.layerwise_class_sums))
        logger.info("layerwise_class_cov: {}".format(self.layerwise_class_cov))

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
        assert N == sum(class_sizes)
        Tr_Sigma_W = 0
        Tr_Sigma_B = 0
        x_idx = 0
        class_block_sums = OrderedDict()
        for c in range(num_classes):
            class_size = class_sizes[c]
            block_sum = torch.sum( K[x_idx : x_idx + class_size, x_idx : x_idx + class_size] )
            class_block_sums[c] = block_sum
            x_idx += class_size

        logger.info("class block sums: {}".format(class_block_sums))
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
        self.class_sizes = training_data.class_sizes
        # one-pass to compute class means
        for data, y, labels in training_data.train_loader:
            model.zero_grad()
            data, y, labels = data.to(device), y.to(device), labels.to(device)
            _ = model(data)
            features = model.activation_features if layer_type == "activation" else model.affine_features
            self._track_sums(features=features, labels=labels.type(torch.int64))
        self._compute_means()
        # second-pass to compute covariance matrices
        for data, y, labels in training_data.train_loader:
            model.zero_grad()
            data, y, labels = data.to(device), y.to(device), labels.to(device)
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
        self.class_sizes = training_data.class_sizes
        # one-pass to compute class means
        for data, y, labels in training_data.train_loader:
            data, y, labels = data.to(device), y.to(device), labels.to(device)
            features = {PLACEHOLDER_LAYER_ID : data}
            self._track_sums(features=features, labels=labels.type(torch.int64))
        self._compute_means()
        # second-pass to compute covariance matrices
        for data, y, labels in training_data.train_loader:
            data, y, labels = data.to(device), y.to(device), labels.to(device)
            features = {PLACEHOLDER_LAYER_ID : data}
            self._track_cov(features=features, labels=labels.type(torch.int64))
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
        self.class_sizes = training_data.class_sizes
        # one-pass to compute class means
        ntk_feat_matrix = torch.zeros(0).to(device)
        for data, y, labels in training_data.train_loader:
            data, y, labels = data.to(device), y.to(device), labels.to(device)
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
        for data, y, labels in training_data.train_loader:
            data, y, labels = data.to(device), y.to(device), labels.to(device)
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
        self.emp_nngp_affine_kernels = OrderedDict()
        self.emp_nngp_activation_kernels = OrderedDict()
        self.emp_ntk_kernels = OrderedDict()

    def _nngp_erf_kernel_helper(self, nngp_kernel):
        diag_vals = torch.diag(nngp_kernel, 0)
        diag_vals_inv_sqrt = torch.sqrt(1.0/(1 + 2*diag_vals))
        diag_matrix_inv_sqrt = torch.diag(diag_vals_inv_sqrt)
        ratios = diag_matrix_inv_sqrt @ (2.0*nngp_kernel) @ diag_matrix_inv_sqrt

        thetas = torch.arcsin(ratios)
        # thetas = torch.arcsin(torch.clip(ratios, min=-1, max=1))
        nngp_erf_kernel = (2/torch.pi) * thetas
        return nngp_erf_kernel

    def _nngp_erf_derivative_kernel_helper(self, nngp_kernel):

        # (4/pi)*det(I_2 + 2K)^{-1/2}
        # K = [[K_11, K_12], [K_21, K_22]]
        # I_2 + 2K = [[1 + 2*K_11, K_12], [K_21, 1+2*K_22]]
        # det(I_2 + 2K)^{-1/2} = ((1+2*K_11)(1+2*K_22) - K_12 K_21)^{-1/2}

        diag_vals = torch.diag(nngp_kernel, 0)
        scaled_diag_vals = 1 + 2*diag_vals
        # scaled_diag_matrix = torch.diag(scaled_diag_vals)
        M = scaled_diag_vals @ scaled_diag_vals.t() - nngp_kernel * nngp_kernel.t()
        nngp_erf_derivative_kernel = (4/torch.pi)*torch.pow(M, -1/2)
        assert not torch.isnan(nngp_erf_derivative_kernel).any()
        assert torch.allclose(nngp_erf_derivative_kernel,nngp_erf_derivative_kernel.t())
        return nngp_erf_derivative_kernel

    def _nngp_relu_kernel_helper(self, nngp_kernel):
        nngp_relu_kernel = torch.zeros_like(nngp_kernel)
        diag_vals = torch.diag(nngp_kernel, 0)
        diag_vals_sqrt = torch.sqrt(diag_vals)
        diag_matrix_sqrt = torch.diag(diag_vals_sqrt)

        diag_vals_inv_sqrt = torch.sqrt(1.0/(diag_vals + EPSILON))
        diag_matrix_inv_sqrt = torch.diag(diag_vals_inv_sqrt)
        ratios = diag_matrix_inv_sqrt @ nngp_kernel @ diag_matrix_inv_sqrt
        
        thetas = torch.arccos(ratios)
        # thetas = torch.arccos(torch.clip(ratios, min=-1, max=1))
        nngp_relu_kernel = diag_matrix_sqrt @ (( torch.sin(thetas) + (torch.pi - thetas)*torch.cos(thetas) )/(2*torch.pi)) @ diag_matrix_sqrt
        assert torch.allclose(nngp_relu_kernel, nngp_relu_kernel.t())
        return nngp_relu_kernel

    def _nngp_relu_derivative_kernel_helper(self, nngp_kernel):
        nngp_relu_derivative_kernel = torch.zeros_like(nngp_kernel)
        diag_vals = torch.diag(nngp_kernel, 0)
        diag_vals_inv_sqrt = torch.sqrt(1.0/(diag_vals + EPSILON))
        diag_matrix_inv_sqrt = torch.diag(diag_vals_inv_sqrt)
        ratios = diag_matrix_inv_sqrt @ nngp_kernel @ diag_matrix_inv_sqrt
        thetas = torch.arccos(ratios)
        # thetas = torch.arccos(torch.clip(ratios, min=-1, max=1))
        nngp_relu_derivative_kernel = (torch.pi - thetas) /(2*torch.pi)
        assert torch.allclose(nngp_relu_derivative_kernel, nngp_relu_derivative_kernel.t())
        return nngp_relu_derivative_kernel

    def _nngp_activation_kernel_helper(self, nngp_kernel):
        if self.context["activation"] == "relu":
            return self._nngp_relu_kernel_helper(nngp_kernel=nngp_kernel)
        elif self.context["activation"] == "erf":
            return self._nngp_erf_kernel_helper(nngp_kernel=nngp_kernel)

    def _nngp_activation_derivative_kernel_helper(self, nngp_kernel):
        if self.context["activation"] == "relu":
            return self._nngp_relu_derivative_kernel_helper(nngp_kernel=nngp_kernel)
        elif self.context["activation"] == "erf":
            return self._nngp_erf_derivative_kernel_helper(nngp_kernel=nngp_kernel)

    def compute_lim_nngp_kernels(self, training_data):
        """
        We use kaiming_normal init for weights and
        standard normal init for bias with:
        \sigma_w^2 = 2,
        \sigma_b^2 = 1.

        The recursive formulation is adapted from: https://arxiv.org/pdf/1711.00165.pdf
        """
        sigma_w_sq = self.context["hidden_weight_std"]**2
        sigma_b_sq = self.context["bias_std"]**2
        L = self.context["L"]
        X = training_data.X[training_data.perm_inv].to(self.context["device"])
        self.nngp_kernels = {}
        self.nngp_activation_kernels = {}
        # base case
        self.nngp_kernels[0] = sigma_b_sq + (sigma_w_sq/self.context["in_features"])*(X @ X.t())
        self.nngp_activation_kernels[0] = self._nngp_activation_kernel_helper(nngp_kernel=self.nngp_kernels[0])
        # Recursive formulation for subsequent layers.
        # note that we include the final layer as well.
        # The last layes doesn't contain activations but we calculate for analysis purposes.
        for l in range(1, L):
            self.nngp_kernels[l] = sigma_b_sq + sigma_w_sq*self.nngp_activation_kernels[l-1]
            self.nngp_activation_kernels[l] = self._nngp_activation_kernel_helper(nngp_kernel=self.nngp_kernels[l])

    def compute_lim_ntk_kernels(self, training_data):
        if not hasattr(self, "nngp_kernels"):
            logger.warning("lim NNGP kernels have not been computed. \
                           Computing them before proceeding.")
            self.compute_lim_nngp_kernels(training_data=training_data)
        L = self.context["L"]
        self.ntk_kernels = {}
        self.nngp_activation_derivative_kernels = {}
        # base case
        self.ntk_kernels[0] = self.nngp_kernels[0]
        self.nngp_activation_derivative_kernels[0] = self._nngp_activation_derivative_kernel_helper(nngp_kernel=self.nngp_kernels[0])
        # Recursive formulation for subsequent layers.
        for l in range(1, L):
            self.ntk_kernels[l] = self.nngp_kernels[l] + self.ntk_kernels[l-1] * self.nngp_activation_derivative_kernels[l-1]
            self.nngp_activation_derivative_kernels[l] = self._nngp_activation_derivative_kernel_helper(nngp_kernel=self.nngp_kernels[l])

    def compute_emp_nngp_kernels(self, model, training_data, epoch):
        model.zero_grad()
        X = training_data.X[training_data.perm_inv].to(self.context["device"])
        _ = model(X)
        # affine feature kernels
        for l in range(self.context["L"]):
            affine_features = model.affine_features[l]
            affine_kernel = affine_features @ affine_features.t()
            if l not in self.emp_nngp_affine_kernels:
                self.emp_nngp_affine_kernels[l] = {}
            self.emp_nngp_affine_kernels[l].update({epoch: affine_kernel})
        # activation feature kernels
        for l in range(self.context["L"]):
            activation_features = model.activation_features[l]
            # features = F.normalize(features, p=2, dim=1)
            activation_kernel = activation_features @ activation_features.t()
            if l not in self.emp_nngp_activation_kernels:
                self.emp_nngp_activation_kernels[l] = {}
            self.emp_nngp_activation_kernels[l].update({epoch: activation_kernel})

    def _get_ntk_feat(self, model_copy):
        device = self.context["device"]
        ntk_feat = []
        for param in model_copy.parameters():
            ntk_feat.append(param.grad.view(-1))
        ntk_feat = torch.cat(ntk_feat).to(device)
        ntk_feat = ntk_feat.unsqueeze(0)
        logger.debug("ntk_feat shape: {}".format(ntk_feat.shape))
        return ntk_feat

    def compute_emp_ntk_kernel(self, model, training_data, epoch):
        assert self.context["batch_size"] == self.context["N"]
        device = self.context["device"]
        model_copy = copy.deepcopy(model)
        ntk_feat_matrix = torch.zeros(0).to(device)
        for data, y, labels in training_data.train_loader:
            data, y, labels = data.to(device), y.to(device), labels.to(device)
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
        features = ntk_feat_matrix[training_data.perm_inv]
        # features = F.normalize(features, p=2, dim=1)
        self.emp_ntk_kernels[epoch] = features @ features.t()
