import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict
from typing import Dict, Any, List, Optional
import pandas as pd
import torch
import torch.nn.functional as F
from torch_scatter import scatter
import matplotlib.pyplot as plt
from shallow_collapse.data import Circle2D

class MetricTracker():
    """
    Track summary metrics based on intermediate features during training/inference.
    Args:
        context: Dictionary of model training parameters
    """
    def __init__(self, context: Dict[str, Any]) -> None:
        self.context = context
        self.epoch_ntk_collapse_metrics = OrderedDict()
        self.epoch_post_activation_collapse_metrics = OrderedDict()
        self.epoch_pred_collapse_metrics = OrderedDict()
        self.epoch_loss = OrderedDict()
        self.epoch_accuracy = OrderedDict()

    @staticmethod
    @torch.no_grad()
    def compute_layerwise_nc1(
        features: Dict[int, torch.Tensor],
        labels: torch.Tensor,
        layer_idx_filter: Optional[List[int]] = None):
        """
        compute the NC1 metric from features matrix H as per the
        associated label matrix Y.
        Args:
            features: Mappings from layer idx to corresponding feature matrix of shape: N x d.
                Here 'N' is the number of data points and 'd' is the dimension of the features.
            labels: 1 D labels tensor
            layer_idx_filter: (optional) compute nc1 only for specific layers.
                Defaults to None, implying the nc1 metrics are computed for
                all layers.
        """
        collapse_metrics = {}
        for layer_idx, feat in features.items():
            if layer_idx_filter is not None and layer_idx not in layer_idx_filter:
                continue
            logger.debug("layer id: {} shape of feat: {}".format(layer_idx, feat.shape))
            class_means = scatter(feat, labels, dim=0, reduce="mean")
            expanded_class_means = torch.index_select(class_means, dim=0, index=labels)
            z = feat - expanded_class_means
            num_data_points = z.shape[0]

            # S_W : d x d
            S_W = z.t() @ z
            S_W /= num_data_points

            global_mean = torch.mean(class_means, dim=0)
            z = class_means - global_mean
            num_classes = class_means.shape[0]

            # S_B : d x d
            S_B = z.t() @ z
            S_B /= num_classes

            try:
                nc1 = torch.trace(S_W @ torch.linalg.pinv(S_B)) / num_classes
            except Exception as e:
                nc1 = torch.Tensor(-1)
                logger.error("Exception raised while computing NC1: {}".format(str(e)))

            nc1_hat = torch.trace(S_W)/torch.trace(S_B)
            collapse_metrics[layer_idx] = {}
            collapse_metrics[layer_idx]["trace_S_W_pinv_S_B"] = nc1.detach().cpu().numpy()
            collapse_metrics[layer_idx]["trace_S_W_div_S_B"] = nc1_hat.detach().cpu().numpy()
            collapse_metrics[layer_idx]["trace_S_W"] = torch.trace(S_W).detach().cpu().numpy()
            collapse_metrics[layer_idx]["trace_S_B"] = torch.trace(S_B).detach().cpu().numpy()
        return collapse_metrics

    def compute_ntk_collapse_metrics(self, training_data, ntk_feat_matrix, epoch):
        layer_idx = -1
        self.ntk_collapse_metrics = self.compute_layerwise_nc1(
            features={layer_idx: ntk_feat_matrix},
            labels=training_data.labels
        )
        ntk_collapse_metrics_df = pd.DataFrame.from_dict(self.ntk_collapse_metrics)
        logger.debug("\nmetrics of empirical NTK at epoch {}:\n{}".format(epoch, ntk_collapse_metrics_df))
        self.epoch_ntk_collapse_metrics[epoch] = self.ntk_collapse_metrics[layer_idx]

    def plot_ntk_collapse_metrics(self):
        x = list(self.epoch_ntk_collapse_metrics.keys())
        values = list(self.epoch_ntk_collapse_metrics.values())
        df = pd.DataFrame(values, index=x).astype(float)
        logger.info("NC1 metrics for empirical ntk across epochs:\n{}".format(df))
        df = df[["trace_S_W_div_S_B", "trace_S_W_pinv_S_B"]]
        df.plot(grid=True, xlabel="epoch", ylabel="NC1")
        plt.savefig("{}ntk_nc_metrics.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def compute_data_collapse_metrics(self, training_data):
        self.data_collapse_metrics = self.compute_layerwise_nc1(
            features={-1: training_data.X},
            labels=training_data.labels
        )
        data_collapse_metrics_df = pd.DataFrame.from_dict(self.data_collapse_metrics)
        logger.info("\nmetrics of data:\n{}".format(data_collapse_metrics_df))

    def compute_pre_activation_collapse_metrics(self, model, training_data, epoch):
        self.pre_activation_collapse_metrics = self.compute_layerwise_nc1(
            features=model.pre_activations,
            labels=training_data.labels
        )
        pre_activation_collapse_metrics_df = pd.DataFrame.from_dict(self.pre_activation_collapse_metrics)
        logger.debug("\nmetrics of layer-wise pre-activations at epoch {}:\n{}".format(epoch, pre_activation_collapse_metrics_df))

    def compute_post_activation_collapse_metrics(self, model, training_data, epoch):
        self.post_activation_collapse_metrics = self.compute_layerwise_nc1(
            features=model.post_activations,
            labels=training_data.labels
        )
        post_activation_collapse_metrics_df = pd.DataFrame.from_dict(self.post_activation_collapse_metrics)
        logger.debug("\nmetrics of layer-wise post-activations at epoch {}:\n{}".format(epoch, post_activation_collapse_metrics_df))
        self.epoch_post_activation_collapse_metrics[epoch] = self.post_activation_collapse_metrics[self.context["L"]-2]

    def plot_post_activation_collapse_metrics(self):
        x = list(self.epoch_post_activation_collapse_metrics.keys())
        values = list(self.epoch_post_activation_collapse_metrics.values())
        df = pd.DataFrame(values, index=x).astype(float)
        logger.info("NC1 metrics for post-activations across epochs:\n{}".format(df))
        df = df[["trace_S_W_div_S_B", "trace_S_W_pinv_S_B"]]
        df.plot(grid=True, xlabel="epoch", ylabel="NC1")
        plt.savefig("{}post_activation_nc_metrics.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def compute_pred_collapse_metrics(self, pred, training_data, epoch):
        layer_idx=-1
        self.pred_collapse_metrics = self.compute_layerwise_nc1(
            features={layer_idx: pred},
            labels=training_data.labels
        )
        pred_collapse_metrics_df = pd.DataFrame.from_dict(self.pred_collapse_metrics)
        logger.debug("\nmetrics of model predictions at epoch {}:\n{}".format(epoch, pred_collapse_metrics_df))
        self.epoch_pred_collapse_metrics[epoch] = self.pred_collapse_metrics[layer_idx]

    def plot_pred_collapse_metrics(self):
        x = list(self.epoch_pred_collapse_metrics.keys())
        values = list(self.epoch_pred_collapse_metrics.values())
        df = pd.DataFrame(values, index=x).astype(float)
        logger.info("NC1 metrics for pred across epochs:\n{}".format(df))
        df = df[["trace_S_W_div_S_B", "trace_S_W_pinv_S_B"]]
        df.plot(grid=True, xlabel="epoch", ylabel="NC1")
        plt.savefig("{}pred_nc_metrics.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def store_loss(self, loss, epoch):
        self.epoch_loss[epoch] = {"loss": loss}

    def plot_loss(self):
        x = list(self.epoch_loss.keys())
        values = list(self.epoch_loss.values())
        df = pd.DataFrame(values, index=x).astype(float)
        logger.info("loss across epochs:\n{}".format(df))
        df.plot(grid=True, xlabel="epoch", ylabel="loss")
        plt.savefig("{}loss.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def store_accuracy(self, accuracy, epoch):
        self.epoch_accuracy[epoch] = {"accuracy": accuracy}

    def plot_accuracy(self):
        x = list(self.epoch_accuracy.keys())
        values = list(self.epoch_accuracy.values())
        df = pd.DataFrame(values, index=x).astype(float)
        logger.info("accuracy across epochs:\n{}".format(df))
        df.plot(grid=True, xlabel="epoch", ylabel="accuracy")
        plt.savefig("{}accuracy.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def plot_kernel_spectrum(self, K, filename):
        """
        Plot the spectrum of the kernel
        """
        S = torch.linalg.svdvals(K)
        log2_S = torch.log2(S)
        plt.plot(log2_S)
        plt.xlabel("k")
        plt.ylabel("$\log_2(\lambda_k)$")
        plt.grid()
        plt.savefig("{}{}".format(self.context["vis_dir"], filename))
        plt.clf()

    def compute_empirical_nngp_nc1_hat_ratio(self):
        data_nc1_hat = self.data_collapse_metrics[-1]["trace_S_W_div_S_B"]
        post_activations_nc1_hat = self.post_activation_collapse_metrics[self.context["L"]-2]["trace_S_W_div_S_B"]
        nngp_nc1_hat_ratio = post_activations_nc1_hat/data_nc1_hat
        logger.info("post_activations_nc1_hat/data_nc1_hat: {}\n".format(nngp_nc1_hat_ratio))

    def compute_kernel_nc1(self, K):
        """
        Make sure that the kernel matrix is ordered in blocks. The nngp
        limiting kernel takes care of it by default. We assume 2 classes.
        """
        N = self.context["N"]
        n = N//2
        Tr_Sigma_W = 0
        Tr_Sigma_B = 0
        block1_sum = torch.sum( K[0:n, 0:n] )
        block2_sum = torch.sum( K[n:2*n, n:2*n] )
        Tr_Sigma_W = (0.5/n)*torch.sum(torch.diag(K)) - (0.5/(n*n))*(block1_sum + block2_sum)
        logger.info("Tr_Sigma_W: {}".format(Tr_Sigma_W))
        Tr_Sigma_B = (0.5/(n*n))*(block1_sum + block2_sum) - (0.25/(n*n))*torch.sum(K)
        logger.info("Tr_Sigma_B: {}".format(Tr_Sigma_B))
        return Tr_Sigma_W/Tr_Sigma_B


    def compute_limiting_nngp_matrix(self, training_data):
        """
        Since we use kaiming_normal init for weights with 2d data and
        standard normal init for bias, we have \sigma_w^2 = 2,
        \sigma_b^2 = 1.
        """
        sigma_b_sq = 1
        sigma_w_sq = 2
        X = training_data.X[training_data.perm_inv]
        self.nngp_layer1 = sigma_b_sq + (sigma_w_sq/self.context["in_features"])*(X @ X.t())
        N = self.context["N"]
        self.nngp_layer1_relu = torch.empty_like(self.nngp_layer1)
        for i in range(N):
            for j in range(N):
                K_ii = self.nngp_layer1[i, i]
                K_ij = self.nngp_layer1[i, j]
                K_jj = self.nngp_layer1[j, j]
                theta = torch.arccos( K_ij/(torch.sqrt(K_ii*K_jj) + 1e-6) )
                val = (torch.sqrt(K_ii*K_jj)*( torch.sin(theta) + (torch.pi - theta)*torch.cos(theta) )) /(2*torch.pi)
                self.nngp_layer1_relu[i,j] = val
        return self.nngp_layer1_relu

    def plot_limiting_nngp_matrix(self, training_data):
        N=self.context["N"]
        limiting_nngp_matrix = self.compute_limiting_nngp_matrix(training_data=training_data)
        plt.imshow(limiting_nngp_matrix, cmap='viridis')
        plt.colorbar()
        plt.savefig("{}limiting_nngp_matrix.jpg".format(self.context["vis_dir"]))
        plt.clf()
        self.plot_kernel_spectrum(K=limiting_nngp_matrix, filename="limiting_nngp_spectrum.jpg")

    def plot_limiting_nngp_circlar2d(self, training_data):
        """
        Use only for cicular data
        """
        assert isinstance(training_data, Circle2D)
        N=self.context["N"]
        angles = training_data.thetas[training_data.perm_inv]
        limiting_nngp_matrix = self.compute_limiting_nngp_matrix(training_data=training_data)
        sim = limiting_nngp_matrix[N//2]
        plt.plot(angles, sim)
        plt.xlabel("angle (x,x')")
        plt.ylabel("Limiting NNGP")
        plt.grid()
        plt.savefig("{}limiting_nngp_angle_sim.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def plot_empirical_nngp_matrix(self, model, training_data):
        features = model.post_activations[self.context["L"]-2]
        features = features[training_data.perm_inv]
        normalized_features = F.normalize(features, p=2, dim=1)
        nngp_matrix = normalized_features @ normalized_features.t()
        plt.imshow(nngp_matrix, cmap='viridis')
        plt.colorbar()
        plt.savefig("{}empirical_nngp_matrix.jpg".format(self.context["vis_dir"]))
        plt.clf()
        self.plot_kernel_spectrum(K=nngp_matrix, filename="empirical_nngp_spectrum.jpg")

    def plot_empirical_nngp_circular2d(self, model, training_data):
        """
        Use only for cicular data
        """
        assert isinstance(training_data, Circle2D)
        N=self.context["N"]
        features = model.post_activations[self.context["L"]-2]
        angles = training_data.thetas[training_data.perm_inv]
        features = features[training_data.perm_inv]
        normalized_features = F.normalize(features, p=2, dim=1)
        nngp_matrix = normalized_features @ normalized_features.t()
        sim = nngp_matrix[N//2]
        plt.plot(angles, sim)
        plt.xlabel("angle (x,x')")
        plt.ylabel("Empirical NNGP")
        plt.grid()
        plt.savefig("{}empirical_nngp_angle_sim.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def compute_limiting_ntk_matrix(self, training_data):
        N = self.context["N"]
        sigma_b_sq = 1
        sigma_w_sq = 2
        self.compute_limiting_nngp_matrix(training_data=training_data)
        nngp_layer2 = sigma_b_sq + sigma_w_sq*self.nngp_layer1_relu
        ntk_layer1 = self.nngp_layer1
        nngp_layer1_relu_derivative = torch.empty_like(self.nngp_layer1)
        for i in range(N):
            for j in range(N):
                K_ii = self.nngp_layer1[i, i]
                K_ij = self.nngp_layer1[i, j]
                K_jj = self.nngp_layer1[j, j]
                theta = torch.arccos( K_ij/(torch.sqrt(K_ii*K_jj) + 1e-6) )
                val = (torch.pi - theta) /(2*torch.pi)
                nngp_layer1_relu_derivative[i,j] = val
        self.ntk_layer2 = nngp_layer2 + ntk_layer1 * nngp_layer1_relu_derivative * sigma_w_sq
        return self.ntk_layer2

    def plot_limiting_ntk_matrix(self, training_data):
        """
        Plot the kernel matrix of the limiting ntk
        """
        limiting_ntk_matrix = self.compute_limiting_ntk_matrix(training_data=training_data)
        # normalize to unit peak
        limiting_ntk_matrix = limiting_ntk_matrix/torch.max(limiting_ntk_matrix)
        plt.imshow(limiting_ntk_matrix, cmap='viridis')
        plt.colorbar()
        plt.savefig("{}limiting_ntk_matrix.jpg".format(self.context["vis_dir"]))
        plt.clf()
        self.plot_kernel_spectrum(K=limiting_ntk_matrix, filename="limiting_ntk_spectrum.jpg")

    def plot_limiting_ntk_circular2d(self, training_data):
        """
        Use only for cicular data
        """
        assert isinstance(training_data, Circle2D)
        N=self.context["N"]
        angles = training_data.thetas[training_data.perm_inv]
        limiting_ntk_matrix = self.compute_limiting_ntk_matrix(training_data=training_data)
        # normalize to unit peak
        limiting_ntk_matrix = limiting_ntk_matrix/torch.max(limiting_ntk_matrix)
        # select the N//2-1 th row as that the thetas defined in the Circular2D data class
        # corresponding to this data point.
        sim = limiting_ntk_matrix[N//2]
        plt.plot(angles, sim)
        plt.xlabel("angle (x,x')")
        plt.ylabel("Limiting NTK")
        plt.grid()
        plt.savefig("{}limiting_ntk_angle_sim.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def plot_empirical_ntk_matrix(self, training_data, ntk_feat_matrix, epoch):
        features = ntk_feat_matrix
        features = features[training_data.perm_inv]
        normalized_features = F.normalize(features, p=2, dim=1)
        empirical_ntk_matrix = normalized_features @ normalized_features.t()
        plt.imshow(empirical_ntk_matrix, cmap='viridis')
        plt.colorbar()
        plt.savefig("{}empirical_ntk_matrix_epoch{}.jpg".format(self.context["vis_dir"], epoch))
        plt.clf()
        self.plot_kernel_spectrum(K=empirical_ntk_matrix, filename="empirical_ntk_spectrum_epoch{}.jpg".format(epoch))

    def plot_empirical_ntk_circular2d(self, training_data, ntk_feat_matrix, epoch):
        """
        Use only for cicular data
        """
        assert isinstance(training_data, Circle2D)
        N=self.context["N"]
        features = ntk_feat_matrix
        angles = training_data.thetas[training_data.perm_inv]
        features = features[training_data.perm_inv]
        normalized_features = F.normalize(features, p=2, dim=1)
        empirical_ntk_matrix = normalized_features @ normalized_features.t()
        sim = empirical_ntk_matrix[N//2]
        plt.plot(angles, sim)
        plt.xlabel("angle (x,x')")
        plt.ylabel("Empirical NTK")
        plt.grid()
        plt.savefig("{}empirical_ntk_angle_sim_epoch{}.jpg".format(self.context["vis_dir"], epoch))
        plt.clf()

