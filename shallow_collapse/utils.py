import logging
logger = logging.getLogger(__name__)
from typing import Dict, Any, List, Optional
import pandas as pd
import torch
from torch_scatter import scatter

class MetricTracker():
    """
    Track summary metrics based on intermediate features during training/inference.
    Args:
        context: Dictionary of model training parameters
    """
    def __init__(self, context: Dict[str, Any]) -> None:
        self.context = context

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
            logger.info("layer id: {} shape of feat: {}".format(layer_idx, feat.shape))
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

    def compute_data_collapse_metrics(self, training_data):
        self.data_collapse_metrics = self.compute_layerwise_nc1(
            features={-1: training_data.X},
            labels=training_data.labels
        )
        data_collapse_metrics_df = pd.DataFrame.from_dict(self.data_collapse_metrics)
        logger.info("\nmetrics of data:\n{}".format(data_collapse_metrics_df))

    def compute_pre_activation_collapse_metrics(self, model, training_data):
        self.pre_activation_collapse_metrics = self.compute_layerwise_nc1(
            features=model.pre_activations,
            labels=training_data.labels
        )
        pre_activation_collapse_metrics_df = pd.DataFrame.from_dict(self.pre_activation_collapse_metrics)
        logger.info("\nmetrics of layer-wise pre-activations:\n{}".format(pre_activation_collapse_metrics_df))

    def compute_post_activation_collapse_metrics(self, model, training_data):
        self.post_activation_collapse_metrics = self.compute_layerwise_nc1(
            features=model.post_activations,
            labels=training_data.labels
        )
        post_activation_collapse_metrics_df = pd.DataFrame.from_dict(self.post_activation_collapse_metrics)
        logger.info("\nmetrics of layer-wise post-activations:\n{}".format(post_activation_collapse_metrics_df))

    def compute_nngp_nc1_hat_ratio(self):
        data_nc1_hat = self.data_collapse_metrics[-1]["trace_S_W_div_S_B"]
        post_activations_nc1_hat = self.post_activation_collapse_metrics[self.context["L"]-2]["trace_S_W_div_S_B"]
        nngp_nc1_hat_ratio = post_activations_nc1_hat/data_nc1_hat
        logger.info("\npost_activations_nc1_hat/data_nc1_hat: {}\n".format(nngp_nc1_hat_ratio))
