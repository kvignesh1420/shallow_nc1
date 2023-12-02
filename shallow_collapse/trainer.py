import logging
logger = logging.getLogger(__name__)
from typing import Dict, Any
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from shallow_collapse.utils import MetricTracker
from shallow_collapse.data import Circle2D

class Trainer():
    """
    Model trainer class with custom training loops
    Args:
        context: Dictionary of model training parameters
    """
    def __init__(self, context: Dict[str, Any], tracker: MetricTracker) -> None:
        self.context = context
        self.tracker = tracker

    def plot_pred(self, pred):
        plt.plot(pred)
        plt.savefig("{}pred.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def _compute_kernel_stats_at_init(self, model, training_data):

        self.tracker.plot_empirical_nngp_matrix(model=model, training_data=training_data)
        limiting_nngp_matrix = self.tracker.compute_limiting_nngp_matrix(training_data=training_data)
        self.tracker.plot_limiting_nngp_matrix(training_data=training_data)
        kernel_nc1 = self.tracker.compute_kernel_nc1(K=limiting_nngp_matrix)
        logger.info("Limiting NNGP NC1: {}".format(kernel_nc1))
        if isinstance(training_data, Circle2D):
            self.tracker.plot_limiting_nngp_circlar2d(training_data=training_data)

        limiting_ntk_matrix = self.tracker.compute_limiting_ntk_matrix(training_data=training_data)
        # normalize to unit peak
        limiting_ntk_matrix = limiting_ntk_matrix/torch.max(limiting_ntk_matrix)
        self.tracker.plot_limiting_ntk_matrix(training_data=training_data)
        kernel_nc1 = self.tracker.compute_kernel_nc1(K=limiting_ntk_matrix)
        logger.info("Limiting NTK NC1: {}".format(kernel_nc1))
        if isinstance(training_data, Circle2D):
            self.tracker.plot_limiting_ntk_circular2d(training_data=training_data)

    def forward_pass_at_init(self, model, training_data):
        assert model.pre_activations == {}
        assert model.post_activations == {}
        X = training_data.X
        model.zero_grad()
        _ = model(X)
        assert model.pre_activations != {}
        assert model.post_activations != {}
        self.tracker.compute_data_collapse_metrics(training_data=training_data)
        # enable if necessary
        # self.tracker.compute_pre_activation_collapse_metrics(model=model, training_data=training_data, epoch=0)
        self.tracker.compute_post_activation_collapse_metrics(model=model, training_data=training_data, epoch=0)
        self.tracker.compute_empirical_nngp_nc1_hat_ratio()
        self._compute_kernel_stats_at_init(model=model, training_data=training_data)

    def prepare_ntk_feat_matrix(self, model, training_data):
        X = training_data.X
        model_copy = copy.deepcopy(model)
        ntk_feat_matrix = torch.empty(0).to(self.context["device"])
        for x_idx in range(self.context["N"]):
            model_copy.zero_grad()
            x = X[x_idx, : ]
            pred = model_copy(x)
            logger.debug(pred.shape)
            pred.backward(retain_graph=True)
            ntk_feat = torch.empty(0).to(self.context["device"])
            for layer_idx in range(len(model_copy.hidden_layers)):
                layer_weight_grad = model_copy.hidden_layers[layer_idx].weight.grad.flatten()
                layer_bias_grad = model_copy.hidden_layers[layer_idx].bias.grad.flatten()
                logger.debug("layer: {}, weight grad shape: {}, bias grad shape: {}".format(
                    layer_idx, layer_weight_grad.shape, layer_bias_grad.shape))
                ntk_feat = torch.cat([ntk_feat, layer_weight_grad])
                ntk_feat = torch.cat([ntk_feat, layer_bias_grad])
            logger.debug("ntk_feat shape: {}".format(ntk_feat.shape))
            ntk_feat = ntk_feat.unsqueeze(0)
            ntk_feat_matrix = torch.cat([ntk_feat_matrix, ntk_feat], 0)
        logger.info("ntk_feat_matrix shape: {}".format(ntk_feat_matrix.shape))
        return ntk_feat_matrix

    def probe(self, model, training_data, epoch):
        model.zero_grad()
        X = training_data.X
        pred=model(X)
        if self.context["probe_features"]:
            # enable if necessary
            # self.tracker.compute_pre_activation_collapse_metrics(model=model, training_data=training_data, epoch=epoch)
            logger.debug("pred shape: {}".format(pred.shape))
            self.tracker.compute_post_activation_collapse_metrics(model=model, training_data=training_data, epoch=epoch)
            self.tracker.compute_pred_collapse_metrics(pred=pred, training_data=training_data, epoch=epoch)
        if self.context["probe_ntk_features"]:
            self.ntk_feat_matrix = self.prepare_ntk_feat_matrix(model=model, training_data=training_data)
            self.tracker.compute_ntk_collapse_metrics(
                ntk_feat_matrix=self.ntk_feat_matrix,
                training_data=training_data,
                epoch=epoch
            )
            self.tracker.plot_empirical_ntk_matrix(
                ntk_feat_matrix=self.ntk_feat_matrix,
                training_data=training_data,
                epoch=epoch
            )
            if isinstance(training_data, Circle2D):
                self.tracker.plot_empirical_ntk_matrix(
                    ntk_feat_matrix=self.ntk_feat_matrix,
                    training_data=training_data,
                    epoch=epoch
                )


    def train(self, model, training_data):
        N = self.context["N"]
        BATCH_SIZE = self.context["BATCH_SIZE"]
        NUM_EPOCHS = self.context["NUM_EPOCHS"]
        X = training_data.X
        Y = training_data.Y
        loss_criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=self.context["lr"],
            momentum=self.context["momentum"],
            weight_decay=self.context["weight_decay"]
        )
        num_batches = N//BATCH_SIZE
        logger.debug("Number of batches: {}".format(num_batches))
        for epoch in tqdm(range(NUM_EPOCHS)):
            for batch in range(num_batches):
                model.zero_grad()
                optimizer.zero_grad()
                x = X[ batch*(BATCH_SIZE):(batch+1)*BATCH_SIZE, : ]
                y = Y[ batch*(BATCH_SIZE):(batch+1)*BATCH_SIZE, : ]
                pred = model(x)
                loss = loss_criterion(pred, y)
                loss.backward()
                optimizer.step()
            if epoch%self.context["probing_frequency"] == 0:
                loss_value = loss.cpu().detach().numpy()
                logger.debug("epoch: {} loss: {}".format(epoch, loss_value))
                self.tracker.store_loss(loss=loss_value, epoch=epoch)
                self.probe(model=model, training_data=training_data, epoch=epoch)

        pred=model(X)
        self.plot_pred(pred[training_data.perm_inv].detach().cpu().numpy())
        self.tracker.plot_loss()
        if self.context["probe_features"]:
            self.tracker.plot_post_activation_collapse_metrics()
            self.tracker.plot_pred_collapse_metrics()
        if self.context["probe_ntk_features"]:
            self.tracker.plot_ntk_collapse_metrics()
