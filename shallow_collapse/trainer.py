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

    def compute_accuracy(self, pred, labels):
        pred = (pred >= 0.5).squeeze().type(torch.int64)
        logging.debug("pred: {}, labels: {}".format(pred, labels))
        acc = torch.mean((pred == labels).type(torch.float))
        return acc

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
        ntk_feat_matrix = torch.empty(0)
        for x_idx in range(self.context["N"]):
            model_copy.zero_grad()
            x = X[x_idx, : ]
            pred = model_copy(x)
            logger.debug(pred.shape)
            pred.backward(retain_graph=True)
            ntk_feat = torch.empty(0)
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

    def train(self, model, training_data, probe_features=False, probe_ntk_features=False):
        N = self.context["N"]
        BATCH_SIZE = self.context["BATCH_SIZE"]
        NUM_EPOCHS = self.context["NUM_EPOCHS"]
        X = training_data.X
        Y = training_data.Y
        loss_criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=0.0002,
            momentum=0,
            weight_decay=5e-4
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
            if epoch%100 == 0:
                loss_value = loss.cpu().detach().numpy()
                logger.debug("epoch: {} loss: {}".format(epoch, loss_value))
                self.tracker.store_loss(loss=loss_value, epoch=epoch)
                # ensure an entire pass over the data before the NC metrics are computed
                with torch.no_grad():
                    model.zero_grad()
                    pred=model(X)
                    accuracy = self.compute_accuracy(pred=pred, labels=training_data.labels)
                    accuracy_value = accuracy.cpu().detach().numpy()
                self.tracker.store_accuracy(accuracy=accuracy_value, epoch=epoch)
                if probe_features:
                    # enable if necessary
                    # self.tracker.compute_pre_activation_collapse_metrics(model=model, training_data=training_data, epoch=epoch)
                    logger.debug("pred shape: {}".format(pred.shape))
                    self.tracker.compute_post_activation_collapse_metrics(model=model, training_data=training_data, epoch=epoch)
                    self.tracker.compute_pred_collapse_metrics(pred=pred, training_data=training_data, epoch=epoch)
                if probe_ntk_features:
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

        pred=model(X)
        accuracy = self.compute_accuracy(pred=pred, labels=training_data.labels)
        logger.info("final accuracy: {}".format(accuracy.cpu().detach().numpy()))
        self.plot_pred(pred[training_data.perm_inv].detach().cpu().numpy())
        self.tracker.plot_loss()
        self.tracker.plot_accuracy()
        if probe_features:
            self.tracker.plot_post_activation_collapse_metrics()
            self.tracker.plot_pred_collapse_metrics()
        if probe_ntk_features:
            self.tracker.plot_ntk_collapse_metrics()
