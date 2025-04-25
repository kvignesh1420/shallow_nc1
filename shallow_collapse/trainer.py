import logging

logger = logging.getLogger(__name__)
from typing import Dict, Any
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.linewidth": 2,
        "axes.labelsize": 18,
    }
)
from tqdm import tqdm
from shallow_collapse.tracker import MetricTracker


class Trainer:
    """
    Model trainer class with custom training loops
    Args:
        context: Dictionary of model training parameters
    """

    def __init__(self, context: Dict[str, Any], tracker: MetricTracker) -> None:
        self.context = context
        self.tracker = tracker

    def apply_tracker(self, model, training_data, loss, epoch):
        if epoch == 0:
            self.tracker.store_data_nc_metrics(training_data=training_data)
        if loss is not None:
            loss_value = loss.cpu().detach().numpy()
            logger.debug("epoch: {} loss: {}".format(epoch, loss_value))
            self.tracker.store_loss(loss=loss_value, epoch=epoch)
        if self.context["probe_weights"]:
            self.tracker.store_weight_cov(model=model, epoch=epoch)
        if self.context["probe_features"]:
            self.tracker.store_activation_features_nc_metrics(
                model=model, training_data=training_data, epoch=epoch
            )
        if self.context["probe_kernels"] and epoch == 0:
            self.tracker.store_lim_kernels(training_data=training_data)
            self.tracker.compute_lim_kernels_nc1(training_data=training_data)

    def plot_pred(self, model, training_data):
        if self.context["out_features"] == 1:
            pred = model(training_data.X)
            pred = pred[training_data.perm_inv].detach().cpu().numpy()
            sign_pred = np.sign(pred)
            plt.plot(sign_pred)
            plt.xlabel("sample index")
            plt.ylabel("sign pred")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("{}sign_pred.jpg".format(self.context["vis_dir"]))
            plt.clf()

    def plot_results(self, model, training_data):
        self.plot_pred(model=model, training_data=training_data)
        self.tracker.plot_loss()
        if self.context["probe_weights"]:
            self.tracker.plot_weight_cov()
        if self.context["probe_features"]:
            self.tracker.plot_activation_features_nc_metrics()
        if self.context["probe_kernels"]:
            self.tracker.plot_lim_kernels_nc1()
            self.tracker.plot_lim_nngp_kernels()
            self.tracker.plot_lim_nngp_activation_kernels()
            self.tracker.plot_lim_ntk_kernels()
            self.tracker.plot_lim_kernel_spectrums()

    def train(self, model, training_data):
        device = self.context["device"]
        N = self.context["N"]
        batch_size = self.context["batch_size"]
        num_epochs = self.context["num_epochs"]
        loss_criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=self.context["lr"],
            momentum=self.context["momentum"],
            weight_decay=self.context["weight_decay"],
        )
        num_batches = int(math.ceil(N / batch_size))
        logger.debug("Number of batches: {}".format(num_batches))
        # probe at init
        self.apply_tracker(model=model, training_data=training_data, loss=None, epoch=0)
        # loop from 1 as epoch=0 indicates init
        for epoch in tqdm(range(1, num_epochs + 1)):
            for data, y, labels in training_data.train_loader:
                model.zero_grad()
                data, y, labels = data.to(device), y.to(device), labels.to(device)
                if self.context["out_features"] > 1:
                    y = F.one_hot(
                        y.type(torch.int64), num_classes=self.context["num_classes"]
                    )
                    y = y.type(torch.float)
                else:
                    y = y.unsqueeze(1)
                pred = model(data)
                loss = loss_criterion(pred, y)
                loss.backward()
                optimizer.step()
            if epoch % self.context["probing_frequency"] == 0:
                self.apply_tracker(
                    model=model, training_data=training_data, loss=loss, epoch=epoch
                )
        self.plot_results(model=model, training_data=training_data)
        plt.close()
