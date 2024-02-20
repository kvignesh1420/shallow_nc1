import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from shallow_collapse.probes import NCProbe
from shallow_collapse.probes import DataNCProbe
from shallow_collapse.probes import NTKNCProbe
from shallow_collapse.probes import KernelProbe
from shallow_collapse.probes import PLACEHOLDER_LAYER_ID

class MetricTracker():
    """
    Track summary metrics based on intermediate features during training/inference.
    Args:
        context: Dictionary of model training parameters
    """
    def __init__(self, context: Dict[str, Any]) -> None:
        self.context = context
        self.epoch_ntk_features_nc_metrics = OrderedDict()
        self.epoch_affine_features_nc_metrics = OrderedDict()
        self.epoch_activation_features_nc_metrics = OrderedDict()
        self.epoch_loss = OrderedDict()
        self.epoch_accuracy = OrderedDict()
        self.initialize_probes()

    def initialize_probes(self):
        self.kernel_probe = KernelProbe(context=self.context)
        self.data_nc_probe = DataNCProbe(context=self.context)
        self.ntk_features_nc_probe = NTKNCProbe(context=self.context)
        self.affine_features_nc_probe = NCProbe(context=self.context)
        self.activation_features_nc_probe = NCProbe(context=self.context)

    def store_data_nc_metrics(self, training_data):
        self.data_nc_metrics = self.data_nc_probe.capture(training_data=training_data)
        data_nc_df = pd.DataFrame.from_dict([self.data_nc_metrics[PLACEHOLDER_LAYER_ID]])
        logger.info("\nNC metrics of data:\n{}".format(data_nc_df))

    def store_ntk_features_nc_metrics(self, model, training_data, epoch):
        ntk_features_nc_metrics = self.ntk_features_nc_probe.capture(model=model, training_data=training_data)
        ntk_features_nc_df = pd.DataFrame.from_dict(ntk_features_nc_metrics[PLACEHOLDER_LAYER_ID])
        logger.debug("\nNC metrics of NTK features at epoch {}:\n{}".format(epoch, ntk_features_nc_df))
        self.epoch_ntk_features_nc_metrics[epoch] = ntk_features_nc_metrics[PLACEHOLDER_LAYER_ID]

    def plot_ntk_features_nc_metrics(self):
        x = list(self.epoch_ntk_features_nc_metrics.keys())
        values = list(self.epoch_ntk_features_nc_metrics.values())
        df = pd.DataFrame(values, index=x).astype(float)
        logger.info("NC1 metrics for ntk features across epochs:\n{}".format(df))
        df.plot(grid=True, xlabel="epoch", ylabel="NC1 ($\log10$)")
        plt.savefig("{}ntk_features_nc_metrics.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def store_affine_features_nc_metrics(self, model, training_data, epoch):
        affine_features_nc_metrics = self.affine_features_nc_probe.capture(model=model, training_data=training_data, layer_type="affine")
        affine_features_nc_df = pd.DataFrame.from_dict(affine_features_nc_metrics)
        logger.debug("\nmetrics of layer-wise affine features at epoch {}:\n{}".format(epoch, affine_features_nc_df))
        self.epoch_affine_features_nc_metrics[epoch] = affine_features_nc_metrics

    def plot_affine_features_nc_metrics(self):
        self._plot_epochwise_affine_features_nc_metrics()
        self._plot_layerwise_affine_features_nc_metrics()

    def _plot_epochwise_affine_features_nc_metrics(self):
        epochs = list(self.epoch_affine_features_nc_metrics.keys())
        for epoch in epochs:
            x = list(self.epoch_affine_features_nc_metrics[epoch].keys())
            values = list(self.epoch_affine_features_nc_metrics[epoch].values())
            x_with_data = [*x, len(x)]
            values_with_data = [self.data_nc_metrics[PLACEHOLDER_LAYER_ID], *values]
            df = pd.DataFrame(values_with_data, index=x_with_data).astype(float)
            # df = pd.DataFrame(values, index=x).astype(float)
            logger.info("NC1 metrics for affine features across depth at epoch{}:\n{}".format(epoch, df))
            df.plot(grid=True, xlabel="layer idx", ylabel="$\log10$ (NC1)")
            # for k, v in self.data_nc_metrics[PLACEHOLDER_LAYER_ID].items():
                # ax.axhline(v, label=k, linestyle="dashed")
            plt.savefig("{}affine_features_nc_metrics_epoch{}.jpg".format(self.context["vis_dir"], epoch))
            plt.clf()

    def _plot_layerwise_affine_features_nc_metrics(self):
        epochs = list(self.epoch_affine_features_nc_metrics.keys())
        for layer_idx in range(self.context["L"]):
            values = []
            for epoch in epochs:
                value = self.epoch_affine_features_nc_metrics[epoch][layer_idx]
                values.append(value)

            df = pd.DataFrame(values, index=epochs).astype(float)
            df.plot(grid=True, xlabel="epoch", ylabel="$\log10$ (NC1)")
            plt.tight_layout()
            plt.savefig("{}affine_features_nc_metrics_layer{}.jpg".format(self.context["vis_dir"], layer_idx+1))
            plt.clf()

    def store_activation_features_nc_metrics(self, model, training_data, epoch):
        activation_features_nc_metrics = self.activation_features_nc_probe.capture(model=model, training_data=training_data, layer_type="activation")
        activation_features_nc_df = pd.DataFrame.from_dict(activation_features_nc_metrics)
        logger.debug("\nmetrics of layer-wise activation features at epoch {}:\n{}".format(epoch, activation_features_nc_df))
        self.epoch_activation_features_nc_metrics[epoch] = activation_features_nc_metrics

    def plot_activation_features_nc_metrics(self):
        self._plot_epochwise_activation_features_nc_metrics()
        self._plot_layerwise_activation_features_nc_metrics()

    def _plot_epochwise_activation_features_nc_metrics(self):
        epochs = list(self.epoch_activation_features_nc_metrics.keys())
        for epoch in epochs:
            x = list(self.epoch_activation_features_nc_metrics[epoch].keys())
            values = list(self.epoch_activation_features_nc_metrics[epoch].values())
            x_with_data = [*x, len(x)]
            values_with_data = [self.data_nc_metrics[PLACEHOLDER_LAYER_ID], *values]
            df = pd.DataFrame(values_with_data, index=x_with_data).astype(float)
            logger.info("NC1 metrics for activation features across depth at epoch{}:\n{}".format(epoch, df))
            # colors = ["blue", "orange", "green", "red"]
            df.plot(grid=True, xlabel="layer idx", ylabel="$\log10$ (NC1)")
            plt.savefig("{}activation_features_nc_metrics_epoch{}.jpg".format(self.context["vis_dir"], epoch))
            plt.clf()

    def _plot_layerwise_activation_features_nc_metrics(self):
        epochs = list(self.epoch_activation_features_nc_metrics.keys())
        for layer_idx in range(self.context["L"]):
            values = []
            for epoch in epochs:
                value = self.epoch_activation_features_nc_metrics[epoch][layer_idx]
                values.append(value)

            df = pd.DataFrame(values, index=epochs).astype(float)
            df.plot(grid=True, xlabel="epoch", ylabel="$\log10$ (NC1)")
            plt.tight_layout()
            # count layers from 1 as we use 0 for data/input layer
            plt.savefig("{}activation_features_nc_metrics_layer{}.jpg".format(self.context["vis_dir"], layer_idx+1))
            plt.clf()

            # plot non-log values for trace_S_W_div_S_B to compare with adaptive kernels
            df_nl = df["trace_S_W_div_S_B"].map(lambda x: np.power(10.0, x))
            df_nl.plot(grid=True, xlabel="epoch", ylabel="NC1")
            plt.legend()
            plt.tight_layout()
            # count layers from 1 as we use 0 for data/input layer
            plt.savefig("{}non_log_trace_S_W_div_S_B_layer{}.jpg".format(self.context["vis_dir"], layer_idx+1))
            plt.clf()

    def compute_emp_nngp_nc1_hat_ratio(self):
        data_nc1_hat = self.data_nc_metrics[PLACEHOLDER_LAYER_ID]["trace_S_W_div_S_B"]
        activation_features_nc1_hat = self.epoch_activation_features_nc_metrics[0][0]["trace_S_W_div_S_B"]
        nngp_nc1_hat_ratio = activation_features_nc1_hat/data_nc1_hat
        logger.info("activation_features_nc1_hat/data_nc1_hat: {}\n".format(nngp_nc1_hat_ratio))

    def store_loss(self, loss, epoch):
        self.epoch_loss[epoch] = {"loss": loss}

    def plot_loss(self):
        x = list(self.epoch_loss.keys())
        values = list(self.epoch_loss.values())
        if len(values) == 0:
            return
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

    def store_lim_kernels(self, training_data):
        self.kernel_probe.compute_lim_nngp_kernels(training_data=training_data)
        self.kernel_probe.compute_lim_ntk_kernels(training_data=training_data)

    def store_emp_kernels(self, model, training_data, epoch):
        self.kernel_probe.compute_emp_nngp_kernels(model=model, training_data=training_data, epoch=epoch)
        # uncomment if needed
        # self.kernel_probe.compute_emp_ntk_kernel(model=model, training_data=training_data, epoch=epoch)

    def compute_lim_kernels_nc1(self, training_data):
        L = self.context["L"]
        N = self.context["N"]
        lim_kernels_nc1 = {"nngp": [], "ntk": [], "nngp_act": []}
        for l in range(L):
            nngp_kernel = self.kernel_probe.nngp_kernels[l]
            ntk_kernel = self.kernel_probe.ntk_kernels[l]
            nngp_activation_kernel = self.kernel_probe.nngp_activation_kernels[l]
            for name, K in [("nngp", nngp_kernel), ("ntk", ntk_kernel), ("nngp_act", nngp_activation_kernel)]:
                assert torch.allclose(K, K.t())
                nc1_val = NCProbe.compute_kernel_nc1(
                    K=K, N=N, class_sizes=training_data.class_sizes)
                nc1_val = torch.log10(nc1_val)
                lim_kernels_nc1[name].append(nc1_val.cpu().numpy())
        self.lim_kernels_nc1 = lim_kernels_nc1

    def plot_lim_kernels_nc1(self):
        L = self.context["L"]
        df = pd.DataFrame(self.lim_kernels_nc1, index=list(range(L))).astype(float)
        logger.info("limiting kernel NC1 values:\n{}".format(df))
        df.plot(grid=True, xlabel="layer idx", ylabel="$\log_{10}(Tr(\Sigma_W)/Tr(\Sigma_B))$")
        plt.savefig("{}lim_kernels_nc1.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def plot_lim_nngp_kernels(self):
        """
        Plot the kernels of the lim nngp
        """
        L = self.context["L"]
        for l in range(L):
            nngp_kernel = self.kernel_probe.nngp_kernels[l]
            plt.imshow(nngp_kernel.cpu(), cmap='viridis')
            plt.colorbar()
            plt.savefig("{}lim_nngp_layer{}.jpg".format(self.context["vis_dir"], l+1))
            plt.clf()

    def plot_lim_nngp_activation_kernels(self):
        """
        Plot the kernels of the lim nngp post-activations
        """
        L = self.context["L"]
        for l in range(L):
            nngp_activation_kernel = self.kernel_probe.nngp_activation_kernels[l]
            plt.imshow(nngp_activation_kernel.cpu(), cmap='viridis')
            plt.colorbar()
            plt.savefig("{}lim_nngp_activation_layer{}.jpg".format(self.context["vis_dir"], l+1))
            plt.clf()

    def plot_lim_ntk_kernels(self):
        """
        Plot the kernels of the lim ntk
        """
        L = self.context["L"]
        for l in range(L):
            ntk_kernel = self.kernel_probe.ntk_kernels[l]
            plt.imshow(ntk_kernel.cpu(), cmap='viridis')
            plt.colorbar()
            plt.savefig("{}lim_ntk_layer{}.jpg".format(self.context["vis_dir"], l+1))
            plt.clf()

    def plot_lim_kernel_spectrums(self):
        """
        Plot the spectrum of the lim nngp and ntk kernels
        """
        L = self.context["L"]
        for l in range(L):
            nngp_kernel = self.kernel_probe.nngp_kernels[l]
            ntk_kernel = self.kernel_probe.ntk_kernels[l]
            for K, label in [(nngp_kernel, "nngp"), (ntk_kernel, "ntk")]:
                S = torch.linalg.svdvals(K)
                log10_S = torch.log10(S)
                plt.plot(log10_S.cpu(), label=label)
            plt.xlabel("k")
            plt.ylabel("$\log_{10}(\lambda_k)$")
            plt.grid()
            plt.legend()
            plt.savefig("{}{}".format(self.context["vis_dir"], "lim_kernel_spectrums_layer{}".format(l+1)))
            plt.clf()

    def plot_epoch_diff_kernel(self, epoch_kernels, name):
        epochs = list(epoch_kernels.keys())
        initial_kernel = epoch_kernels[epochs[0]]
        final_kernel = epoch_kernels[epochs[-1]]
        diff_kernel = final_kernel - initial_kernel
        plt.imshow(diff_kernel.cpu(), cmap='viridis')
        plt.colorbar()
        plt.savefig(name)
        plt.clf()

    def plot_emp_nngp_kernels(self):
        L = self.context["L"]
        for l in range(L):
            epoch_emp_nngp_affine_kernels = self.kernel_probe.emp_nngp_affine_kernels[l]
            for epoch, K in epoch_emp_nngp_affine_kernels.items():
                plt.imshow(K.cpu(), cmap='viridis')
                plt.colorbar()
                plt.savefig("{}emp_nngp_affine_kernel_layer{}_epoch{}.jpg".format(self.context["vis_dir"], l+1, epoch))
                plt.clf()

            name = "{}emp_nngp_affine_kernel_initial_final_diff_layer{}.jpg".format(self.context["vis_dir"], l+1)
            self.plot_epoch_diff_kernel(epoch_kernels=epoch_emp_nngp_affine_kernels, name=name)

            epoch_emp_nngp_activation_kernels = self.kernel_probe.emp_nngp_activation_kernels[l]
            for epoch, K in epoch_emp_nngp_activation_kernels.items():
                plt.imshow(K.cpu(), cmap='viridis')
                plt.colorbar()
                plt.savefig("{}emp_nngp_activation_kernel_layer{}_epoch{}.jpg".format(self.context["vis_dir"], l+1, epoch))
                plt.clf()

            name = "{}emp_nngp_activation_kernel_initial_final_diff_layer{}.jpg".format(self.context["vis_dir"], l+1)
            self.plot_epoch_diff_kernel(epoch_kernels=epoch_emp_nngp_activation_kernels, name=name)

    def plot_emp_ntk_kernels(self):
        emp_ntk_kernels = self.kernel_probe.emp_ntk_kernels
        for epoch, K in emp_ntk_kernels.items():
            plt.imshow(K.cpu(), cmap='viridis')
            plt.colorbar()
            plt.savefig("{}emp_ntk_kernel_epoch{}.jpg".format(self.context["vis_dir"], epoch))
            plt.clf()

