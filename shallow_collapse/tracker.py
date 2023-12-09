import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict
from typing import Dict, Any, List, Optional
import pandas as pd
import torch
import matplotlib.pyplot as plt
from shallow_collapse.data import Circle2D
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
        data_nc_df = pd.DataFrame.from_dict(self.data_nc_metrics[PLACEHOLDER_LAYER_ID])
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
        epochs = list(self.epoch_affine_features_nc_metrics.keys())
        for epoch in epochs:
            x = list(self.epoch_affine_features_nc_metrics[epoch].keys())
            values = list(self.epoch_affine_features_nc_metrics[epoch].values())
            df = pd.DataFrame(values, index=x).astype(float)
            logger.info("NC1 metrics for affine features across depth at epoch{}:\n{}".format(epoch, df))
            df.plot(grid=True, xlabel="layer idx", ylabel="NC1 ($\log10$)")
            plt.savefig("{}affine_features_nc_metrics_epoch{}.jpg".format(self.context["vis_dir"], epoch))
            plt.clf()

    def store_activation_features_nc_metrics(self, model, training_data, epoch):
        activation_features_nc_metrics = self.activation_features_nc_probe.capture(model=model, training_data=training_data, layer_type="activation")
        activation_features_nc_df = pd.DataFrame.from_dict(activation_features_nc_metrics)
        logger.debug("\nmetrics of layer-wise activation features at epoch {}:\n{}".format(epoch, activation_features_nc_df))
        self.epoch_activation_features_nc_metrics[epoch] = activation_features_nc_metrics

    def plot_activation_features_nc_metrics(self):
        epochs = list(self.epoch_activation_features_nc_metrics.keys())
        for epoch in epochs:
            x = list(self.epoch_activation_features_nc_metrics[epoch].keys())
            values = list(self.epoch_activation_features_nc_metrics[epoch].values())
            df = pd.DataFrame(values, index=x).astype(float)
            logger.info("NC1 metrics for activation features across depth at epoch{}:\n{}".format(epoch, df))
            df.plot(grid=True, xlabel="layer idx", ylabel="NC1 ($\log10$)")
            plt.savefig("{}activation_features_nc_metrics_epoch{}.jpg".format(self.context["vis_dir"], epoch))
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

    def store_emp_kernels(self, model, training_data):
        self.kernel_probe.compute_emp_nngp_kernels(model=model, training_data=training_data)
        self.kernel_probe.compute_emp_ntk_kernel(model=model, training_data=training_data)

    def compute_lim_kernels_nc1(self, training_data):
        lim_kernel_nc1 = {"nngp": [], "ntk": [], "nngp_relu": []}
        L = self.context["L"]
        N = self.context["N"]
        for l in range(L):
            nngp_kernel = self.kernel_probe.nngp_kernels[l]
            ntk_kernel = self.kernel_probe.ntk_kernels[l]
            nngp_relu_kernel = self.kernel_probe.nngp_relu_kernels[l]
            for name, K in [("nngp", nngp_kernel), ("ntk", ntk_kernel), ("nngp_relu", nngp_relu_kernel)]:
                nc1_val = NCProbe.compute_kernel_nc1(
                    K=K, N=N, class_sizes=training_data.class_sizes)
                lim_kernel_nc1[name].append(nc1_val)
        df = pd.DataFrame(lim_kernel_nc1, index=range(L))
        logger.info("limiting kernel NC1 values:\n{}".format(df))


    def plot_lim_nngp_kernels(self):
        """
        Plot the kernels of the lim nngp
        """
        L = self.context["L"]
        for l in range(L):
            nngp_kernel = self.kernel_probe.nngp_kernels[l]
            plt.imshow(nngp_kernel.cpu(), cmap='viridis')
            plt.colorbar()
            plt.savefig("{}lim_nngp_layer{}.jpg".format(self.context["vis_dir"], l))
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
            plt.savefig("{}lim_ntk_layer{}.jpg".format(self.context["vis_dir"], l))
            plt.clf()

    def plot_lim_kernels_circle2d(self, training_data):
        """
        Use only for cicular data
        """
        if not isinstance(training_data, Circle2D):
            return
        N=self.context["N"]
        angles = training_data.thetas[training_data.perm_inv]
        L = self.context["L"]
        for l in range(L):
            nngp_kernel = self.kernel_probe.nngp_kernels[l]
            ntk_kernel = self.kernel_probe.ntk_kernels[l]
            for K, label in [(nngp_kernel, "nngp"), (ntk_kernel, "ntk")]:
                sim = K[N//2]
                plt.plot(angles, sim, label=label)
            plt.xlabel("angle (x,x')")
            plt.ylabel("K(.,.)")
            plt.grid()
            plt.legend()
            plt.savefig("{}lim_kernels_circle2d_layer{}.jpg".format(self.context["vis_dir"], l))
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
            plt.savefig("{}{}".format(self.context["vis_dir"], "lim_kernel_spectrums_layer{}".format(l)))
            plt.clf()

    def plot_emp_nngp_kernels(self):
        L = self.context["L"]
        for l in range(L):
            emp_nngp_affine_kernel = self.kernel_probe.emp_nngp_affine_kernels[l]
            plt.imshow(emp_nngp_affine_kernel.cpu(), cmap='viridis')
            plt.colorbar()
            plt.savefig("{}emp_nngp_affine_kernel_layer{}.jpg".format(self.context["vis_dir"], l))
            plt.clf()
            emp_nngp_activation_kernel = self.kernel_probe.emp_nngp_activation_kernels[l]
            plt.imshow(emp_nngp_activation_kernel.cpu(), cmap='viridis')
            plt.colorbar()
            plt.savefig("{}emp_nngp_activation_kernel_layer{}.jpg".format(self.context["vis_dir"], l))
            plt.clf()

    def plot_emp_ntk_kernel(self):
        emp_ntk_kernel = self.kernel_probe.emp_ntk_kernel
        plt.imshow(emp_ntk_kernel.cpu(), cmap='viridis')
        plt.colorbar()
        plt.savefig("{}emp_ntk_kernel.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def plot_emp_kernels_circule2d(self, training_data):
        """
        Use only for cicular data
        """
        if not isinstance(training_data, Circle2D):
            return
        N=self.context["N"]
        angles = training_data.thetas[training_data.perm_inv]
        L = self.context["L"]
        for l in range(L):
            emp_nngp_affine_kernel = self.kernel_probe.emp_nngp_affine_kernels[l]
            emp_nngp_activation_kernel = self.kernel_probe.emp_nngp_activation_kernels[l]
            for K, label in [(emp_nngp_affine_kernel, "affine"), (emp_nngp_activation_kernel, "activation")]:
                sim = K[N//2]
                plt.plot(angles, sim, label=label)
            plt.xlabel("angle (x,x')")
            plt.ylabel("emp NNGP")
            plt.grid()
            plt.legend()
            plt.savefig("{}emp_nngp_kernels_circle2d_layer{}.jpg".format(self.context["vis_dir"], l))
            plt.clf()
        emp_ntk_kernel = self.kernel_probe.emp_ntk_kernel
        sim = emp_ntk_kernel[N//2]
        plt.plot(angles, sim)
        plt.xlabel("angle (x,x')")
        plt.ylabel("emp NTK")
        plt.grid()
        plt.savefig("{}emp_ntk_kernel_circle2d.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def plot_emp_kernel_spectrums(self):
        N=self.context["N"]
        L = self.context["L"]
        for l in range(L):
            emp_nngp_affine_kernel = self.kernel_probe.emp_nngp_affine_kernels[l]
            emp_nngp_activation_kernel = self.kernel_probe.emp_nngp_activation_kernels[l]
            for K, label in [(emp_nngp_affine_kernel, "affine"), (emp_nngp_activation_kernel, "activation")]:
                S = torch.linalg.svdvals(K)
                log10_S = torch.log10(S)
                plt.plot(log10_S.cpu(), label=label)
            plt.xlabel("k")
            plt.ylabel("$\log_{10}(\lambda_k)$")
            plt.grid()
            plt.legend()
            plt.savefig("{}{}".format(self.context["vis_dir"], "emp_nngp_kernel_spectrums_layer{}".format(l)))
            plt.clf()
        emp_ntk_kernel = self.kernel_probe.emp_ntk_kernel
        S = torch.linalg.svdvals(emp_ntk_kernel)
        log10_S = torch.log10(S)
        plt.plot(log10_S.cpu())
        plt.xlabel("k")
        plt.ylabel("$\log_10(\lambda_k)$")
        plt.grid()
        plt.savefig("{}{}".format(self.context["vis_dir"], "emp_ntk_kernel_spectrum"))
        plt.clf()

