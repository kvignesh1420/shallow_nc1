import logging

logger = logging.getLogger(__name__)
from collections import OrderedDict
from typing import Dict, Any
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("dark")
import seaborn_image as isns

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.linewidth": 2,
    }
)
from shallow_collapse.probes import WeightProbe
from shallow_collapse.probes import NCProbe
from shallow_collapse.probes import DataNCProbe
from shallow_collapse.probes import KernelProbe
from shallow_collapse.probes import PLACEHOLDER_LAYER_ID


class MetricTracker:
    """
    Track summary metrics based on intermediate features during training/inference.
    Args:
        context: Dictionary of model training parameters
    """

    def __init__(self, context: Dict[str, Any]) -> None:
        self.context = context
        self.epoch_weight_cov = OrderedDict()
        self.epoch_ntk_features_nc_metrics = OrderedDict()
        self.epoch_affine_features_nc_metrics = OrderedDict()
        self.epoch_activation_features_nc_metrics = OrderedDict()
        self.epoch_loss = OrderedDict()
        self.epoch_accuracy = OrderedDict()
        self.initialize_probes()

    def initialize_probes(self):
        self.weight_probe = WeightProbe(context=self.context)
        self.kernel_probe = KernelProbe(context=self.context)
        self.data_nc_probe = DataNCProbe(context=self.context)
        self.affine_features_nc_probe = NCProbe(context=self.context)
        self.activation_features_nc_probe = NCProbe(context=self.context)

    def store_weight_cov(self, model, epoch):
        weight_cov = self.weight_probe.capture(model=model)
        self.epoch_weight_cov[epoch] = weight_cov
        weight_cov_df = pd.DataFrame.from_dict(weight_cov)
        logger.debug(
            "\n cov of layer-wise weights at epoch {}:\n{}".format(epoch, weight_cov_df)
        )

    def plot_weight_cov(self):
        epochs = list(self.epoch_weight_cov.keys())
        for layer_idx in range(self.context["L"]):
            initial_epoch = epochs[0]
            final_epoch = epochs[-1]
            initial_cov = self.epoch_weight_cov[initial_epoch][layer_idx]["cov"]
            initial_S = torch.linalg.svdvals(initial_cov)
            initial_S /= torch.max(initial_S)

            final_cov = self.epoch_weight_cov[final_epoch][layer_idx]["cov"]
            final_S = torch.linalg.svdvals(final_cov)
            final_S /= torch.max(final_S)

            plt.hist(initial_S, bins=100, label="init")
            plt.hist(final_S, bins=100, label="epoch={}".format(final_epoch), alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                "{}weight_cov_hist_layer{}.jpg".format(
                    self.context["vis_dir"], layer_idx
                )
            )
            plt.clf()

            plt.plot(initial_S, label="init")
            plt.plot(final_S, label="epoch={}".format(final_epoch))
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                "{}weight_cov_plot_layer{}.jpg".format(
                    self.context["vis_dir"], layer_idx
                )
            )
            plt.clf()

            isns.imgplot(
                final_cov - initial_cov, cmap="viridis", cbar=True, showticks=True
            )
            plt.savefig(
                "{}weight_cov_diff_layer{}.jpg".format(
                    self.context["vis_dir"], layer_idx
                )
            )
            plt.clf()

    def store_data_nc_metrics(self, training_data):
        self.data_nc_metrics = self.data_nc_probe.capture(training_data=training_data)
        data_nc_df = pd.DataFrame.from_dict(
            [self.data_nc_metrics[PLACEHOLDER_LAYER_ID]]
        )
        logger.info("\nNC metrics of data:\n{}".format(data_nc_df))

    def store_activation_features_nc_metrics(self, model, training_data, epoch):
        activation_features_nc_metrics = self.activation_features_nc_probe.capture(
            model=model, training_data=training_data, layer_type="activation"
        )
        activation_features_nc_df = pd.DataFrame.from_dict(
            activation_features_nc_metrics
        )
        logger.debug(
            "\nmetrics of layer-wise activation features at epoch {}:\n{}".format(
                epoch, activation_features_nc_df
            )
        )
        self.epoch_activation_features_nc_metrics[epoch] = (
            activation_features_nc_metrics
        )

    def plot_activation_features_nc_metrics(self):
        self._plot_layerwise_activation_features_nc_metrics()

    def _plot_layerwise_activation_features_nc_metrics(self):
        epochs = list(self.epoch_activation_features_nc_metrics.keys())
        for layer_idx in range(self.context["L"]):
            values = []
            for epoch in epochs:
                value = self.epoch_activation_features_nc_metrics[epoch][layer_idx]
                values.append(value)

            df = pd.DataFrame(values, index=epochs).astype(float)
            df = df["trace_S_W_div_S_B"].map(lambda x: np.log10(x))
            df.plot(grid=True, xlabel="epoch", ylabel="$\log10(NC1(H))$")
            plt.tight_layout()
            plt.savefig(
                "{}activation_features_nc_metrics_layer{}.jpg".format(
                    self.context["vis_dir"], layer_idx
                )
            )
            plt.clf()

    def compute_emp_nngp_nc1_hat_ratio(self):
        data_nc1_hat = self.data_nc_metrics[PLACEHOLDER_LAYER_ID]["trace_S_W_div_S_B"]
        activation_features_nc1_hat = self.epoch_activation_features_nc_metrics[0][0][
            "trace_S_W_div_S_B"
        ]
        nngp_nc1_hat_ratio = activation_features_nc1_hat / data_nc1_hat
        logger.info(
            "activation_features_nc1_hat/data_nc1_hat: {}\n".format(nngp_nc1_hat_ratio)
        )

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

    def compute_lim_kernels_nc1(self, training_data):
        N = self.context["N"]
        lim_kernels_nc1 = {"nngp": [], "ntk": [], "nngp_act": []}
        for name, Ks in [
            ("nngp", self.kernel_probe.nngp_kernels),
            ("ntk", self.kernel_probe.ntk_kernels),
            ("nngp_act", self.kernel_probe.nngp_activation_kernels),
        ]:
            for l, K in Ks.items():
                assert torch.allclose(K, K.t())
                nc1_val = NCProbe.compute_kernel_nc1(
                    K=K, N=N, class_sizes=training_data.class_sizes
                )["nc1"]
                nc1_val = np.log10(nc1_val)
                lim_kernels_nc1[name].append(nc1_val)
        self.lim_kernels_nc1 = lim_kernels_nc1

    def plot_lim_kernels_nc1(self):
        for name in ["nngp", "ntk", "nngp_act"]:
            df = pd.DataFrame(self.lim_kernels_nc1[name]).astype(float)
            logger.info("limiting kernel: {} NC1 values:\n{}".format(name, df))
            df.plot(
                grid=True,
                xlabel="layer idx",
                ylabel="$\log_{10}(Tr(\Sigma_W)/Tr(\Sigma_B))$",
            )
        plt.savefig("{}lim_kernels_nc1.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def plot_lim_nngp_kernels(self):
        """
        Plot the kernels of the lim nngp
        """
        L = self.context["L"]
        for l in range(L):
            nngp_kernel = self.kernel_probe.nngp_kernels[l]
            isns.imgplot(nngp_kernel.cpu(), cmap="viridis", cbar=True, showticks=True)
            plt.savefig("{}lim_nngp_layer{}.jpg".format(self.context["vis_dir"], l))
            plt.clf()

    def plot_lim_nngp_activation_kernels(self):
        """
        Plot the kernels of the lim nngp post-activations
        """
        L = self.context["L"]
        for l in range(L - 1):
            nngp_activation_kernel = self.kernel_probe.nngp_activation_kernels[l]
            isns.imgplot(
                nngp_activation_kernel.cpu(), cmap="viridis", cbar=True, showticks=True
            )
            plt.savefig(
                "{}lim_nngp_activation_layer{}.jpg".format(self.context["vis_dir"], l)
            )
            plt.clf()

    def plot_lim_ntk_kernels(self):
        """
        Plot the kernels of the lim ntk
        """
        L = self.context["L"]
        for l in range(L):
            ntk_kernel = self.kernel_probe.ntk_kernels[l]
            isns.imgplot(ntk_kernel.cpu(), cmap="viridis", cbar=True, showticks=True)
            plt.savefig("{}lim_ntk_layer{}.jpg".format(self.context["vis_dir"], l))
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
            plt.savefig(
                "{}{}".format(
                    self.context["vis_dir"], "lim_kernel_spectrums_layer{}".format(l)
                )
            )
            plt.clf()

    def plot_epoch_diff_kernel(self, epoch_kernels, name):
        epochs = list(epoch_kernels.keys())
        initial_kernel = epoch_kernels[epochs[0]]
        final_kernel = epoch_kernels[epochs[-1]]
        diff_kernel = final_kernel - initial_kernel
        plt.imshow(diff_kernel.cpu(), cmap="viridis")
        plt.colorbar()
        plt.savefig(name)
        plt.clf()


class EoSTracker:
    def __init__(self, context):
        self.context = context
        self.N = self.context["N"]
        self.step_Sigma = OrderedDict()
        self.step_Q1 = OrderedDict()
        self.step_fbar_kernel = OrderedDict()
        self.step_Q1_nc1 = OrderedDict()
        self.step_Q1_nc1_bounds = OrderedDict()
        self.step_fbar_nc1 = OrderedDict()
        self.step_loss = OrderedDict()

    def plot_kernel(self, K, name):
        if isinstance(K, torch.Tensor):
            K = K.detach().numpy()

        isns.imgplot(K, cmap="viridis", cbar=True, showticks=True)
        plt.tight_layout()
        plt.savefig("{}{}_kernel.jpg".format(self.context["vis_dir"], name))
        plt.clf()

    def plot_kernel_sv(self, K, name):
        if isinstance(K, torch.Tensor):
            K = K.detach().numpy()
        _, S, _ = np.linalg.svd(K)
        plt.plot(S)
        plt.tight_layout()
        plt.grid(True)
        plt.savefig("{}{}_kernel_sv.jpg".format(self.context["vis_dir"], name))
        plt.clf()

    def plot_fbar(self, fbar, name):
        plt.plot(fbar.detach().numpy(), marker="o")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig("{}{}.jpg".format(self.context["vis_dir"], name))
        plt.clf()

    def store_Sigma(self, Sigma, step):
        self.step_Sigma[step] = Sigma.detach()

    def plot_initial_final_Sigma_esd(self):
        steps = list(self.step_Sigma.keys())
        initial_step = steps[0]
        final_step = steps[-1]
        initial_Sigma = self.step_Sigma[initial_step]
        final_Sigma = self.step_Sigma[final_step]
        Ui, Si, Vhi = torch.linalg.svd(initial_Sigma, full_matrices=False)
        Uf, Sf, Vhf = torch.linalg.svd(final_Sigma, full_matrices=False)
        plt.hist(Si.detach().numpy(), bins=100, label="initial")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig("{}initial_Sigma_esd.jpg".format(self.context["vis_dir"]))
        plt.clf()
        plt.hist(Sf.detach().numpy(), bins=100, label="step{}".format(final_step))
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig("{}final_Sigma_esd.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def plot_initial_final_Sigma(self):
        steps = list(self.step_Sigma.keys())
        initial_step = steps[0]
        final_step = steps[-1]
        initial_Sigma = self.step_Sigma[initial_step]
        final_Sigma = self.step_Sigma[final_step]
        self.plot_kernel(K=initial_Sigma, name="Sigma_step{}".format(initial_step))
        self.plot_kernel(K=final_Sigma, name="Sigma_step{}".format(final_step))
        self.plot_kernel(K=final_Sigma - initial_Sigma, name="Sigma_diff")

    def plot_Sigma_trace(self):
        steps = list(self.step_Sigma.keys())
        Sigmas = list(self.step_Sigma.values())
        traces = np.array([torch.trace(Sigma) for Sigma in Sigmas])
        plt.plot(steps, traces, marker="o")
        plt.xlabel("steps")
        plt.ylabel("Tr(Sigma)")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig("{}Sigma_traces.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def plot_Sigma_svd(self):
        steps = list(self.step_Sigma.keys())
        Sigmas = list(self.step_Sigma.values())

        initial_step = steps[0]
        final_step = steps[-1]
        initial_cov = Sigmas[0]
        initial_S = torch.linalg.svdvals(initial_cov)
        initial_S /= torch.max(initial_S)

        final_cov = Sigmas[-1]
        final_S = torch.linalg.svdvals(final_cov)
        final_S /= torch.max(final_S)

        plt.hist(initial_S, bins=100, label="init")
        plt.hist(final_S, bins=100, label="final".format(final_step), alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("{}Sigma_svd_hist.jpg".format(self.context["vis_dir"]))
        plt.clf()

        plt.plot(initial_S, label="init".format(initial_step))
        plt.plot(final_S, label="final".format(final_step))
        plt.legend()
        plt.tight_layout()
        plt.savefig("{}Sigma_svd_plot.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def compute_and_store_data_kernel_nc1(self, X_train, class_sizes):
        # NC1 of data
        K = X_train @ X_train.t()
        self.data_nc1 = NCProbe.compute_kernel_nc1(
            K=K.detach(), N=self.N, class_sizes=class_sizes
        )
        logging.info("data nc1: {}".format(self.data_nc1))
        self.plot_kernel(K=K, name="data")

    def compute_and_store_Q1_nc1(self, Q1, class_sizes, step):
        # NC1 of Q1
        self.step_Q1[step] = Q1
        nc1 = NCProbe.compute_kernel_nc1(
            K=Q1.detach(), N=self.N, class_sizes=class_sizes
        )
        self.step_Q1_nc1[step] = nc1
        # Theoretical bounds
        # nc1_bounds = NCProbe.compute_kernel_nc1_bounds(K=Q1.detach(), N=self.N, class_sizes=class_sizes)
        # self.step_Q1_nc1_bounds[step] = nc1_bounds

    def compute_and_store_fbar_kernel_nc1(self, fbar, class_sizes, step):
        # NC1 of fbar kernel
        K = fbar @ fbar.t()
        self.step_fbar_kernel[step] = K
        nc1 = NCProbe.compute_kernel_nc1(
            K=K.detach(), N=self.N, class_sizes=class_sizes
        )
        self.step_fbar_nc1[step] = nc1

    def plot_initial_final_Q1(self):
        steps = list(self.step_Q1.keys())
        initial_step = steps[0]
        final_step = steps[-1]
        initial_Q1 = self.step_Q1[initial_step]
        final_Q1 = self.step_Q1[final_step]
        self.plot_kernel(K=initial_Q1, name="Q1_step{}".format(initial_step))
        self.plot_kernel(K=final_Q1, name="Q1_step{}".format(final_step))
        self.plot_kernel(K=final_Q1 - initial_Q1, name="Q1_diff")

    def plot_initial_final_fbar_kernel(self):
        steps = list(self.step_fbar_kernel.keys())
        initial_step = steps[0]
        final_step = steps[-1]
        initial_fbar_kernel = self.step_fbar_kernel[initial_step]
        final_fbar_kernel = self.step_fbar_kernel[final_step]
        self.plot_kernel(
            K=initial_fbar_kernel, name="fbar_kernel_step{}".format(initial_step)
        )
        self.plot_kernel(
            K=final_fbar_kernel, name="fbar_kernel_step{}".format(final_step)
        )
        self.plot_kernel(
            K=final_fbar_kernel - initial_fbar_kernel, name="fbar_kernel_diff"
        )

    def plot_Q1_nc1(self):
        steps = list(self.step_Q1_nc1.keys())
        values = list(self.step_Q1_nc1.values())
        nc1_values = [value["nc1"] for value in values]
        plt.plot(steps, nc1_values, marker="o", label="trace_S_W_div_S_B")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("NC1")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("{}Q1_nc1.jpg".format(self.context["vis_dir"]))
        plt.clf()

        df = pd.DataFrame(values)
        for column in ["Tr_Sigma_W", "Tr_Sigma_B"]:
            df[column].astype(float).plot(label=column)
        plt.xlabel("step")
        plt.ylabel("Traces")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig("{}Q1_nc1_traces.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def plot_fbar_kernel_nc1(self):
        steps = list(self.step_fbar_nc1.keys())
        values = list(self.step_fbar_nc1.values())
        values = [value["nc1"] for value in values]
        plt.plot(steps, values, marker="o", label="trace_S_W_div_S_B")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("NC1")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("{}fbar_nc1.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def store_loss(self, loss, step):
        self.step_loss[step] = loss

    def plot_step_loss(self):
        steps = list(self.step_loss.keys())
        values = list(self.step_loss.values())
        plt.plot(steps, values, marker="o")
        plt.xlabel("step")
        plt.ylabel("mse")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("{}mse_loss.jpg".format(self.context["vis_dir"]))
        plt.clf()


class BulkBalancedTracker:
    @staticmethod
    def prepare_res_info(IN_FEATURES_LIST, info):
        res_info = []
        for in_features in IN_FEATURES_LIST:
            mean_nc1 = np.mean(info[in_features])
            std_nc1 = np.std(info[in_features])
            # intervals for plotting
            lower_nc1 = mean_nc1 - std_nc1
            upper_nc1 = mean_nc1 + std_nc1
            res_info.append(
                {
                    "d": in_features,
                    "val": mean_nc1,
                    "lower": lower_nc1,
                    "upper": upper_nc1,
                }
            )
        return res_info

    @staticmethod
    def plot_dfs(dfs, N_LIST, name, context):
        fig, ax = plt.subplots()
        for N, df in zip(N_LIST, dfs):
            ax.plot(df["d"], df["val"], marker="o", label=f"N={N}")
            ax.set(xlabel="$d_0$")
            ax.set(ylabel="$\log_{10}(NC1(H))$")
            ax.fill_between(df["d"], df.lower, df.upper, alpha=0.4)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("{}{}".format(context["vis_dir"], name))
        plt.clf()

    @staticmethod
    def plot_rel_dfs(dfs, N_LIST, name, context):
        fig, ax = plt.subplots()
        for N, df in zip(N_LIST, dfs):
            ax.plot(df["d"], df["val"], marker="o", label=f"N={N}")
            ax.set(xlabel="$d_0$")
            ax.set(ylabel="$\log_{10}(NC1(H)/NC1(X))$")
            ax.fill_between(df["d"], df.lower, df.upper, alpha=0.4)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("{}{}".format(context["vis_dir"], name))
        plt.clf()


class BulkImbalancedTracker:
    @staticmethod
    def prepare_res_info(IN_FEATURES_LIST, info):
        res_info = []
        for in_features in IN_FEATURES_LIST:
            mean_nc1 = np.mean(info[in_features])
            std_nc1 = np.std(info[in_features])
            # intervals for plotting
            lower_nc1 = mean_nc1 - std_nc1
            upper_nc1 = mean_nc1 + std_nc1
            res_info.append(
                {
                    "d": in_features,
                    "val": mean_nc1,
                    "lower": lower_nc1,
                    "upper": upper_nc1,
                }
            )
        return res_info

    @staticmethod
    def plot_dfs(dfs, CLASS_SIZES_LIST, name, context):
        fig, ax = plt.subplots()
        for class_sizes, df in zip(CLASS_SIZES_LIST, dfs):
            ax.plot(df["d"], df["val"], marker="o", label=f"$n_c={class_sizes}$")
            ax.set(xlabel="$d_0$")
            ax.set(ylabel="$\log_{10}(NC1(H))$")
            ax.fill_between(df["d"], df.lower, df.upper, alpha=0.4)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("{}{}".format(context["vis_dir"], name))
        plt.clf()

    @staticmethod
    def plot_rel_dfs(dfs, CLASS_SIZES_LIST, name, context):
        fig, ax = plt.subplots()
        for class_sizes, df in zip(CLASS_SIZES_LIST, dfs):
            ax.plot(df["d"], df["val"], marker="o", label=f"$n_c={class_sizes}$")
            ax.set(xlabel="$d_0$")
            ax.set(ylabel="$\log_{10}(NC1(H)/NC1(X))$")
            ax.fill_between(df["d"], df.lower, df.upper, alpha=0.4)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("{}{}".format(context["vis_dir"], name))
        plt.clf()
