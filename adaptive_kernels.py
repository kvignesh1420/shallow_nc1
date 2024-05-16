"""
Torch implementation of the Adaptive Kernels

Reference: https://www.nature.com/articles/s41467-023-36361-y
"""

from typing import Optional
from collections import OrderedDict
import logging

import scipy
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
import seaborn_image as isns
plt.rcParams.update({
    'font.size': 15,
    'axes.linewidth': 2,
})
import torch

from shallow_collapse.probes import NCProbe
from shallow_collapse.utils import setup_runtime_context
from shallow_collapse.utils import data_cls_map


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
        _,S,_ = np.linalg.svd(K)
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
        self.plot_kernel(K = initial_Sigma, name="Sigma_step{}".format(initial_step))
        self.plot_kernel(K = final_Sigma, name="Sigma_step{}".format(final_step))
        self.plot_kernel(K = final_Sigma - initial_Sigma, name="Sigma_diff")

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

        plt.hist(initial_S, bins=100,  label="init")
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
        self.data_nc1 = NCProbe.compute_kernel_nc1(K=K.detach(), N=self.N, class_sizes=class_sizes)
        logging.info("data nc1: {}".format(self.data_nc1))
        self.plot_kernel(K=K, name="data")

    def compute_and_store_Q1_nc1(self, Q1, class_sizes, step):
        # NC1 of Q1
        self.step_Q1[step] = Q1
        nc1 = NCProbe.compute_kernel_nc1(K=Q1.detach(), N=self.N, class_sizes=class_sizes)
        self.step_Q1_nc1[step] = nc1
        # Theoretical bounds
        # nc1_bounds = NCProbe.compute_kernel_nc1_bounds(K=Q1.detach(), N=self.N, class_sizes=class_sizes)
        # self.step_Q1_nc1_bounds[step] = nc1_bounds

    def compute_and_store_fbar_kernel_nc1(self, fbar, class_sizes, step):
        # NC1 of fbar kernel
        K = fbar @ fbar.t()
        self.step_fbar_kernel[step] = K
        nc1 = NCProbe.compute_kernel_nc1(K=K.detach(), N=self.N, class_sizes=class_sizes)
        self.step_fbar_nc1[step] = nc1

    def plot_initial_final_Q1(self):
        steps = list(self.step_Q1.keys())
        initial_step = steps[0]
        final_step = steps[-1]
        initial_Q1 = self.step_Q1[initial_step]
        final_Q1 = self.step_Q1[final_step]
        self.plot_kernel(K = initial_Q1, name="Q1_step{}".format(initial_step))
        self.plot_kernel(K = final_Q1, name="Q1_step{}".format(final_step))
        self.plot_kernel(K = final_Q1 - initial_Q1, name="Q1_diff")

    def plot_initial_final_fbar_kernel(self):
        steps = list(self.step_fbar_kernel.keys())
        initial_step = steps[0]
        final_step = steps[-1]
        initial_fbar_kernel = self.step_fbar_kernel[initial_step]
        final_fbar_kernel = self.step_fbar_kernel[final_step]
        self.plot_kernel(K = initial_fbar_kernel, name="fbar_kernel_step{}".format(initial_step))
        self.plot_kernel(K = final_fbar_kernel, name="fbar_kernel_step{}".format(final_step))
        self.plot_kernel(K = final_fbar_kernel - initial_fbar_kernel, name="fbar_kernel_diff")

    def plot_Q1_nc1(self):
        steps = list(self.step_Q1_nc1.keys())
        values = list(self.step_Q1_nc1.values())
        nc1_values = [value["nc1"] for value in values]
        plt.plot(steps, nc1_values,  marker="o", label="trace_S_W_div_S_B")
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


class EoSSolver:
    def __init__(self, context):
        self.context = context
        self.tracker = EoSTracker(context=context)
        self.d = self.context["in_features"]
        self.h = self.context["h"]
        self.N = self.context["N"]
        self.sig2 = self.context["sig2"]
        self.sigw2 = self.context["sigw2"]
        self.siga2 = self.context["siga2"]
        self.optim_step = 0

    def getQ1(self, Sigma: torch.Tensor, X: torch.Tensor):
        # Sigma is of shape: d_0 \times d_0
        # X is of shape: N \times d_0
        K = X @ Sigma.to(self.context["device"]) @ X.t()
        diag_K_vector = torch.diag(K) # returns a vector with diag elements
        scaled_diag_K_vector = torch.pow(2 * diag_K_vector + 1, -1/2) # elementise pow of -0.5
        scaled_diag_K = torch.diag(scaled_diag_K_vector) # convert the vector to a diag matrix
        coeffs = scaled_diag_K @ (2*K) @ scaled_diag_K
        Q1 = self.siga2 * (2/torch.pi) * torch.arcsin(coeffs)
        return Q1

    def getfbar(self, y_train: torch.Tensor, Q1: torch.Tensor):
        # y_train is a vector of shape: (N)
        # Q1 is the post-activation kernel of shape: N x N
        fbar = Q1 @ torch.linalg.inv(Q1 + self.sig2*torch.eye(Q1.shape[0]).to(self.context["device"])) @ y_train
        fbar = fbar.unsqueeze(-1)
        return fbar

    def get_new_Sigma(self, Sigma: torch.Tensor, X_train: torch.Tensor, y_train: torch.Tensor, annealing_factor: Optional[int] = None):
        if not Sigma.requires_grad:
            Sigma.requires_grad_(True)
        Sigma.grad = None
        Q1 = self.getQ1(Sigma=Sigma, X=X_train)
        Q1_inert = Q1.clone().detach() # clone and detach the tensor. Detach automatically sets required_grad = False
        fbar = self.getfbar(y_train=y_train, Q1=Q1_inert)
        A = - self.sig2 * (y_train - fbar) @ (y_train - fbar).t() + torch.linalg.inv(Q1_inert + self.sig2*torch.eye(Q1_inert.shape[0]).to(self.context["device"]))
        assert A.requires_grad == False
        assert fbar.requires_grad == False

        trace_val = torch.trace( A @ Q1 )
        trace_val.backward()
        Sigma_shift = Sigma.grad
        assert Sigma_shift.shape == Sigma.shape

        annealing_factor = self.h if annealing_factor is None else annealing_factor
        Sigma_shift *= (1/annealing_factor) # ratio corresponding to (output_dim/hidden_layer_dim)
        new_Sigma_inv = torch.eye(self.d)*(self.d/self.sigw2) + Sigma_shift
        new_Sigma = torch.linalg.inv(new_Sigma_inv)
        return new_Sigma.detach()

    def root_fn(self, Sigma, X_train, y_train, annealing_factor):
        if isinstance(Sigma, np.ndarray):
            Sigma = torch.Tensor(Sigma)
        new_Sigma = self.get_new_Sigma(Sigma=Sigma, X_train=X_train, y_train=y_train, annealing_factor=annealing_factor)
        return (Sigma - new_Sigma).detach().numpy()

    def compute_loss(self, fbar: torch.Tensor, y_train: torch.Tensor):
        tbar = y_train - fbar
        return torch.mean(tbar**2)

    def solve(self, training_data):
        # arrange data into blocks for kernel calculations
        X_train = training_data.X[training_data.perm_inv]
        y_train = training_data.y[training_data.perm_inv]
        print("X_train.shape: {} y_train.shape: {}".format(X_train.shape, y_train.shape))

        self.tracker.compute_and_store_data_kernel_nc1(X_train=X_train, class_sizes=training_data.class_sizes)

        # start with the NNGP limit solution
        # W = torch.randn(self.d, self.h)
        # Sigma =  W @ W.t()/ float(self.d)
        Sigma = self.sigw2*torch.eye(self.d)/float(self.d)
        self.tracker.store_Sigma(Sigma=Sigma, step=0)
        Q1 = self.getQ1(Sigma=Sigma, X=X_train)
        print("Q1 shape: {}".format(Q1.shape))
        self.tracker.compute_and_store_Q1_nc1(Q1=Q1, class_sizes=training_data.class_sizes, step=0)
        self.tracker.plot_kernel(K=Q1, name="Q1")
        self.tracker.plot_kernel_sv(K=Q1, name="Q1")
        self.tracker.plot_kernel_sv(K=Sigma, name="Sigma")

        fbar = self.getfbar(y_train=y_train, Q1=Q1)
        print("fbar shape: {}".format(fbar.shape))
        self.tracker.compute_and_store_fbar_kernel_nc1(fbar=fbar, class_sizes=training_data.class_sizes, step=0)
        self.tracker.plot_fbar(fbar=fbar, name="fbar_gp")

        loss = self.compute_loss(fbar=fbar, y_train=y_train)
        self.tracker.store_loss(loss=loss.detach(), step=0)
        logging.info("At GP limit loss: {}".format(loss.detach()))

        for annealing_factor in tqdm(self.context["annealing_factors"]):
            self.optim_step += 1
            if self.context["eos_update_strategy"] == "default":
                new_Sigma = self.get_new_Sigma(Sigma=Sigma, X_train=X_train, y_train=y_train, annealing_factor=annealing_factor)
            elif self.context["eos_update_strategy"] == "newton-krylov":
                assert annealing_factor is not None, "annealing factor cannot be None when using newton-krylov strategy."
                F = lambda inp: self.root_fn(Sigma=inp, X_train=X_train, y_train=y_train, annealing_factor=annealing_factor)
                new_Sigma = scipy.optimize.newton_krylov(F, Sigma, verbose=False, f_tol=5e-5)

            if isinstance(new_Sigma, np.ndarray):
                new_Sigma = torch.Tensor(new_Sigma)
            self.tracker.store_Sigma(Sigma=new_Sigma, step=self.optim_step)
            new_Q1 = self.getQ1(Sigma=new_Sigma, X=X_train)
            fbar = self.getfbar(y_train=y_train, Q1=new_Q1)
            # self.tracker.plot_fbar(fbar=fbar, name="fbar_{}".format(self.optim_step))
            Sigma = new_Sigma
            Q1 = new_Q1

            self.tracker.compute_and_store_Q1_nc1(Q1=Q1.detach(), class_sizes=training_data.class_sizes, step=self.optim_step)
            self.tracker.compute_and_store_fbar_kernel_nc1(fbar=fbar.detach(), class_sizes=training_data.class_sizes, step=self.optim_step)
            loss = self.compute_loss(fbar=fbar, y_train=y_train)
            # print(loss)
            self.tracker.store_loss(loss=loss, step=self.optim_step)
            self.tracker.plot_step_loss()
            self.tracker.plot_Q1_nc1()
            self.tracker.plot_fbar_kernel_nc1()
            self.tracker.plot_Sigma_trace()

        self.tracker.plot_initial_final_Q1()
        self.tracker.plot_initial_final_fbar_kernel()
        self.tracker.plot_fbar(fbar=fbar, name="fbar_final")
        self.tracker.plot_initial_final_Sigma()
        self.tracker.plot_Sigma_svd()


if __name__ == "__main__":
    exp_context = {
        "name": "adaptive_kernels",
        "training_data_cls": "Gaussian2DNL",
        # note that the mean/std values will be broadcasted across `in_features`
        "class_means": [-2, 2],
        "class_stds": [2, 2],
        "class_sizes": [512, 512],
        "in_features": 128,
        "num_classes": 2,
        "N": 1024,
        "batch_size": 1024,
        "h": 500,
        "sigw2": 1,
        "siga2": 1/128,
        "sig2": 1e-6,
        # should be one of "default" of "newton-krylov"
        # if "eos_update_strategy": "default", then annealing factors should be None.
        # Simply use "annealing_factors": [None]*100 where 100 is the number of default eos updates.
        # Else if "eos_update_strategy": "newton-krylov", then annealing factors are required.
        # For ex: "annealing_factors": list(range(100_000, 1_000, -2_000))
        # "eos_update_strategy": "default",
        # "annealing_factors": [10**5, 9*10**4],
        "eos_update_strategy": "newton-krylov",
        "annealing_factors": list(range(100_000, 10_000, -10_000)) + list(range(10_000, 1000, -1000)) + list(range(1_000, 400, -100))
    }
    context = setup_runtime_context(context=exp_context)
    logging.basicConfig(
        filename=context["results_file"],
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        level=logging.INFO
    )
    logging.info("context: \n{}".format(context))

    training_data = data_cls_map[context["training_data_cls"]](context=context)
    solver = EoSSolver(context=context)
    solver.solve(training_data=training_data)

