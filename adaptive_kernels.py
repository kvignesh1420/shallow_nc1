"""
Torch implementation of the Adaptive Kernels

Reference: https://www.nature.com/articles/s41467-023-36361-y
"""

from collections import OrderedDict
import logging

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
torch.manual_seed(4)

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
        self.step_fbar_nc1 = OrderedDict()
        self.step_loss = OrderedDict()

    def plot_kernel(self, K, name):
        plt.imshow(K.detach().numpy(), cmap='viridis')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("{}{}_kernel.jpg".format(self.context["vis_dir"], name))
        plt.clf()

    def plot_fbar(self, fbar, name):
        plt.plot(fbar.detach().numpy(), marker="o")
        plt.tight_layout()
        plt.savefig("{}{}.jpg".format(self.context["vis_dir"], name))
        plt.clf()

    def store_Sigma(self, Sigma, step):
        self.step_Sigma[step] = Sigma

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
        plt.savefig("{}initial_Sigma_esd.jpg".format(self.context["vis_dir"]))
        plt.clf()
        plt.hist(Sf.detach().numpy(), bins=100, label="step{}".format(final_step))
        plt.legend()
        plt.tight_layout()
        plt.savefig("{}final_Sigma_esd.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def plot_initial_final_Sigma(self):
        steps = list(self.step_Sigma.keys())
        initial_step = steps[0]
        final_step = steps[-1]
        initial_Sigma = self.step_Sigma[initial_step]
        final_Sigma = self.step_Sigma[final_step]
        self.plot_kernel(K = initial_Sigma.detach(), name="Sigma_step{}".format(initial_step))
        self.plot_kernel(K = final_Sigma.detach(), name="Sigma_step{}".format(final_step))
        self.plot_kernel(K = (final_Sigma - initial_Sigma).detach(), name="Sigma_diff")

    def plot_Sigma_trace(self):
        steps = list(self.step_Sigma.keys())
        Sigmas = list(self.step_Sigma.values())
        traces = [np.trace(Sigma) for Sigma in Sigmas]
        plt.plot(steps, traces, marker="o")
        plt.xlabel("steps")
        plt.ylabel("Tr(Sigma)")
        plt.tight_layout()
        plt.savefig("{}Sigma_traces.jpg".format(self.context["vis_dir"]))
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
        plt.plot(steps, values,  marker="o", label="trace_S_W_div_S_B")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("NC1")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("{}Q1_nc1.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def plot_fbar_kernel_nc1(self):
        steps = list(self.step_fbar_nc1.keys())
        values = list(self.step_fbar_nc1.values())
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

    def getQ1(self, Sigma: torch.Tensor, X: torch.Tensor):
        # Sigma is of shape: d_0 \times d_0
        # X is of shape: N \times d_0
        K = X @ Sigma @ X.t()
        diag_K_vector = torch.diag(K) # returns a vector with diag elements
        scaled_diag_K_vector = torch.pow(2 * diag_K_vector + 1, -1/2) # elementise pow of -0.5
        scaled_diag_K = torch.diag(scaled_diag_K_vector) # convert the vector to a diag matrix
        coeffs = scaled_diag_K @ (2*K) @ scaled_diag_K
        Q1 = self.siga2 * (2/torch.pi) * torch.arcsin(coeffs)
        return Q1

    def getfbar(self, y_train: torch.Tensor, Q1: torch.Tensor):
        # y_train is a vector of shape: (N)
        # Q1 is the post-activation kernel of shape: N x N
        fbar = Q1 @ torch.linalg.inv(Q1 + self.sig2*torch.eye(Q1.shape[0])) @ y_train
        fbar = fbar.unsqueeze(-1)
        return fbar

    def get_new_Sigma(self, Sigma: torch.Tensor, X_train: torch.Tensor, y_train: torch.Tensor, fbar: torch.Tensor):
        if not Sigma.requires_grad:
            Sigma.requires_grad_(True)
        Sigma.grad = None
        Q1 = self.getQ1(Sigma=Sigma, X=X_train)
        Q1_inert = Q1.clone().detach() # clone and detach the tensor. Detach automatically sets required_grad = False
        fbar = self.getfbar(y_train=y_train, Q1=Q1_inert)
        A = - self.sig2 * (y_train - fbar) @ (y_train - fbar).t() + torch.linalg.inv(Q1_inert + self.sig2*torch.eye(Q1_inert.shape[0]))
        assert A.requires_grad == False
        assert fbar.requires_grad == False

        trace_val = torch.trace( A @ Q1 )
        trace_val.backward()
        Sigma_shift = Sigma.grad
        assert Sigma_shift.shape == Sigma.shape

        Sigma_shift *= (1/self.context["h"]) # ratio corresponding to output dim/hidden layer dim
        new_Sigma_inv = torch.eye(self.d)*(self.d/self.sigw2) + Sigma_shift
        new_Sigma = torch.linalg.inv(new_Sigma_inv)
        return new_Sigma

    def solve(self, training_data):
        # arrange data into blocks for kernel calculations
        X_train = training_data.X[training_data.perm_inv]
        y_train = training_data.y[training_data.perm_inv]
        print("X_train.shape: {} y_train.shape: {}".format(X_train.shape, y_train.shape))

        self.tracker.compute_and_store_data_kernel_nc1(X_train=X_train, class_sizes=training_data.class_sizes)

        # start with the NNGP limit solution
        Sigma = self.sigw2*torch.eye(self.d)/float(self.d)
        self.tracker.store_Sigma(Sigma=Sigma, step=0)
        Q1 = self.getQ1(Sigma=Sigma, X=X_train)
        print("Q1 shape: {}".format(Q1.shape))
        self.tracker.compute_and_store_Q1_nc1(Q1=Q1, class_sizes=training_data.class_sizes, step=0)
        fbar = self.getfbar(y_train=y_train, Q1=Q1)
        print("fbar shape: {}".format(fbar.shape))
        self.tracker.compute_and_store_fbar_kernel_nc1(fbar=fbar, class_sizes=training_data.class_sizes, step=0)
        self.tracker.plot_fbar(fbar=fbar, name="fbar_gp")

        loss = torch.sum( (fbar - y_train)**2 )/self.N
        self.tracker.store_loss(loss=loss.detach(), step=0)
        logging.info("At GP limit loss: {}".format(loss.detach()))

        for step in tqdm(range(1, self.context["max_steps"]+1)):
            new_Sigma = self.get_new_Sigma(Sigma=Sigma, X_train=X_train, y_train=y_train, fbar=fbar)
            new_Q1 = self.getQ1(Sigma=new_Sigma, X=X_train)
            fbar = self.getfbar(y_train=y_train, Q1=new_Q1)
            Sigma = new_Sigma
            Q1 = new_Q1

            if step % self.context["probe_freq"] == 0:
                logging.info("probing at step: {}".format(step))
                loss = torch.sum( (fbar - y_train)**2 )/self.N
                self.tracker.store_loss(loss=loss, step=step)

                self.tracker.compute_and_store_Q1_nc1(Q1=Q1.detach(), class_sizes=training_data.class_sizes, step=step)
                self.tracker.compute_and_store_fbar_kernel_nc1(fbar=fbar.detach(), class_sizes=training_data.class_sizes, step=step)
                
                self.tracker.plot_Q1_nc1()
                self.tracker.plot_fbar_kernel_nc1()
                self.tracker.plot_step_loss()

        self.tracker.plot_initial_final_Q1()
        self.tracker.plot_initial_final_fbar_kernel()

        self.tracker.plot_fbar(fbar=fbar, name="fbar_final")
        self.tracker.plot_initial_final_Sigma()
        self.tracker.plot_initial_final_Sigma_esd()
        self.tracker.plot_Sigma_trace()


if __name__ == "__main__":
    exp_context = {
        "name": "adaptive_kernels",
        "training_data_cls": "Gaussian2DNL",
        # note that the mean/std values will be broadcasted across `in_features`
        "class_means": [-2, 2],
        "class_stds": [0.3, 0.3],
        "class_sizes": [512, 512],
        "in_features": 32,
        "num_classes": 2,
        "N": 1024,
        "batch_size": 1024,
        "h": 1024,
        "sigw2": 2,
        "siga2": 2/256,
        "sig2": 0.001,
        "max_steps": 100,
        "probe_freq": 1
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

