"""
Torch implementation of the Adaptive Kernels

Reference: https://www.nature.com/articles/s41467-023-36361-y
"""

from typing import Optional
import logging
import scipy
from tqdm import tqdm
import numpy as np
import torch

from shallow_collapse.tracker import EoSTracker


class EoSSolver:
    def __init__(self, context):
        self.context = context
        self.lightweight = self.context["lightweight"]  # fast run without plots/metrics
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
        diag_K_vector = torch.diag(K)  # returns a vector with diag elements
        scaled_diag_K_vector = torch.pow(
            2 * diag_K_vector + 1, -1 / 2
        )  # elementise pow of -0.5
        scaled_diag_K = torch.diag(
            scaled_diag_K_vector
        )  # convert the vector to a diag matrix
        coeffs = scaled_diag_K @ (2 * K) @ scaled_diag_K
        Q1 = self.siga2 * (2 / torch.pi) * torch.arcsin(coeffs)
        return Q1

    def getfbar(self, y_train: torch.Tensor, Q1: torch.Tensor):
        # y_train is a vector of shape: (N)
        # Q1 is the post-activation kernel of shape: N x N
        fbar = (
            Q1
            @ torch.linalg.inv(
                Q1 + self.sig2 * torch.eye(Q1.shape[0]).to(self.context["device"])
            )
            @ y_train
        )
        fbar = fbar.unsqueeze(-1)
        return fbar

    def get_new_Sigma(
        self,
        Sigma: torch.Tensor,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        annealing_factor: Optional[int] = None,
    ):
        if not Sigma.requires_grad:
            Sigma.requires_grad_(True)
        Sigma.grad = None
        Q1 = self.getQ1(Sigma=Sigma, X=X_train)
        Q1_inert = (
            Q1.clone().detach()
        )  # clone and detach the tensor. Detach automatically sets required_grad = False
        fbar = self.getfbar(y_train=y_train, Q1=Q1_inert)
        A = -self.sig2 * (y_train - fbar) @ (y_train - fbar).t() + torch.linalg.inv(
            Q1_inert
            + self.sig2 * torch.eye(Q1_inert.shape[0]).to(self.context["device"])
        )
        assert A.requires_grad == False
        assert fbar.requires_grad == False

        trace_val = torch.trace(A @ Q1)
        trace_val.backward()
        Sigma_shift = Sigma.grad
        assert Sigma_shift.shape == Sigma.shape

        annealing_factor = self.h if annealing_factor is None else annealing_factor
        Sigma_shift *= (
            1 / annealing_factor
        )  # ratio corresponding to (output_dim/hidden_layer_dim)
        new_Sigma_inv = torch.eye(self.d) * (self.d / self.sigw2) + Sigma_shift
        new_Sigma = torch.linalg.inv(new_Sigma_inv)
        return new_Sigma.detach()

    def root_fn(self, Sigma, X_train, y_train, annealing_factor):
        if isinstance(Sigma, np.ndarray):
            Sigma = torch.Tensor(Sigma)
        new_Sigma = self.get_new_Sigma(
            Sigma=Sigma,
            X_train=X_train,
            y_train=y_train,
            annealing_factor=annealing_factor,
        )
        return (Sigma - new_Sigma).detach().numpy()

    def compute_loss(self, fbar: torch.Tensor, y_train: torch.Tensor):
        tbar = y_train - fbar
        return torch.mean(tbar**2)

    def solve(self, training_data):
        # arrange data into blocks for kernel calculations
        X_train = training_data.X[training_data.perm_inv]
        y_train = training_data.y[training_data.perm_inv]
        print(
            "X_train.shape: {} y_train.shape: {}".format(X_train.shape, y_train.shape)
        )
        if not self.lightweight:
            self.tracker.compute_and_store_data_kernel_nc1(
                X_train=X_train, class_sizes=training_data.class_sizes
            )

        # start with the NNGP limit solution
        # W = torch.randn(self.d, self.h)
        # Sigma =  W @ W.t()/ float(self.d)
        Sigma = self.sigw2 * torch.eye(self.d) / float(self.d)
        Q1 = self.getQ1(Sigma=Sigma, X=X_train)  # shape: N x N

        if not self.lightweight:
            self.tracker.store_Sigma(Sigma=Sigma, step=0)
            self.tracker.compute_and_store_Q1_nc1(
                Q1=Q1, class_sizes=training_data.class_sizes, step=0
            )
            self.tracker.plot_kernel(K=Q1, name="Q1")
            self.tracker.plot_kernel_sv(K=Q1, name="Q1")
            self.tracker.plot_kernel_sv(K=Sigma, name="Sigma")

            fbar = self.getfbar(y_train=y_train, Q1=Q1)
            print("fbar shape: {}".format(fbar.shape))
            self.tracker.compute_and_store_fbar_kernel_nc1(
                fbar=fbar, class_sizes=training_data.class_sizes, step=0
            )
            self.tracker.plot_fbar(fbar=fbar, name="fbar_gp")

            loss = self.compute_loss(fbar=fbar, y_train=y_train)
            self.tracker.store_loss(loss=loss.detach(), step=0)
            logging.info("At GP limit loss: {}".format(loss.detach()))

        for annealing_factor in tqdm(self.context["annealing_factors"]):
            self.optim_step += 1
            assert (
                annealing_factor is not None
            ), "annealing factor cannot be None when using newton-krylov strategy."
            F = lambda inp: self.root_fn(
                Sigma=inp,
                X_train=X_train,
                y_train=y_train,
                annealing_factor=annealing_factor,
            )
            new_Sigma = scipy.optimize.newton_krylov(
                F, Sigma, verbose=False, f_tol=5e-4, maxiter=100, inner_maxiter=100
            )

            if isinstance(new_Sigma, np.ndarray):
                new_Sigma = torch.Tensor(new_Sigma)
            new_Q1 = self.getQ1(Sigma=new_Sigma, X=X_train)
            fbar = self.getfbar(y_train=y_train, Q1=new_Q1)
            # self.tracker.plot_fbar(fbar=fbar, name="fbar_{}".format(self.optim_step))
            Sigma = new_Sigma
            Q1 = new_Q1

            if not self.lightweight:
                self.tracker.store_Sigma(Sigma=new_Sigma, step=self.optim_step)
                self.tracker.compute_and_store_Q1_nc1(
                    Q1=Q1.detach(),
                    class_sizes=training_data.class_sizes,
                    step=self.optim_step,
                )
                self.tracker.compute_and_store_fbar_kernel_nc1(
                    fbar=fbar.detach(),
                    class_sizes=training_data.class_sizes,
                    step=self.optim_step,
                )
                loss = self.compute_loss(fbar=fbar, y_train=y_train)
                # print(loss)
                self.tracker.store_loss(loss=loss, step=self.optim_step)
                self.tracker.plot_step_loss()
                self.tracker.plot_Q1_nc1()
                self.tracker.plot_fbar_kernel_nc1()
                self.tracker.plot_Sigma_trace()

        if not self.lightweight:
            self.tracker.plot_initial_final_Q1()
            self.tracker.plot_initial_final_fbar_kernel()
            self.tracker.plot_fbar(fbar=fbar, name="fbar_final")
            self.tracker.plot_initial_final_Sigma()
            self.tracker.plot_Sigma_svd()

        return Q1
