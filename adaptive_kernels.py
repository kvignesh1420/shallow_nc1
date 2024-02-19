"""
Code adopted from: https://github.com/zringel/AdaptiveGPs
Paper: https://www.nature.com/articles/s41467-023-36361-y

"""

from collections import OrderedDict
from functools import partial
import sys
import os
import json
import hashlib
import logging

import scipy
import numpy as np
import matplotlib.pyplot as plt
import torch
# torch.manual_seed(4)
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.config import config
config.update("jax_enable_x64", True)

# key = random.PRNGKey(0)
# x = random.normal(key, (10,))

from shallow_collapse.data import GaussiandD
from shallow_collapse.data import Gaussian2D
from shallow_collapse.probes import NCProbe


data_cls_map = {
    "GaussiandD": GaussiandD,
    "Gaussian2D": Gaussian2D
}

@jit
def getQ1_(Sigma, x, y, siga2): # Post-activation kernel of the hidden layer
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    Sigma = jnp.asarray(Sigma)
    xsy = jnp.einsum('a,ab,b',x,Sigma,y) #x@Sigma@y.T)
    ysy = jnp.einsum('a,ab,b',y,Sigma,y) #y@Sigma@y.T
    xsx = jnp.einsum('a,ab,b',x,Sigma,x) #x@Sigma@x.T
    deno_fac_x = jnp.sqrt(1+2*xsx)
    deno_fac_y = jnp.sqrt(1+2*ysy)
    # return (sigw2)*(2./np.pi)*jnp.arcsin(2*xsy/(deno_fac_x*deno_fac_y))
    return (siga2)*(2./np.pi)*jnp.arcsin(2*xsy/(deno_fac_x*deno_fac_y))

def getQ1(Sigma, x, y, siga2): # Jax wrapper
    x = jnp.array(x)
    y = jnp.array(y.transpose())
    vv = lambda x,y: getQ1_(Sigma=Sigma, x=x, y=y, siga2=siga2) #lambda x, y: jnp.vdot(x, y)  #  ([a], [a]) -> []
    mv = vmap(vv, (0, None), 0)      #  ([b,a], [a]) -> [b]      (b is the mapped axis)
    mm = vmap(mv, (None, 1), 1)      #  ([b,a], [a,c]) -> [b,c]
    vgetQf = jax.jit(mm)
    res = vgetQf(x,y)
    return res

def getfbar(y_train, Qf, sig2):
    Qftilde = Qf + sig2*np.eye(Qf.shape[0])
    fbar = Qf @ np.linalg.inv(Qftilde) @ y_train
    return fbar

@partial(jit, static_argnums=5)
def get_Sigma_shift_potential(Sigma, x, y, Q1_inert, t, N, siga2, sig2, C1, C2): # the shift here refers to the last term on the r.h.s in the EoS for \Sigma^{-1}
    Q1 = getQ1(Sigma=Sigma, x=x, y=y, siga2=siga2)
    tt = t*t.T / (sig2**2)
    KfInv = jax.scipy.linalg.inv(Q1_inert + sig2*np.eye(N))
    return jnp.trace((tt - KfInv)@Q1)/C2

get_Sigma_shift = jax.grad(get_Sigma_shift_potential, argnums=0)


class EoSTracker:
    def __init__(self, context):
        self.context = context
        self.N = self.context["N"]
        self.step_Q1_nc1 = OrderedDict()
        self.step_fbar_nc1 = OrderedDict()
        self.step_loss = OrderedDict()

    def plot_kernel(self, K, name):
        plt.imshow(K, cmap='viridis')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("{}{}_kernel.jpg".format(self.context["vis_dir"], name))
        plt.clf()

    def plot_fbar(self, fbar, fig_name):
        plt.plot(fbar)
        plt.tight_layout()
        plt.savefig("{}{}.jpg".format(self.context["vis_dir"], fig_name))
        plt.clf()

    def compute_and_store_data_kernel_nc1(self, X_train, class_sizes):
        # NC1 of data
        K = X_train @ X_train.transpose()
        self.data_nc1 = NCProbe.compute_kernel_nc1(K=torch.tensor(K), N=self.N, class_sizes=class_sizes)
        logging.info("data nc1: {}".format(self.data_nc1))
        self.plot_kernel(K=K, name="data")

    def compute_and_store_Q1_nc1(self, Q1, class_sizes, step, fig_name=None):
        # NC1 of Q1
        nc1 = NCProbe.compute_kernel_nc1(K=Q1, N=self.N, class_sizes=class_sizes)
        self.step_Q1_nc1[step] = nc1
        if fig_name:
            self.plot_kernel(K=Q1, name=fig_name)

    def compute_and_store_fbar_kernel_nc1(self, fbar, class_sizes, step, fig_name=None):
        # NC1 of fbar kernel
        K = fbar @ fbar.t()
        nc1 = NCProbe.compute_kernel_nc1(K=K, N=self.N, class_sizes=class_sizes)
        self.step_fbar_nc1[step] = nc1
        if fig_name:
            self.plot_kernel(K=K, name=fig_name)

    def plot_Q1_nc1(self):
        steps = list(self.step_Q1_nc1.keys())
        values = list(self.step_Q1_nc1.values())
        plt.plot(steps, values)
        plt.xlabel("step")
        plt.ylabel("NC1 ($\log10$)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("{}Q1_nc1.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def plot_fbar_kernel_nc1(self):
        steps = list(self.step_fbar_nc1.keys())
        values = list(self.step_fbar_nc1.values())
        plt.plot(steps, values)
        plt.xlabel("step")
        plt.ylabel("NC1 ($\log10$)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("{}fbar_nc1.jpg".format(self.context["vis_dir"]))
        plt.clf()

    def store_loss(self, loss, step):
        self.step_loss[step] = loss

    def plot_step_loss(self):
        steps = list(self.step_loss.keys())
        values = list(self.step_loss.values())
        plt.plot(steps, values)
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
        self.sigw2 = self.context["sigw2"]
        self.d = self.context["in_features"]
        self.h = self.context["h"]
        self.N = self.context["N"]
        self.sig2 = self.context["sig2"]
        self.siga2 = self.context["siga2"]
        self.sigw2 = self.context["sigw2"]
        self.ind = np.triu_indices(self.d, k=0)
        self.indK = np.triu_indices(self.N, k=0)
        self.prepare_Sigma0()

    def prepare_Sigma0(self):
        self.Sigma0 = self.sigw2*np.eye(self.d)/float(self.d)

    def get_new_Sigma(self, Sigma, x_train, Q1_inert, t, C1, C2): # yields Sigma of the main text
        Sigma_shift = get_Sigma_shift(Sigma,
                                      x=x_train, y=x_train, Q1_inert=Q1_inert, t=t, N=self.N, siga2=self.siga2, sig2=self.sig2, C1=C1, C2=C2)
        retVal = np.linalg.inv(np.eye(self.d)*self.d/self.sigw2 - (1/self.h)*Sigma_shift)
        return retVal

    # rootEq returns a zero vector when all the EoS are satisfied. \
    # The ras is a big concatenated vector of containing Sigma, Q1.
    def rootEq(self, ras, x_train, y_train, C1, C2):
        Sigma = np.zeros((self.d, self.d))
        indSig = np.triu_indices(self.d, k=0)
        Q1 = np.zeros((x_train.shape[0],x_train.shape[0]))
        indQ1 = np.triu_indices(x_train.shape[0], k=0)

        Sigma[indSig] = ras[0:self.ind[0].shape[0]]
        Sigma = Sigma + Sigma.T - np.diag(np.diag(Sigma))
        Q1[indQ1] = ras[self.ind[0].shape[0]:]
        Q1 = Q1 + Q1.T - np.diag(np.diag(Q1))

        Q1_inert = getQ1(Sigma=Sigma, x=x_train, y=x_train, siga2=self.siga2)
        fbar = getfbar(y_train=y_train, Qf=Q1_inert, sig2=self.sig2)

        # fix logic to compute
        t = y_train - fbar

        retVal = Sigma - self.get_new_Sigma(Sigma=Sigma, x_train=x_train, Q1_inert=Q1_inert, t=t, C1=C1, C2=C2)
        retVal2 = Q1 - Q1_inert

        # KfInv = scipy.linalg.inv(Qf + sig2*np.eye(n_train))

        return np.concatenate([retVal[indSig],retVal2[indQ1]])


    def solve(self, training_data):
        # arrange data into blocks for kernel calculations
        X_train = training_data.X[training_data.perm_inv].detach().cpu().numpy()
        Y_train = training_data.labels[training_data.perm_inv].detach().cpu().numpy()

        self.tracker.compute_and_store_data_kernel_nc1(X_train=X_train, class_sizes=training_data.class_sizes)

        # start with the NNGP limit solution
        Q1 = getQ1(Sigma=self.Sigma0, x=X_train, y=X_train, siga2=self.siga2)
        Q1_np = np.asarray(Q1)
        Q1_torch = torch.from_numpy(Q1_np)
        self.tracker.compute_and_store_Q1_nc1(Q1=Q1_torch, class_sizes=training_data.class_sizes, step=0, fig_name="Q1_gp")
        fbar = getfbar(y_train=Y_train, Qf=Q1, sig2=self.sig2)
        fbar_np = np.asarray(fbar)
        fbar_torch = torch.from_numpy(fbar_np).unsqueeze(-1)
        self.tracker.compute_and_store_fbar_kernel_nc1(fbar=fbar_torch, class_sizes=training_data.class_sizes, step=0, fig_name="fbar_gp")
        self.tracker.plot_fbar(fbar=fbar_torch, fig_name="fbar_gp")

        x_res = [np.concatenate([self.Sigma0[self.ind],Q1[self.indK]])]

        # fluc = np.trace(Q1 - Q1@np.linalg.inv(Q1+np.eye(self.N)*self.sig2)@Q1)/self.N # GP prediction for DNN-ensemble fluctuations

        # logging.info("At GP limit, alpha train is {}".format(np.sum(fbar*Y_train)/np.sum(Y_train*Y_train)))
        loss = np.sum( (fbar - Y_train)**2 )/self.N
        self.tracker.store_loss(loss=loss, step=0)
        logging.info("At GP limit loss: {}".format(loss))
        # logging.info("At GP limit, RMSE/y_std is {}".format(np.sqrt(((self.sig2**2)*np.mean(fbar**2)+fluc)/np.var(Y_train))))


        # scaling annealing schedule. This required fine tuning, as making the gaps too large leads to convergence issues
        # Cs = [100000, 95000, 90000, 85000, 80000, 60000, 40000, 35000, 30000, 28000, 26000, 24000, 20000,15000,10000,8000,7000,6000,5000,4000,3000,2000,1800,1600,1400,1200,1000,900,800,700,650,600,570,530,500,470,430,400,370,350,330,310,290,270,250,230,210,200,190,180,170,160,150,140,130,120,110,100,95,90]
        # Cs = [1_000_000, 900_000, 800_000, 700_000, 600_000, 500_000, 400_000, 300_000, 200_000, 100_000, 97_000, 95000, 92000, 90000, 87000, 85000, 82000, 80000, 70000, 60000, 50000, 40000, 35000, 30000, 28000, 26000, 24000, 22000, 20000]
        Cs = [1_000_000, 900_000, 800_000, 700_000, 600_000, 500_000, 400_000, 300_000, 200_000, 100_000, 97_000, 95000, 92000, 90000, 87000, 85000, 82000, 80000, 70000, 60000, 50000, 40000, 35000, 30000, 28000, 26000, 24000, 22000, 20000, 15000,10000,8000,7000,6000,5000,4000,3000,2000,1800,1600,1400,1200,1000,900,800,700,650,600,570,530,500,470,430,400,370,350,330,310,290,270,250,230,210,200,190,180,170,160,150,140,130,120,110,100,95,90]
        # Cs = [1000,900,800,700,650,600,570,530,500,470,430,400,370,350,330,310,290,270,250,230,210,200,190,180,170,160,150,140,130,120,110,100,95,90, 80, 70, 60, 50, 40, 30, 20, 10]

        x_res = [np.concatenate([self.Sigma0[self.ind],Q1[self.indK]])] # Vectorizing this so it can passed to rootEq

        for idx, C in enumerate(Cs):
            # steps as a proxy for counting state updates
            step = idx + 1

            logging.info('*************************** {} *************************'.format(C))
            # F = lambda x: rootEq(x,X_train,Y_train,C,C)
            F = lambda x: self.rootEq(ras=x, x_train=X_train, y_train=Y_train, C1=C, C2=C)

            x_res += [scipy.optimize.newton_krylov(F, x_res[-1], f_tol=1e-6, verbose=True)]
            # np.savez('FCN_newton_n_'+str(n_train)+'_N_'+str(C)+'_d'+str(d)+'_Nteacher'+'.npz',last_x_res=x_res[-1]) # Saving current results
            Sigma = self.Sigma0*0
            Sigma[self.ind] = x_res[-1][0:self.ind[0].shape[0]]
            Sigma = Sigma + Sigma.T - np.diag(np.diag(Sigma))

            # ## track hidden layer activation kernel
            Q1 = getQ1(Sigma=Sigma, x=X_train, y=X_train, siga2=self.siga2)
            Q1_np = np.asarray(Q1)
            Q1_torch = torch.from_numpy(Q1_np)
            self.tracker.compute_and_store_Q1_nc1(Q1=Q1_torch, class_sizes=training_data.class_sizes, step=step)

            fbar = getfbar(y_train=Y_train, Qf=Q1, sig2=self.sig2)
            fbar_np = np.asarray(fbar)
            fbar_torch = torch.from_numpy(fbar_np).unsqueeze(-1)
            self.tracker.compute_and_store_fbar_kernel_nc1(fbar=fbar_torch, class_sizes=training_data.class_sizes, step=step)
            self.tracker.plot_fbar(fbar=fbar_torch, fig_name="fbar_step{}".format(step))

            # fluc = np.trace(Q1 - Q1@np.linalg.inv(Q1+np.eye(self.N)*self.sig2)@Q1)/self.N

            # logging.info(" At {} channels, alpha is {}".format(C, np.sum(fbar*Y_train)/np.sum(Y_train*Y_train)))
            loss = np.sum( (fbar - Y_train)**2 )/self.N
            self.tracker.store_loss(loss=loss, step=step)
            # logging.info(" At {} channels, RMSE/y_std is {}".format(C, rmse/np.var(Y_train)))

            self.tracker.plot_Q1_nc1()
            self.tracker.plot_fbar_kernel_nc1()
            self.tracker.plot_step_loss()



def prepare_config_hash(context):
    _string_context = json.dumps(context, sort_keys=True).encode("utf-8")
    parsed_context_hash = hashlib.md5(_string_context).hexdigest()
    return parsed_context_hash

def setup_runtime_context(context):
    # create a unique hash for the model
    if context["training_data_cls"] not in data_cls_map:
        sys.exit("Invalid training_data_cls. Choose from {}".format(list(data_cls_map.keys())))
    config_uuid = prepare_config_hash(context=context)
    context["config_uuid"] = config_uuid
    context["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    context["out_dir"] = "out/"
    vis_dir = context["out_dir"] + context["config_uuid"] + "/plots/"
    results_dir = context["out_dir"] + context["config_uuid"] + "/results/"
    results_file = results_dir + "run.txt"
    if not os.path.exists(vis_dir):
        print("Vis folder does not exist. Creating {}".format(vis_dir))
        os.makedirs(vis_dir)
    else:
        print("Vis folder {} already exists!".format(vis_dir))
    if not os.path.exists(results_dir):
        print("Resuls folder does not exist. Creating {}".format(results_dir))
        os.makedirs(results_dir)
    else:
        print("Resuls folder {} already exists!".format(results_dir))
    context["vis_dir"] = vis_dir
    context["results_file"] = results_file

    return context



if __name__ == "__main__":
    exp_context = {
        "name": "adaptive_kernels",
        "training_data_cls": "Gaussian2D",
        # note that the mean/std values will be broadcasted across `in_features`
        "class_means": [-1, 1],
        "class_stds": [0.3, 0.3],
        "class_sizes": [512, 512],
        "in_features": 1,
        "num_classes": 2,
        "N": 1024,
        "batch_size": 1024,
        "h": 512,
        "sigw2": 1./2,
        "siga2": 1./2, 
        "sig2": 0.001,
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

