import scipy
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 18,
    'axes.linewidth': 2,
})
import seaborn as sns
sns.set_style("darkgrid")
from shallow_collapse.probes import NCProbe, DataNCProbe

from shallow_collapse.utils import setup_runtime_context
from shallow_collapse.utils import data_cls_map
from adaptive_kernels import EoSSolver

TAU = 1e-8

def prepare_res_data(IN_FEATURES_LIST, data):
    res_data = []
    for in_features in IN_FEATURES_LIST:
        mean_nc1 = np.mean(data[in_features])
        std_nc1 = np.std(data[in_features])
        # intervals for plotting
        lower_nc1 = mean_nc1 - std_nc1
        upper_nc1 = mean_nc1 + std_nc1
        res_data.append({
            "d": in_features,
            "val": mean_nc1,
            "lower": lower_nc1,
            "upper": upper_nc1,
        })
    return res_data

def plot_dfs(dfs, N_LIST, name, context):
    fig, ax = plt.subplots()
    for N, df in zip(N_LIST, dfs):
        ax.plot(df["d"], df["val"], marker="o", label=f"N={N}")
        ax.set(xlabel="$d_0$")
        ax.set(ylabel="$\log_{10}(NC1(H))$")
        ax.fill_between(df["d"], df.lower, df.upper, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("{}{}".format(context["vis_dir"], name))
    plt.clf()

def plot_rel_dfs(dfs, N_LIST, name, context):
    fig, ax = plt.subplots()
    for N, df in zip(N_LIST, dfs):
        ax.plot(df["d"], df["val"], marker="o", label=f"N={N}")
        ax.set(xlabel="$d_0$")
        ax.set(ylabel="$\log_{10}(NC1(H)/NC1(X))$")
        ax.fill_between(df["d"], df.lower, df.upper, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("{}{}".format(context["vis_dir"], name))
    plt.clf()


class NoVisEosSolver(EoSSolver):
    def __init__(self, context):
        super().__init__(context)

    def solve(self, training_data):
        # arrange data into blocks for kernel calculations
        X_train = training_data.X[training_data.perm_inv]
        y_train = training_data.y[training_data.perm_inv]

        Sigma = self.sigw2*torch.eye(self.d)/float(self.d)
        Q1 = self.getQ1(Sigma=Sigma, X=X_train)

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

            new_Q1 = self.getQ1(Sigma=new_Sigma, X=X_train)
            Sigma = new_Sigma
            Q1 = new_Q1
        return Q1


def main():
    base_context = {
        "name": "adaptive_kernels",
        "training_data_cls": "Gaussian2DNL",
        # note that the mean/std values will be broadcasted across `in_features`
        "class_means": [-2, 2],
        "class_stds": [0.5, 0.5],
        "out_features": 1,
        "num_classes" : 2,
        "h": 200,
        "bias_std": 0,
        "hidden_weight_std": 1,
        "final_weight_std": 1,
        "use_cache": False,
        "sigw2": 1,
        "siga2": 1/128,
        "sig2": 1e-6,
        # should be one of "default" of "newton-krylov"
        # if "eos_update_strategy": "default", then annealing factors should be None.
        # Simply use "annealing_factors": [None]*100 where 100 is the number of default eos updates.
        # Else if "eos_update_strategy": "newton-krylov", then annealing factors are required.
        # For ex: "annealing_factors": list(range(100_000, 1_000, -2_000))
        # "eos_update_strategy": "default",
        # "annealing_factors": [None]*10,
        "eos_update_strategy": "newton-krylov",
        "annealing_factors": list(range(100_000, 10_000, -10_000)) + list(range(10_000, 1000, -1000)) + list(range(1_000, 100, -100))
    }
    context = setup_runtime_context(context=base_context)
    logging.basicConfig(
        filename=context["results_file"],
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        level=logging.INFO
    )
    logging.info("context: \n{}".format(context))

    N_LIST = [128, 256, 512, 1024]
    IN_FEATURES_LIST = [1, 2, 8, 32, 128]
    REPEAT = 2

    ak_dfs = []
    ak_rel_dfs = []
    for N in tqdm(N_LIST):
        ak_nc1_data = defaultdict(list)
        ak_rel_nc1_data = defaultdict(list)

        for in_features in IN_FEATURES_LIST:
            context["in_features"] = in_features
            context["N"] = N
            context["batch_size"] = N
            context["class_sizes"] = [N//2, N//2]
            for _ in range(REPEAT):
                training_data = data_cls_map[context["training_data_cls"]](context=context)
                data_nc_probe = DataNCProbe(context=context)
                training_data_nc1 = data_nc_probe.capture(training_data=training_data)[-1]["trace_S_W_div_S_B"]

                solver = NoVisEosSolver(context=context)
                ak_kernel = solver.solve(training_data=training_data)

                ak_kernel_nc1 = NCProbe.compute_kernel_nc1(K=ak_kernel, N=N, class_sizes=training_data.class_sizes)["nc1"]
                ak_nc1_data[in_features].append(np.log10(ak_kernel_nc1))
                ak_rel_nc1_data[in_features].append(  np.log10( ak_kernel_nc1/(training_data_nc1 + TAU) ) )


        ak_res_data = prepare_res_data(IN_FEATURES_LIST=IN_FEATURES_LIST, data=ak_nc1_data)
        ak_rel_res_data = prepare_res_data(IN_FEATURES_LIST=IN_FEATURES_LIST, data=ak_rel_nc1_data)

        ak_df = pd.DataFrame(ak_res_data)
        ak_rel_df = pd.DataFrame(ak_rel_res_data)

        ak_dfs.append(ak_df)
        ak_rel_dfs.append(ak_rel_df)

    fig_h = abs(context["h"])
    fig_mu = abs(context["class_means"][0])
    fig_std = abs(context["class_stds"][0])
    plot_dfs(dfs=ak_dfs, N_LIST=N_LIST, name="ak_nc1_erf_h_{}_mu_{}_std_{}_balanced.jpg".format(fig_h, fig_mu, fig_std), context=context)
    plot_rel_dfs(dfs=ak_rel_dfs, N_LIST=N_LIST, name="ak_rel_nc1_erf_h_{}_mu_{}_std_{}_balanced.jpg".format(fig_h, fig_mu, fig_std), context=context)


if __name__ == "__main__":
    main()
