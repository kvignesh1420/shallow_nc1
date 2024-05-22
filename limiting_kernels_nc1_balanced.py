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
from shallow_collapse.probes import KernelProbe, NCProbe, DataNCProbe

from shallow_collapse.utils import setup_runtime_context
from shallow_collapse.utils import data_cls_map

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


def main():
    base_context = {
        "training_data_cls": "Gaussian2DNL",
        # note that the mean/std values will be broadcasted across `in_features`
        "class_means": [-6, 6],
        "class_stds": [0.5, 0.5],
        "L": 2,
        "out_features": 1,
        "num_classes" : 2,
        "bias_std": 0,
        "hidden_weight_std": 1,
        "final_weight_std": 1,
        "activation": "erf",
        "use_cache": False
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
    REPEAT = 10

    nngp_act_dfs = []
    nngp_act_rel_dfs = []
    ntk_dfs = []
    ntk_rel_dfs = []
    for N in tqdm(N_LIST):
        nngp_act_nc1_data = defaultdict(list)
        nngp_act_rel_nc1_data = defaultdict(list)

        ntk_nc1_data = defaultdict(list)
        ntk_rel_nc1_data = defaultdict(list)

        for in_features in IN_FEATURES_LIST:
            context["in_features"] = in_features
            context["N"] = N
            context["batch_size"] = N
            C = context["num_classes"]
            context["class_sizes"] = [N//C for _ in range(C)]
            for _ in range(REPEAT):
                training_data = data_cls_map[context["training_data_cls"]](context=context)
                data_nc_probe = DataNCProbe(context=context)
                training_data_nc1 = data_nc_probe.capture(training_data=training_data)[-1]["trace_S_W_div_S_B"]
                kernel_probe = KernelProbe(context=context)
                kernel_probe.compute_lim_nngp_kernels(training_data=training_data)
                kernel_probe.compute_lim_ntk_kernels(training_data=training_data)

                nngp_act_kernels = kernel_probe.nngp_activation_kernels
                # we focus only on L=2 as of now
                assert len(nngp_act_kernels) == 1
                nngp_act_kernel = nngp_act_kernels[0]
                nngp_act_kernel_nc1 = NCProbe.compute_kernel_nc1(K=nngp_act_kernel, N=N, class_sizes=training_data.class_sizes)["nc1"]
                nngp_act_nc1_data[in_features].append(np.log10(nngp_act_kernel_nc1))
                nngp_act_rel_nc1_data[in_features].append(  np.log10( nngp_act_kernel_nc1/(training_data_nc1 + TAU) ) )

                ntk_kernels = kernel_probe.ntk_kernels
                assert len(ntk_kernels) == 2
                # choose the second kernel and the first one is the same as nngp
                ntk_kernel = ntk_kernels[1]
                ntk_kernel_nc1 = NCProbe.compute_kernel_nc1(K=ntk_kernel, N=N, class_sizes=training_data.class_sizes)["nc1"]
                ntk_nc1_data[in_features].append(np.log10(ntk_kernel_nc1))
                ntk_rel_nc1_data[in_features].append(  np.log10( ntk_kernel_nc1/(training_data_nc1 + TAU) ) )

        nngp_act_res_data = prepare_res_data(IN_FEATURES_LIST=IN_FEATURES_LIST, data=nngp_act_nc1_data)
        nngp_act_rel_res_data = prepare_res_data(IN_FEATURES_LIST=IN_FEATURES_LIST, data=nngp_act_rel_nc1_data)
        ntk_res_data = prepare_res_data(IN_FEATURES_LIST=IN_FEATURES_LIST, data=ntk_nc1_data)
        ntk_rel_res_data = prepare_res_data(IN_FEATURES_LIST=IN_FEATURES_LIST, data=ntk_rel_nc1_data)

        nngp_act_df = pd.DataFrame(nngp_act_res_data)
        nngp_act_rel_df = pd.DataFrame(nngp_act_rel_res_data)

        ntk_res_df = pd.DataFrame(ntk_res_data)
        ntk_rel_res_df = pd.DataFrame(ntk_rel_res_data)

        nngp_act_dfs.append(nngp_act_df)
        nngp_act_rel_dfs.append(nngp_act_rel_df)

        ntk_dfs.append(ntk_res_df)
        ntk_rel_dfs.append(ntk_rel_res_df)

    activation = context["activation"]
    fig_L = context["L"]
    fig_mu = abs(context["class_means"][0])
    fig_std = abs(context["class_stds"][0])
    plot_dfs(dfs=nngp_act_dfs, N_LIST=N_LIST,
             name="nngp_act_nc1_{}_mu_{}_std_{}_L_{}_C_{}_balanced.jpg".format(activation, fig_mu, fig_std, fig_L, C), context=context)
    plot_rel_dfs(dfs=nngp_act_rel_dfs, N_LIST=N_LIST,
                 name="nngp_act_rel_nc1_{}_mu_{}_std_{}_L_{}_C_{}_balanced.jpg".format(activation, fig_mu, fig_std, fig_L, C), context=context)

    plot_dfs(dfs=ntk_dfs, N_LIST=N_LIST,
             name="ntk_nc1_{}_mu_{}_std_{}_L_{}_C_{}_balanced.jpg".format(activation, fig_mu, fig_std, fig_L, C), context=context)
    plot_rel_dfs(dfs=ntk_rel_dfs, N_LIST=N_LIST,
                 name="ntk_rel_nc1_{}_mu_{}_std_{}_L_{}_C_{}_balanced.jpg".format(activation, fig_mu, fig_std, fig_L, C), context=context)


if __name__ == "__main__":
    main()
