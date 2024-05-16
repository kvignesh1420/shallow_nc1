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
from shallow_collapse.probes import DataNCProbe
from shallow_collapse.model import MLPModel
from shallow_collapse.tracker import MetricTracker
from shallow_collapse.trainer import Trainer

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
        "class_means": [-2, 2],
        "class_stds": [0.5, 0.5],
        "num_epochs": 1000,
        "L": 3,
        "out_features": 1,
        "hidden_features": 500,
        "num_classes" : 2,
        "use_batch_norm": False,
        "lr": 1e-3,
        "momentum": 0.0,
        "weight_decay": 1e-6,
        "bias_std": 0,
        "hidden_weight_std": 1,
        "final_weight_std": 1.97,
        "activation": "erf",
        "probe_features": True,
        "probe_kernels": False,
        "probe_weights": False,
        "probing_frequency": 1000,
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

    N_LIST = [1024]
    IN_FEATURES_LIST = [128]
    REPEAT = 1

    act_dfs = []
    act_rel_dfs = []
    
    for N in tqdm(N_LIST):
        act_nc1_data = defaultdict(list)
        act_rel_nc1_data = defaultdict(list)

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

                model = MLPModel(context=context)
                # model_path = os.path.join(context["model_dir"], "model.pth")
                # if os.path.exists(model_path) and context.get("use_cache", True):
                #     print("Loading the init state of model from {}".format(model_path))
                #     model.load_state_dict(torch.load(model_path))
                # else:
                #     print("Saving the init state of model to {}".format(model_path))
                #     torch.save(model.state_dict(), model_path)

                model = model.to(context["device"])
                tracker = MetricTracker(context=context)
                trainer = Trainer(context=context, tracker=tracker)
                logging.info("Model: {}".format(model))
                trainer.train(model=model, training_data=training_data)

                epochs = list(tracker.epoch_activation_features_nc_metrics.keys())
                L = context["L"]
                # zero indexed and penultimate layer: so L-2
                assert len(tracker.epoch_activation_features_nc_metrics[epochs[-1]]) == L
                act_nc1 = tracker.epoch_activation_features_nc_metrics[epochs[-1]][L-2]["trace_S_W_div_S_B"]
                if np.isnan(np.log10(act_nc1)):
                    continue

                act_nc1_data[in_features].append(np.log10(act_nc1))
                act_rel_nc1_data[in_features].append(  np.log10( act_nc1/(training_data_nc1 + TAU) ) )

        act_res_data = prepare_res_data(IN_FEATURES_LIST=IN_FEATURES_LIST, data=act_nc1_data)
        act_rel_res_data = prepare_res_data(IN_FEATURES_LIST=IN_FEATURES_LIST, data=act_rel_nc1_data)

        act_df = pd.DataFrame(act_res_data)
        act_rel_df = pd.DataFrame(act_rel_res_data)
     
        act_dfs.append(act_df)
        act_rel_dfs.append(act_rel_df)

    activation = context["activation"]
    fig_L = context["L"]
    fig_h = context["hidden_features"]
    fig_mu = abs(context["class_means"][0])
    fig_std = abs(context["class_stds"][0])
    plot_dfs(dfs=act_dfs, N_LIST=N_LIST,
             name="act_nc1_{}_h_{}_mu_{}_std_{}_L_{}_C_{}_balanced.jpg".format(activation, fig_h, fig_mu, fig_std, fig_L, C), context=context)
    plot_rel_dfs(dfs=act_rel_dfs, N_LIST=N_LIST,
                 name="act_rel_nc1_{}_h_{}_mu_{}_std_{}_L_{}_C_{}_balanced.jpg".format(activation, fig_h, fig_mu, fig_std, fig_L, C), context=context)

if __name__ == "__main__":
    main()
