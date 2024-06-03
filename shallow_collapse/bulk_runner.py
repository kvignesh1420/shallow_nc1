import logging

logger = logging.getLogger(__name__)
from collections import defaultdict
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.linewidth": 2,
    }
)
from tqdm import tqdm
from shallow_collapse.eos import EoSSolver
from shallow_collapse.trainer import Trainer
from shallow_collapse.tracker import (
    MetricTracker,
    BulkBalancedTracker,
    BulkImbalancedTracker,
)
from shallow_collapse.probes import DataNCProbe, KernelProbe, NCProbe
from shallow_collapse.model import MLPModel
from shallow_collapse.utils import data_cls_map


def _get_data_nc1(context, training_data):
    data_nc_probe = DataNCProbe(context=context)
    training_data_nc1 = data_nc_probe.capture(training_data=training_data)[-1][
        "trace_S_W_div_S_B"
    ]
    return training_data_nc1


def _get_fcn_nc1(context, training_data):
    model = MLPModel(context=context)
    model = model.to(context["device"])
    tracker = MetricTracker(context=context)
    trainer = Trainer(context=context, tracker=tracker)
    logging.info("Model: {}".format(model))
    trainer.train(model=model, training_data=training_data)

    epochs = list(tracker.epoch_activation_features_nc_metrics.keys())
    L = context["L"]
    # penultimate layer with zero indexing = L-2
    assert len(tracker.epoch_activation_features_nc_metrics[epochs[-1]]) == L
    nc1 = tracker.epoch_activation_features_nc_metrics[epochs[-1]][L - 2][
        "trace_S_W_div_S_B"
    ]
    return nc1


def _get_nngp_nc1(context, training_data):
    kernel_probe = KernelProbe(context=context)
    kernel_probe.compute_lim_nngp_kernels(training_data=training_data)
    nngp_act_kernels = kernel_probe.nngp_activation_kernels
    # we focus only on L=2 as of now
    assert len(nngp_act_kernels) == 1
    nngp_act_kernel = nngp_act_kernels[0]
    nc1 = NCProbe.compute_kernel_nc1(
        K=nngp_act_kernel, N=context["N"], class_sizes=training_data.class_sizes
    )["nc1"]
    return nc1


def _get_ntk_nc1(context, training_data):
    kernel_probe = KernelProbe(context=context)
    kernel_probe.compute_lim_ntk_kernels(training_data=training_data)

    ntk_kernels = kernel_probe.ntk_kernels
    assert len(ntk_kernels) == 2
    # choose the second kernel and the first one is the same as nngp
    ntk_kernel = ntk_kernels[1]
    nc1 = NCProbe.compute_kernel_nc1(
        K=ntk_kernel, N=context["N"], class_sizes=training_data.class_sizes
    )["nc1"]
    return nc1


def _get_ak_nc1(context, training_data):
    solver = EoSSolver(context=context)
    ak_kernel = solver.solve(training_data=training_data)

    nc1 = NCProbe.compute_kernel_nc1(
        K=ak_kernel, N=context["N"], class_sizes=training_data.class_sizes
    )["nc1"]
    return nc1


def get_nc1_helper(context, fn):
    training_data = data_cls_map[context["training_data_cls"]](context=context)
    training_data_nc1 = _get_data_nc1(context=context, training_data=training_data)
    nc1 = fn(context=context, training_data=training_data)
    return (nc1, training_data_nc1)


class _BulkBalancedRunner:
    def __init__(self, context):
        self._context = context
        self.dfs = []
        self.rel_dfs = []
        self.kind = ""

    def get_fig_suffix(self):
        fig_activation = self._context["activation"]
        fig_L = self._context["L"]
        fig_C = self._context["num_classes"]
        fig_h = self._context["hidden_features"]
        fig_mu = abs(self._context["class_means"][0])
        fig_std = abs(self._context["class_stds"][0])
        fig_suffix = "{}_nonlin_{}_h_{}_mu_{}_std_{}_L_{}_C_{}_balanced".format(
            self.kind, fig_activation, fig_h, fig_mu, fig_std, fig_L, fig_C
        )
        return fig_suffix

    def get_nc1(self, context):
        raise NotImplementedError()

    def run(self, IN_FEATURES_LIST, N_LIST, REPEAT, TAU):
        for N in tqdm(N_LIST):
            nc1_info = defaultdict(list)
            rel_nc1_info = defaultdict(list)
            for in_features in IN_FEATURES_LIST:
                context = deepcopy(self._context)
                context["in_features"] = in_features
                context["N"] = N
                context["batch_size"] = N
                C = context["num_classes"]
                context["class_sizes"] = [N // C for _ in range(C)]
                for _ in range(REPEAT):
                    nc1, data_nc1 = self.get_nc1(context=context)
                    if np.isnan(np.log10(nc1)):
                        continue
                    nc1_info[in_features].append(np.log10(nc1))
                    rel_nc1_info[in_features].append(np.log10(nc1 / (data_nc1 + TAU)))

            res_info = BulkBalancedTracker.prepare_res_info(
                IN_FEATURES_LIST=IN_FEATURES_LIST, info=nc1_info
            )
            rel_res_info = BulkBalancedTracker.prepare_res_info(
                IN_FEATURES_LIST=IN_FEATURES_LIST, info=rel_nc1_info
            )

            df = pd.DataFrame(res_info)
            rel_df = pd.DataFrame(rel_res_info)

            self.dfs.append(df)
            self.rel_dfs.append(rel_df)
        self.plot_results(N_LIST=N_LIST)

    def plot_results(self, N_LIST):
        fig_suffix = self.get_fig_suffix()
        BulkBalancedTracker.plot_dfs(
            dfs=self.dfs,
            N_LIST=N_LIST,
            name="nc1_{}.jpg".format(fig_suffix),
            context=self._context,
        )
        BulkBalancedTracker.plot_rel_dfs(
            dfs=self.rel_dfs,
            N_LIST=N_LIST,
            name="rel_nc1_{}.jpg".format(fig_suffix),
            context=self._context,
        )


class _BulkImbalancedRunner:
    def __init__(self, context):
        self._context = context
        self.dfs = []
        self.rel_dfs = []
        self.kind = ""

    def get_fig_suffix(self):
        fig_activation = self._context["activation"]
        fig_L = self._context["L"]
        fig_N = self._context["N"]
        fig_C = self._context["num_classes"]
        fig_h = self._context["hidden_features"]
        fig_mu = abs(self._context["class_means"][0])
        fig_std = abs(self._context["class_stds"][0])

        fig_suffix = (
            "{}_nonlin_{}_h_{}_mu_{}_std_{}_L_{}_N_{}_C_{}_imbalanced.jpg".format(
                self.kind, fig_activation, fig_h, fig_mu, fig_std, fig_L, fig_N, fig_C
            )
        )
        return fig_suffix

    def get_nc1(self, context):
        raise NotImplementedError()

    def run(self, N, CLASS_SIZES_LIST, IN_FEATURES_LIST, REPEAT, TAU):

        for class_sizes in tqdm(CLASS_SIZES_LIST):
            nc1_info = defaultdict(list)
            rel_nc1_info = defaultdict(list)

            for in_features in IN_FEATURES_LIST:
                context = deepcopy(self._context)
                context["N"] = N
                context["batch_size"] = N
                context["in_features"] = in_features
                context["class_sizes"] = class_sizes
                assert sum(class_sizes) == N
                for _ in range(REPEAT):
                    nc1, data_nc1 = self.get_nc1(context=context)
                    if np.isnan(np.log10(nc1)):
                        continue
                    nc1_info[in_features].append(np.log10(nc1))
                    rel_nc1_info[in_features].append(np.log10(nc1 / (data_nc1 + TAU)))

            res_info = BulkImbalancedTracker.prepare_res_info(
                IN_FEATURES_LIST=IN_FEATURES_LIST, info=nc1_info
            )
            rel_res_info = BulkImbalancedTracker.prepare_res_info(
                IN_FEATURES_LIST=IN_FEATURES_LIST, info=rel_nc1_info
            )

            df = pd.DataFrame(res_info)
            rel_df = pd.DataFrame(rel_res_info)

            self.dfs.append(df)
            self.rel_dfs.append(rel_df)
        self.plot_results(CLASS_SIZES_LIST=CLASS_SIZES_LIST)

    def plot_results(self, CLASS_SIZES_LIST):
        fig_suffix = self.get_fig_suffix()
        BulkImbalancedTracker.plot_dfs(
            dfs=self.dfs,
            CLASS_SIZES_LIST=CLASS_SIZES_LIST,
            name="nc1_{}.jpg".format(fig_suffix),
            context=self._context,
        )
        BulkImbalancedTracker.plot_rel_dfs(
            dfs=self.rel_dfs,
            CLASS_SIZES_LIST=CLASS_SIZES_LIST,
            name="rel_nc1_{}.jpg".format(fig_suffix),
            context=self._context,
        )


class BulkBalancedRunnerFCN(_BulkBalancedRunner):
    def __init__(self, context):
        super().__init__(context)
        self.kind = "fcn"

    def get_nc1(self, context):
        return get_nc1_helper(context=context, fn=_get_fcn_nc1)


class BulkImbalancedRunnerFCN(_BulkImbalancedRunner):
    def __init__(self, context):
        super().__init__(context)
        self.kind = "fcn"

    def get_nc1(self, context):
        return get_nc1_helper(context=context, fn=_get_fcn_nc1)


class BulkBalancedRunnerNNGP(_BulkBalancedRunner):
    def __init__(self, context):
        super().__init__(context)
        self.kind = "nngp"

    def get_nc1(self, context):
        return get_nc1_helper(context=context, fn=_get_nngp_nc1)


class BulkImbalancedRunnerNNGP(_BulkImbalancedRunner):
    def __init__(self, context):
        super().__init__(context)
        self.kind = "nngp"

    def get_nc1(self, context):
        return get_nc1_helper(context=context, fn=_get_nngp_nc1)


class BulkBalancedRunnerNTK(_BulkBalancedRunner):
    def __init__(self, context):
        super().__init__(context)
        self.kind = "ntk"

    def get_nc1(self, context):
        return get_nc1_helper(context=context, fn=_get_ntk_nc1)


class BulkImbalancedRunnerNTK(_BulkImbalancedRunner):
    def __init__(self, context):
        super().__init__(context)
        self.kind = "ntk"

    def get_nc1(self, context):
        return get_nc1_helper(context=context, fn=_get_ntk_nc1)


class BulkBalancedRunnerEoS(_BulkBalancedRunner):
    def __init__(self, context):
        super().__init__(context)
        # EoS for adaptive kernels
        self.kind = "ak"

    def get_nc1(self, context):
        return get_nc1_helper(context=context, fn=_get_ak_nc1)


class BulkImbalancedRunnerEoS(_BulkImbalancedRunner):
    def __init__(self, context):
        super().__init__(context)
        # EoS for adaptive kernels
        self.kind = "ak"

    def get_nc1(self, context):
        return get_nc1_helper(context=context, fn=_get_ak_nc1)
