import logging

from shallow_collapse.utils import setup_runtime_context, parse_config
from shallow_collapse.bulk_runner import BulkBalancedRunnerFCN


TAU = 1e-8
N_LIST = [128, 256, 512, 1024]
IN_FEATURES_LIST = [1, 2, 8, 32, 128]
REPEAT = 10


def main():
    base_context = {
        "training_data_cls": "Gaussian2DNL",
        # note that the mean/std values will be broadcasted across `in_features`
        "class_means": [-2, 2],
        "class_stds": [0.5, 0.5],
        "num_epochs": 1000,
        "L": 2,
        "out_features": 1,
        "hidden_features": 500,
        "num_classes": 2,
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
        "use_cache": False,
    }
    context = setup_runtime_context(context=base_context)
    logging.basicConfig(
        filename=context["results_file"],
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    logging.info("context: \n{}".format(context))

    bulk_runner = BulkBalancedRunnerFCN(context=context)
    bulk_runner.run(
        IN_FEATURES_LIST=IN_FEATURES_LIST, N_LIST=N_LIST, REPEAT=REPEAT, TAU=TAU
    )


if __name__ == "__main__":
    main()
