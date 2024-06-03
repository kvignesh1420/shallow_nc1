import logging

from shallow_collapse.bulk_runner import BulkBalancedRunnerNNGP, BulkBalancedRunnerNTK
from shallow_collapse.utils import setup_runtime_context, parse_config

N_LIST = [128, 256, 512, 1024]
IN_FEATURES_LIST = [1, 2, 8, 32, 128]
REPEAT = 10
TAU = 1e-8


def main():
    base_context = parse_config()
    context = setup_runtime_context(context=base_context)
    logging.basicConfig(
        filename=context["results_file"],
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    logging.info("context: \n{}".format(context))

    for bulk_runner_cls in [BulkBalancedRunnerNNGP, BulkBalancedRunnerNTK]:
        bulk_runner = bulk_runner_cls(context=context)
        bulk_runner.run(
            IN_FEATURES_LIST=IN_FEATURES_LIST, N_LIST=N_LIST, REPEAT=REPEAT, TAU=TAU
        )


if __name__ == "__main__":
    main()
