import logging

from shallow_collapse.bulk_runner import BulkImbalancedRunnerEoS
from shallow_collapse.utils import setup_runtime_context, parse_config

N = 2048
CLASS_SIZES_LIST = [
    (512 * 2, 512 * 2),
    (384 * 2, 640 * 2),
    (256 * 2, 768 * 2),
    (128 * 2, 896 * 2),
]
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

    bulk_runner = BulkImbalancedRunnerEoS(context=context)
    bulk_runner.run(
        N=N,
        CLASS_SIZES_LIST=CLASS_SIZES_LIST,
        IN_FEATURES_LIST=IN_FEATURES_LIST,
        REPEAT=REPEAT,
        TAU=TAU,
    )


if __name__ == "__main__":
    main()
