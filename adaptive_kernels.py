"""
Torch implementation of the Adaptive Kernels

Reference: https://www.nature.com/articles/s41467-023-36361-y
"""

import logging

from shallow_collapse.eos import EoSSolver
from shallow_collapse.utils import setup_runtime_context
from shallow_collapse.utils import data_cls_map
from shallow_collapse.utils import get_exp_context


if __name__ == "__main__":
    exp_context = get_exp_context()
    context = setup_runtime_context(context=exp_context)
    logging.basicConfig(
        filename=context["results_file"],
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    logging.info("context: \n{}".format(context))

    training_data = data_cls_map[context["training_data_cls"]](context=context)
    solver = EoSSolver(context=context)
    solver.solve(training_data=training_data)
