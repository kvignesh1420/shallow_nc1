import os
import logging
import torch
from shallow_collapse.model import MLPModel
from shallow_collapse.tracker import MetricTracker
from shallow_collapse.trainer import Trainer

from shallow_collapse.utils import setup_runtime_context, data_cls_map, parse_config


def main():
    exp_context = parse_config()
    context = setup_runtime_context(context=exp_context)
    logging.basicConfig(
        filename=context["results_file"],
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    logging.info("context: \n{}".format(context))
    training_data = data_cls_map[context["training_data_cls"]](context=context)

    model = MLPModel(context=context)
    model_path = os.path.join(context["model_dir"], "model.pth")
    if os.path.exists(model_path) and context.get("use_cache", True):
        print("Loading the init state of model from {}".format(model_path))
        model.load_state_dict(torch.load(model_path))
    else:
        print("Saving the init state of model to {}".format(model_path))
        torch.save(model.state_dict(), model_path)

    model = model.to(context["device"])
    tracker = MetricTracker(context=context)
    trainer = Trainer(context=context, tracker=tracker)
    logging.info("Model: {}".format(model))
    trainer.train(model=model, training_data=training_data)


if __name__ == "__main__":
    main()
