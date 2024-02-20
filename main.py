import os
import logging
import torch
from shallow_collapse.model import MLPModel
from shallow_collapse.tracker import MetricTracker
from shallow_collapse.trainer import Trainer

from shallow_collapse.utils import setup_runtime_context
from shallow_collapse.utils import data_cls_map

def main():
    exp_context = {
        "training_data_cls": "Gaussian2DNL",
        "N": 1024,
        # note that the mean/std values will be broadcasted across `in_features`
        "class_means": [-2, 2],
        "class_stds": [0.3, 0.3],
        "class_sizes": [512, 512],
        "batch_size": 1024,
        "num_epochs": 24,
        "L": 2,
        "in_features": 1,
        "hidden_features": 1024,
        "out_features": 1,
        "num_classes" : 2,
        "use_batch_norm": False,
        "lr": 1e-3,
        "momentum": 0.0,
        "weight_decay": 5e-4,
        "bias_std": 0,
        "hidden_weight_std": 1,
        "final_weight_std": 1,
        "activation": "erf",
        "probe_features": True,
        "probe_kernels": True,
        "probing_frequency": 2
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

    model = MLPModel(context=context)
    model_path = os.path.join(context["model_dir"], "model.pth")
    if os.path.exists(model_path):
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
