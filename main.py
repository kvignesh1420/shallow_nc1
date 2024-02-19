import os
import json
import hashlib
import sys
import logging
import torch
# torch.manual_seed(3)
# - setting random seed to 1 leads to zero accuracy for 1 iter (the data is linearly separable
#   from the get go!). However, after 2,3 iters accuracy increases to 0.5
#   and after 4 iters, it becomes 1. This implies that the weight vectors might be rotating?
# - Setting random seed to 9, leads to accuracy 1 even after 1 epoch. However, observe that the data is linearly separable
#   from the get go!
from shallow_collapse.model import MLPModel
from shallow_collapse.data import GaussiandD
from shallow_collapse.data import MNIST2Class
from shallow_collapse.data import MNIST
from shallow_collapse.tracker import MetricTracker
from shallow_collapse.trainer import Trainer

data_cls_map = {
    "GaussiandD": GaussiandD,
    "MNIST2Class": MNIST2Class,
    "MNIST": MNIST
}

def prepare_config_hash(context):
    _string_context = json.dumps(context, sort_keys=True).encode("utf-8")
    parsed_context_hash = hashlib.md5(_string_context).hexdigest()
    return parsed_context_hash

def setup_runtime_context(context):
    # create a unique hash for the model
    if context["training_data_cls"] not in data_cls_map:
        sys.exit("Invalid training_data_cls. Choose from {}".format(list(data_cls_map.keys())))        
    config_uuid = prepare_config_hash(context=context)
    context["config_uuid"] = config_uuid
    context["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    context["out_dir"] = "out/"
    vis_dir = context["out_dir"] + context["config_uuid"] + "/plots/"
    results_dir = context["out_dir"] + context["config_uuid"] + "/results/"
    results_file = results_dir + "run.txt"
    if not os.path.exists(vis_dir):
        print("Vis folder does not exist. Creating {}".format(vis_dir))
        os.makedirs(vis_dir)
    else:
        print("Vis folder {} already exists!".format(vis_dir))
    if not os.path.exists(results_dir):
        print("Resuls folder does not exist. Creating {}".format(results_dir))
        os.makedirs(results_dir)
    else:
        print("Resuls folder {} already exists!".format(results_dir))
    context["vis_dir"] = vis_dir
    context["results_file"] = results_file

    return context


def main():
    exp_context = {
        "training_data_cls": "GaussiandD",
        "N": 1024,
        # note that the mean/std values will be broadcasted across `in_features`
        "class_means": [-2, 2],
        "class_stds": [0.3, 0.3],
        "class_sizes": [512, 512],
        "batch_size": 1024,
        "num_epochs": 1000,
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
        "probing_frequency": 100
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
    model = MLPModel(context=context).to(context["device"])
    tracker = MetricTracker(context=context)
    trainer = Trainer(context=context, tracker=tracker)
    logging.info("Model: {}".format(model))
    trainer.train(model=model, training_data=training_data)

if __name__ == "__main__":
    main()
