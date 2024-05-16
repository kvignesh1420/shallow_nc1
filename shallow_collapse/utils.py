"""Utils for handling context and training"""

import sys
import os
import torch
import json
import hashlib
from shallow_collapse.data import GaussiandD
from shallow_collapse.data import Gaussian2DNL
from shallow_collapse.data import GaussiandD4NL
from shallow_collapse.data import MNIST2Class
from shallow_collapse.data import MNIST

data_cls_map = {
    "GaussiandD": GaussiandD,
    "Gaussian2DNL": Gaussian2DNL,
    "GaussiandD4NL": GaussiandD4NL,
    "MNIST2Class": MNIST2Class,
    "MNIST": MNIST
}

def prepare_data_hash(context):
    relevant_fields = ["training_data_cls", "N", "class_means", "class_stds", "class_sizes", "in_features"]
    data_context = { k: v for k, v in context.items() if k in relevant_fields  }
    _string_data_context = json.dumps(data_context, sort_keys=True).encode("utf-8")
    parsed_data_context_hash = hashlib.md5(_string_data_context).hexdigest()
    return parsed_data_context_hash

def prepare_model_hash(context):
    relevant_fields = ["L", "in_features", "hidden_features", "out_features", "use_batch_norm", "bias_std",
                   "hidden_weight_std", "final_weight_std", "activation"]
    model_context = { k: v for k, v in context.items() if k in relevant_fields  }
    _string_model_context = json.dumps(model_context, sort_keys=True).encode("utf-8")
    parsed_model_context_hash = hashlib.md5(_string_model_context).hexdigest()
    return parsed_model_context_hash


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
    # data
    data_config_uuid = prepare_data_hash(context=context)
    context["data_dir"] = "data/{}".format(data_config_uuid)
    os.makedirs(context["data_dir"], exist_ok=True)
    # model
    if context.get("name", "") != "adaptive_kernels":
        print("skipping model_dir conf for adaptive kernels")
        model_config_uuid = prepare_model_hash(context=context)
        context["model_dir"] = "models/{}".format(model_config_uuid)
        os.makedirs(context["model_dir"], exist_ok=True)
    # results
    context["out_dir"] = "out/" if context.get("name", "") != "adaptive_kernels" else "out/adaptive_kernels/"
    vis_dir = context["out_dir"] + context["config_uuid"] + "/plots/"
    results_dir = context["out_dir"] + context["config_uuid"] + "/results/"
    results_file = results_dir + "run.txt"
    if not os.path.exists(vis_dir):
        print("Vis folder does not exist. Creating {}".format(vis_dir))
        os.makedirs(vis_dir)
    else:
        print("Vis folder {} already exists!".format(vis_dir))
        # sys.exit()
    if not os.path.exists(results_dir):
        print("Resuls folder does not exist. Creating {}".format(results_dir))
        os.makedirs(results_dir)
    else:
        print("Resuls folder {} already exists!".format(results_dir))
    context["vis_dir"] = vis_dir
    context["results_file"] = results_file

    return context