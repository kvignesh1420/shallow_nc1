# Neural Collapse in Shallow Neural Networks

This effort aims to analyze the extent of neural collapse in shallow neural networks via kernel based analysis. Especially, we leverage the NNGP and NTK characterizations of a 2-layer fully connected neural network and analyze NC1 based on the kernel matrices.

## Setup

```bash
$ python3.9 -m virtualenv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Experiments

The `shallow_collapse` package contains the library code for running the experiments.

1. The `main.py` script can be used to run single 2L-FCN experiments.

2. The `main_nc1_balanced.py` script can be used to run multiple 2L-FCN experiments with varying N and data dimension.

3. The `main_nc1_imbalanced.py` script can be used to run multiple 2L-FCN experiments with imbalanced N and data dimension.

4. Similar description applies to `limiting_kernels_nc1_balanced.py` and `limiting_kernels_nc1_imbalanced.py` for NNGP/NTK.

5. Similar description applies to `adaptive_kernels_nc1_balanced.py` and `adaptive_kernels_nc1_imbalanced.py` for EoS based adaptive kernels.

One can modify the `exp_context` dictionary in the scripts to configure the experiment. A sample value is shown below:

```py
exp_context = {
        "training_data_cls": "Gaussian2DNL",
        "N": 1024,
        # note that the mean/std values will be broadcasted across `in_features`
        # i.e, each entry in the list corresponds to each class.
        "class_means": [-2, 2],
        "class_stds": [0.5, 0.5],
        "class_sizes": [512, 512],
        "batch_size": 1024,
        "num_epochs": 1000,
        "L": 2,
        "in_features": 1,
        "hidden_features": 2000,
        "out_features": 1,
        "num_classes" : 2,
        "use_batch_norm": False,
        "lr": 1e-4,
        "momentum": 0.0,
        "weight_decay": 1e-6,
        "bias_std": 0,
        "hidden_weight_std": 1,
        "final_weight_std": 1.97,
        "activation": "erf",
        "probe_features": True,
        "probe_kernels": False,
        "probe_weights": True,
        "probing_frequency": 1000,
        "use_cache": True # set it to False for discarding cache of data and models
    }
```

- To run the experiment, use:
```bash
(.venv) $ python main.py
```

- The outputs are generated in the `out/` folder based on a hash value corresponding to the context.