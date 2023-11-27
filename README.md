# Neural Collapse in shallow neural networks

This effort aims to analyze the extent of neural collapse in shallow neural networks via kernel based analysis. Especially, we leverage the NNGP and NTK characterizations of a 2-layer fully connected neural network and analyze NC1 based on the kernel matrices.

## Setup

```bash
$ python3.9 -m virtualenv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Experiments

- As of now, modify the `exp_context` dictionary in `main.py` to configure the experiment. A sample value is shown below:
```py
exp_context = {
    "training_data_cls": "Circle2D",
    "N": 200,
    "BATCH_SIZE": 200,
    "NUM_EPOCHS": 1,
    "L": 2,
    "in_features": 2,
    "hidden_features": 1024,
    "out_features": 1,
    "lr": 1e-4,
    "weight_decay": 5e-4,
    "bias_std": 1
}
```

- To run the experiment, use:
```bash
(.venv) $ python main.py
```

- The outputs are generated in the `out/` folder based on a hash value corresponding to the context.