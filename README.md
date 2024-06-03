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

1. The `fcn.py` script can be used to run single 2L-FCN experiments.

2. The `fcn_bulk_balanced.py` script can be used to run multiple 2L-FCN experiments with varying dataset sizes and data dimension.

3. The `fcn_bulk_imbalanced.py` script can be used to run multiple 2L-FCN experiments with imbalanced class sizes and data dimension.

4. Similar description applies to `limiting_kernels_bulk_balanced.py` and `limiting_kernels_bulk_imbalanced.py` for NNGP/NTK.

5. Similar description applies to `adaptive_kernels_bulk_balanced.py` and `adaptive_kernels_bulk_imbalanced.py` for EoS based adaptive kernels.

- To run the experiment, use:
```bash
(.venv) $ python fcn.py configs/fcn.yml
```

- The outputs are generated in the `out/` folder based on a hash value corresponding to the context.