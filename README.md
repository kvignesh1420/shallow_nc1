# Neural Collapse in Shallow Neural Networks

This repository provides a comprehensive analysis of neural collapse in shallow neural networks through a kernel-based approach. Our study focuses on the:

- Limiting Neural Network Gaussian Process (NNGP)
- Limiting Neural Tangent Kernel (NTK)
- Adaptive Kernels (derived from NNGP)

We utilize these kernel characterizations of a 2-layer fully connected neural network (2L-FCN) to investigate NC1, which refers to the variability collapse of hidden layer activations.

## Setup

To set up the environment, follow these steps:

```bash
$ python3.9 -m virtualenv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Experiments

The `shallow_collapse` package includes the core library code required to run the experiments.

### Training the Fully Connected Network (FCN)

- **Single run:** Train a 2L-FCN using the following command:

  ```bash
  (.venv) $ python fcn.py configs/fcn.yml
  ```

- **Bulk run (balanced):** Train multiple 2L-FCN on various dataset sizes and data dimensions under balanced conditions:

  ```bash
  (.venv) $ python fcn_bulk_balanced.py configs/fcn_bulk_balanced.yml
  ```

- **Bulk run (imbalanced):** Train multiple 2L-FCN on various dataset sizes and data dimensions under imbalanced conditions:

  ```bash
  (.venv) $ python fcn_bulk_imbalanced.py configs/fcn_bulk_imbalanced.yml
  ```

### Limiting Kernels

- **Bulk run (balanced):** Conduct multiple experiments using NNGP/NTK with balanced datasets:

  ```bash
  (.venv) $ python limiting_kernels_bulk_balanced.py configs/limiting_kernels_bulk_balanced.yml
  ```

- **Bulk run (imbalanced):** Execute multiple experiments using NNGP/NTK with imbalanced datasets:

  ```bash
  (.venv) $ python limiting_kernels_bulk_imbalanced.py configs/limiting_kernels_bulk_imbalanced.yml
  ```

### Adaptive Kernels

- **Single run:** Solve the "Equations of State" (EoS) for the adaptive kernels using the following command:

  ```bash
  (.venv) $ python adaptive_kernels.py configs/adaptive_kernels.yml
  ```

- **Bulk run (balanced):** Perform multiple experiments using EoS-based adaptive kernels with balanced datasets:

  ```bash
  (.venv) $ python adaptive_kernels_bulk_balanced.py configs/adaptive_kernels_bulk_balanced.yml
  ```

- **Bulk run (imbalanced):** Run multiple experiments using EoS-based adaptive kernels with imbalanced datasets:

  ```bash
  (.venv) $ python adaptive_kernels_bulk_imbalanced.py configs/adaptive_kernels_bulk_imbalanced.yml
  ```

_All output files are stored in the `out/` directory. Each output is associated with a unique hash value corresponding to the context of the experiment._

## Citation

```bibtex
@article{
  kothapalli2025can,
  title={Can Kernel Methods Explain How the Data Affects Neural Collapse?},
  author={Vignesh Kothapalli and Tom Tirer},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=MbF1gYfIlY},
  note={}
}
```