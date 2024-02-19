import numpy as np
import torch
from typing import Dict, Any

class Erf(torch.nn.Module):
    r"""Applies the erf function element-wise:

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = Erf()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.special.erf(input)


activation_cls_map = {
    "relu": torch.nn.ReLU,
    "erf": Erf
}

class MLPModel(torch.nn.Module):
    """
    MLP model with kaiming normal initialization of weights
    Args:
        context: Dictionary of model training parameters
    """
    def __init__(self, context : Dict[str, Any]) -> None:
        super().__init__()
        self.L = context["L"]
        self.in_features = context["in_features"]
        self.hidden_features = context["hidden_features"]
        self.out_features = context["out_features"]
        self.hidden_weight_std = context["hidden_weight_std"]
        self.final_weight_std = context["final_weight_std"]
        self.bias_std = context["bias_std"]
        self.use_batch_norm = context["use_batch_norm"]
        self.activation_cls = activation_cls_map[context["activation"]]
        self._initialize_features()
        self._initialize_layers()
        self._assign_hooks()

    def _initialize_features(self):
        self.affine_features = {}
        self.activation_features = {}
        self.post_normalizations = {}

    def _initialize_layers(self):
        self.first_layer = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.hidden_features,
            bias=True
        )
        torch.nn.init.normal_(self.first_layer.weight, std=(self.hidden_weight_std/np.sqrt(self.in_features)))
        torch.nn.init.normal_(self.first_layer.bias, std=self.bias_std)
        self.hidden_layers = [self.first_layer]
        self.activation_layers = [self.activation_cls()]
        if self.use_batch_norm:
            self.normalization_layers = [torch.nn.BatchNorm1d(self.hidden_features)]

        for l in range(1, self.L-1):
            layer = torch.nn.Linear(
                in_features=self.hidden_features,
                out_features=self.hidden_features,
                bias=True
            )
            torch.nn.init.normal_(layer.weight, std=(self.hidden_weight_std/np.sqrt(self.hidden_features)))
            torch.nn.init.normal_(layer.bias, std=self.bias_std)
            self.hidden_layers.append(layer)
            self.activation_layers.append(self.activation_cls())
            if self.use_batch_norm:
                self.normalization_layers.append(torch.nn.BatchNorm1d(self.hidden_features))

        self.final_layer = torch.nn.Linear(
            in_features=self.hidden_features,
            out_features=self.out_features,
            bias=True
        )
        torch.nn.init.normal_(self.final_layer.weight, std=(self.final_weight_std/np.sqrt(self.hidden_features)))
        torch.nn.init.normal_(self.final_layer.bias, std=self.bias_std)
        self.hidden_layers.append(self.final_layer)
        self.activation_layers.append(torch.nn.Identity())

        self.hidden_layers = torch.nn.ModuleList(self.hidden_layers)
        self.activation_layers = torch.nn.ModuleList(self.activation_layers)
        if self.use_batch_norm:
            self.normalization_layers = torch.nn.ModuleList(self.normalization_layers)

    @torch.no_grad()
    def _probe_affine_features(self, idx):
        def hook(model, inp, out):
            self.affine_features[idx] = out.detach()
        return hook

    @torch.no_grad()
    def _probe_activation_features(self, idx):
        def hook(model, inp, out):
            self.activation_features[idx] = out.detach()
        return hook

    @torch.no_grad()
    def _assign_hooks(self):
        self.affine_features = {}
        self.activation_features = {}
        for layer_idx in range(len(self.hidden_layers)):
            self.hidden_layers[layer_idx].register_forward_hook(
                self._probe_affine_features(idx=layer_idx)
            )
        for layer_idx in range(len(self.activation_layers)):
            self.activation_layers[layer_idx].register_forward_hook(
                self._probe_activation_features(idx=layer_idx)
            )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        for l in range(self.L-1):
            x = self.hidden_layers[l](x)
            if self.use_batch_norm:
                x = self.normalization_layers[l](x)
            x = self.activation_layers[l](x)

        x = self.hidden_layers[self.L-1](x)
        x = self.activation_layers[self.L-1](x)
        return x
