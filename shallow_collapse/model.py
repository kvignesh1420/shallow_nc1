"""
Feed-foward model
"""

import torch
from typing import Dict

class DNNModel(torch.nn.Module):
    def __init__(self, args : Dict) -> None:
        super().__init__()
        self.L = args["L"]
        self.in_features = args["in_features"]
        self.hidden_features = args["hidden_features"]
        self.out_features = args["out_features"]
        self.bias = args["bias"]

        self.first_layer = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.hidden_features,
            bias=True
        )
        torch.nn.init.kaiming_normal_(self.first_layer.weight, nonlinearity="relu")
        torch.nn.init.normal_(self.first_layer.bias)
        self.hidden_layers = [self.first_layer]
        self.activation_layers = [torch.nn.ReLU()]
        # self.normalization_layers = [torch.nn.BatchNorm1d(self.hidden_features)]

        for l in range(1, self.L):
            layer = torch.nn.Linear(
                in_features=self.hidden_features,
                out_features=self.hidden_features,
                bias=False
            )
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            # torch.nn.init.normal_(layer.bias)
            self.hidden_layers.append(layer)
            self.activation_layers.append(torch.nn.Identity())
            # self.normalization_layers.append(torch.nn.BatchNorm1d(self.hidden_features))

        self.final_layer = torch.nn.Linear(
            in_features=self.hidden_features,
            out_features=self.out_features,
            bias=True
        )
        torch.nn.init.kaiming_normal_(self.final_layer.weight, nonlinearity="relu")
        torch.nn.init.normal_(self.final_layer.bias)
        self.hidden_layers.append(self.final_layer)
        self.activation_layers.append(torch.nn.Identity())

        self.hidden_layers = torch.nn.ModuleList(self.hidden_layers)
        self.activation_layers = torch.nn.ModuleList(self.activation_layers)
        # self.normalization_layers = torch.nn.ModuleList(self.normalization_layers)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        for l in range(self.L):
            x = self.hidden_layers[l](x)
            x = self.activation_layers[l](x)
            # x = self.normalization_layers[l](x)

        x = self.hidden_layers[self.L](x)
        x = self.activation_layers[self.L](x)
        return x
