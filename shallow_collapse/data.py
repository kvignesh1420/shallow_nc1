import logging
logger = logging.getLogger(__name__)
import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class Gaussian1D():
    """
    Linearly separable 1D gaussian data
    Args:
        context: Dictionary of model training parameters
    """
    def __init__(self, context) -> None:
        self.context = context
        self.prepare_data()
        self.plot()

    def prepare_data(self):
        """
        Prepare data matrices (X, Y) for training
        """
        device = self.context["device"]
        N = self.context["N"]
        # X = torch.randn(size=(N, 1), requires_grad=False)
        X1 = torch.normal(mean=torch.tensor(-2), std=torch.tensor(0.3), size=(N//2, 1), requires_grad=False)
        X2 = torch.normal(mean=torch.tensor(2), std=torch.tensor(0.3), size=(N//2, 1), requires_grad=False)
        X = torch.concat((X1, X2))
        Y = torch.concat((torch.zeros(N//2, 1), torch.ones(N//2, 1)))
        self.perm = torch.randperm(n=N)
        self.perm_inv = torch.argsort(self.perm).to(device)
        self.X = X[self.perm].to(device)
        self.Y = Y[self.perm].to(device)
        self.labels = torch.squeeze_copy(self.Y).type(torch.int64).to(device)

    def plot(self):
        N = self.context["N"]
        points = self.X[self.perm_inv].cpu().detach().numpy()
        plt.plot(points[:N//2], np.zeros(N//2),  color="orange")
        plt.plot(points[N//2:], np.zeros(N//2), color="blue")
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.grid()
        plt.savefig("{}gaussian1d_data.jpg".format(self.context["vis_dir"]))
        plt.clf()

class Circle2D():
    """
    Linearly separable 2D circular data
    Args:
        context: Dictionary of model training parameters
    """
    def __init__(self, context) -> None:
        self.context = context
        self.prepare_data()
        self.plot()

    def prepare_data(self):
        """
        Prepare data matrices (X, Y) for training
        """
        device = self.context["device"]
        N = self.context["N"]
        # X = torch.randn(size=(N, 1), requires_grad=False)
        theta_eps = 1e-2
        theta1s = torch.linspace(-torch.pi + theta_eps, 0 - theta_eps, N//2)
        theta2s = torch.linspace(0 + theta_eps, torch.pi - theta_eps, N//2)
        thetas = torch.cat([theta1s, theta2s])
        X1 = torch.cat([torch.cos(theta1s).unsqueeze(1), torch.sin(theta1s).unsqueeze(1)], 1)
        X2 = torch.cat([torch.cos(theta2s).unsqueeze(1), torch.sin(theta2s).unsqueeze(1)], 1)
        X = torch.cat([X1, X2])
        Y = torch.cat([torch.zeros(N//2, 1), torch.ones(N//2, 1)])
        self.perm = torch.randperm(n=N)
        self.perm_inv = torch.argsort(self.perm).to(device)
        self.thetas = thetas[self.perm]
        self.X = X[self.perm].to(device)
        self.Y = Y[self.perm].to(device)
        self.labels = torch.squeeze_copy(self.Y).type(torch.int64).to(device)

    def plot(self):
        N = self.context["N"]
        points = self.X[self.perm_inv].cpu().detach().numpy()
        plt.plot(points[:N//2,0], points[:N//2,1], color="orange")
        plt.plot(points[N//2:,0], points[N//2:,1], color="blue")
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.grid()
        plt.savefig("{}circle2d_data.jpg".format(self.context["vis_dir"]))
        plt.clf()


class MNIST():
    """
    MNIST training and testing data
    """
    def __init__(self, context) -> None:
        self.context = context
        self.prepare_data()
    
    def prepare_data(self):
        """
        straightforward adaptation from https://github.com/pytorch/examples/blob/main/mnist/main.py
        """
        device = self.context["device"]
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
        flattener = torch.nn.Flatten()
        self.X = flattener(train_dataset.data).to(device)
        self.labels = train_dataset.targets.type(torch.int64).to(device)
        self.Y = self.labels.type(torch.float).unsqueeze(1).to(device)
        _, self.perm_inv = torch.sort(self.labels).to(device)
