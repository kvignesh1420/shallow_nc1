import logging
logger = logging.getLogger(__name__)
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class _SyntheticDataset(Dataset):
    def __init__(self, X, labels) -> None:
        super().__init__()
        self.X = X
        self.labels = labels
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.labels[index]


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
        labels = torch.cat([torch.zeros(N//2), torch.ones(N//2)])
        self.perm = torch.randperm(n=N)
        self.perm_inv = torch.argsort(self.perm).to(device)
        self.X = X[self.perm].to(device)
        self.labels = labels[self.perm].to(device)
        self.class_sizes = torch.Tensor([N//2, N//2]).type(torch.int)
        self.prepare_data_loader()

    def prepare_data_loader(self):
        train_dataset = _SyntheticDataset(X=self.X, labels=self.labels)
        train_kwargs = {"batch_size": self.context["batch_size"], "shuffle": False}
        self.train_loader = DataLoader(train_dataset, **train_kwargs)

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
        labels = torch.cat([torch.zeros(N//2), torch.ones(N//2)])
        self.perm = torch.randperm(n=N)
        self.perm_inv = torch.argsort(self.perm).to(device)
        self.X = X[self.perm].to(device)
        self.labels = labels[self.perm].to(device)
        self.thetas = thetas[self.perm]
        self.class_sizes = torch.Tensor([N//2, N//2]).type(torch.int)
        self.prepare_data_loader()

    def prepare_data_loader(self):
        train_dataset = _SyntheticDataset(X=self.X, labels=self.labels)
        train_kwargs = {"batch_size": self.context["batch_size"], "shuffle": False}
        self.train_loader = DataLoader(train_dataset, **train_kwargs)

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


class MNIST2Class():
    """
    MNIST training and testing data
    """
    def __init__(self, context) -> None:
        self.context = context
        self.prepare_data()
    
    def prepare_data(self):
        """
        Create a 2-class balanced dataset from MNIST training data.
        For simplicity, we choose 0 and 1.
        straightforward adaptation from https://github.com/pytorch/examples/blob/main/mnist/main.py
        """
        device = self.context["device"]
        N = self.context["N"]
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
        flattener = torch.nn.Flatten()
        X = flattener(train_dataset.data).type(torch.float).to(device)
        labels = train_dataset.targets.type(torch.int64).to(device)
        zero_labels = (labels == 0).nonzero().flatten()[:N//2]
        one_labels = (labels == 1).nonzero().flatten()[:N//2]
        X1 = X[zero_labels]
        X2 = X[one_labels]
        X = torch.cat([X1, X2])
        labels = torch.cat([torch.zeros(N//2), torch.ones(N//2)])
        self.perm = torch.randperm(n=N)
        self.perm_inv = torch.argsort(self.perm).to(device)
        self.X = X[self.perm].to(device)
        self.labels = labels[self.perm].to(device)
        self.prepare_data_loader()

    def prepare_data_loader(self):
        train_dataset = _SyntheticDataset(X=self.X, labels=self.labels)
        train_kwargs = {"batch_size": self.context["batch_size"], "shuffle": False}
        self.train_loader = DataLoader(train_dataset, **train_kwargs)
        self.num_classes = self.context["num_classes"]
        self.class_sizes = torch.zeros((self.num_classes)).to(self.context["device"])
        for _, labels in self.train_loader:
            labels = labels.to(self.context["device"])
            class_count = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).sum(dim = 0)
            self.class_sizes += class_count
        self.class_sizes = self.class_sizes.type(torch.int)
    

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
        N = self.context["N"]
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])
        train_kwargs = {"batch_size": self.context["batch_size"]}
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
        self.train_loader = DataLoader(train_dataset, **train_kwargs)
        self.num_classes = self.context["num_classes"]
        self.class_sizes = torch.zeros((self.num_classes)).to(self.context["device"])
        for _, labels in self.train_loader:
            labels = labels.to(self.context["device"])
            class_count = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).sum(dim = 0)
            self.class_sizes += class_count
        self.class_sizes = self.class_sizes.type(torch.int)
