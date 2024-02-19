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

class GaussiandD():
    """
    Linearly separable d-Dimensional gaussian data
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
        self.d = self.context["in_features"]
        self.class_sizes = torch.Tensor(self.context["class_sizes"]).type(torch.int)
        assert N == self.class_sizes.sum()

        num_classes = self.class_sizes.shape[0]
        X = []
        labels = []
        for c in range(num_classes):
            class_mean = self.context["class_means"][c]
            class_std = self.context["class_stds"][c]
            class_size = self.context["class_sizes"][c]
            logger.info("creating data for class: {} with mean: {} std: {} and size: {}".format(
                c, class_mean, class_std, class_size))
            X_c = torch.normal(
                mean=torch.tensor(class_mean),
                std=torch.tensor(class_std),
                size=(class_size, self.d),
                requires_grad=False
            )
            X.append(X_c)
            labels_c = torch.ones(class_size)*c
            labels.append(labels_c)

        X = torch.concat(X)
        labels = torch.cat(labels)
        self.perm = torch.randperm(n=N)
        self.perm_inv = torch.argsort(self.perm).to(device)
        self.X = X[self.perm].to(device)
        self.labels = labels[self.perm].to(device)
        self.prepare_data_loader()

    def prepare_data_loader(self):
        train_dataset = _SyntheticDataset(X=self.X, labels=self.labels)
        train_kwargs = {"batch_size": self.context["batch_size"], "shuffle": False}
        self.train_loader = DataLoader(train_dataset, **train_kwargs)

    def plot(self):
        """
        3d projection of data. A simple plot of the first 3 dimensions of data.
        Pad with zeros, if data is 1D/2D.
        """
        points = self.X[self.perm_inv].cpu().detach().numpy()
        offset = 0
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for idx, class_size in enumerate(self.class_sizes.cpu().detach().numpy()):
            class_points = points[offset:offset+class_size]
            x = class_points[:,0]
            y = class_points[:, 1] if self.d > 1 else np.zeros_like(x)
            z = class_points[:, 2] if self.d > 2 else np.zeros_like(x)
            ax.scatter(x, y, z, marker='o', label="class={},size={}".format(idx, class_size))
            offset += class_size

        # Set labels
        ax.set_xlabel('dim0')
        ax.set_ylabel('dim1')
        ax.set_zlabel('dim2')
        fig.tight_layout()
        plt.legend()
        plt.savefig("{}gaussiandD_data.jpg".format(self.context["vis_dir"]))
        plt.clf()

class Gaussian2D(GaussiandD):
    def __init__(self, context) -> None:
        super().__init__(context=context)
        assert context["num_classes"] == 2

    def prepare_data(self):
        """
        Prepare data matrices (X, Y) for training
        """
        device = self.context["device"]
        N = self.context["N"]
        # X = torch.randn(size=(N, 1), requires_grad=False)
        self.d = self.context["in_features"]
        self.class_sizes = torch.Tensor(self.context["class_sizes"]).type(torch.int)
        assert N == self.class_sizes.sum()

        class_means = self.context["class_means"]
        class_stds = self.context["class_stds"]
        class_sizes = self.context["class_sizes"]
        logger.info("creating data for classes -1, 1 with mean: {} std: {} and size: {}".format(
            class_means, class_stds, class_sizes))
    
        X1 = torch.normal(
            mean=torch.tensor(class_means[0]),
            std=torch.tensor(class_stds[0]),
            size=(class_sizes[0], self.d),
            requires_grad=False
        )
        labels_1 = -torch.ones(class_sizes[0])

        X2 = torch.normal(
            mean=torch.tensor(class_means[1]),
            std=torch.tensor(class_stds[1]),
            size=(class_sizes[1], self.d),
            requires_grad=False
        )
        labels_2 = torch.ones(class_sizes[1])

        X = torch.concat([X1, X2])
        labels = torch.cat([labels_1, labels_2])
        self.perm = torch.randperm(n=N)
        self.perm_inv = torch.argsort(self.perm).to(device)
        self.X = X[self.perm].to(device)
        self.labels = labels[self.perm].to(device)
        self.prepare_data_loader()


# class Circle2D():
#     """
#     Linearly separable 2D circular data
#     Args:
#         context: Dictionary of model training parameters
#     """
#     def __init__(self, context) -> None:
#         self.context = context
#         self.prepare_data()
#         self.plot()

#     def prepare_data(self):
#         """
#         Prepare data matrices (X, Y) for training
#         """
#         device = self.context["device"]
#         N = self.context["N"]
#         # X = torch.randn(size=(N, 1), requires_grad=False)
#         theta_eps = 1e-2
#         theta1s = torch.linspace(-torch.pi + theta_eps, 0 - theta_eps, N//2)
#         theta2s = torch.linspace(0 + theta_eps, torch.pi - theta_eps, N//2)
#         thetas = torch.cat([theta1s, theta2s])
#         X1 = torch.cat([torch.cos(theta1s).unsqueeze(1), torch.sin(theta1s).unsqueeze(1)], 1)
#         X2 = torch.cat([torch.cos(theta2s).unsqueeze(1), torch.sin(theta2s).unsqueeze(1)], 1)
#         X = torch.cat([X1, X2])
#         labels = torch.cat([torch.zeros(N//2), torch.ones(N//2)])
#         self.perm = torch.randperm(n=N)
#         self.perm_inv = torch.argsort(self.perm).to(device)
#         self.X = X[self.perm].to(device)
#         self.labels = labels[self.perm].to(device)
#         self.thetas = thetas[self.perm]
#         self.class_sizes = torch.Tensor([N//2, N//2]).type(torch.int)
#         self.prepare_data_loader()

#     def prepare_data_loader(self):
#         train_dataset = _SyntheticDataset(X=self.X, labels=self.labels)
#         train_kwargs = {"batch_size": self.context["batch_size"], "shuffle": False}
#         self.train_loader = DataLoader(train_dataset, **train_kwargs)

#     def plot(self):
#         N = self.context["N"]
#         points = self.X[self.perm_inv].cpu().detach().numpy()
#         plt.plot(points[:N//2,0], points[:N//2,1], color="orange")
#         plt.plot(points[N//2:,0], points[N//2:,1], color="blue")
#         plt.xlabel("x-axis")
#         plt.ylabel("y-axis")
#         plt.grid()
#         plt.savefig("{}circle2d_data.jpg".format(self.context["vis_dir"]))
#         plt.clf()


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
