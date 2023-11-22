import logging
logger = logging.getLogger(__name__)
import torch
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
        N = self.context["N"]
        # X = torch.randn(size=(N, 1), requires_grad=False)
        X1 = torch.normal(mean=torch.tensor(-2), std=torch.tensor(0.3), size=(N//2, 1), requires_grad=False)
        X2 = torch.normal(mean=torch.tensor(2), std=torch.tensor(0.3), size=(N//2, 1), requires_grad=False)
        X = torch.concat((X1, X2))
        Y = torch.concat((torch.zeros(N//2, 1), torch.ones(N//2, 1)))
        self.perm = torch.randperm(n=N)
        self.perm_inv = torch.argsort(self.perm)
        self.X = X[self.perm]
        self.Y = Y[self.perm]
        self.labels = torch.squeeze_copy(self.Y).type(torch.int64)

    def plot(self):
        plt.plot(self.X[self.perm_inv].cpu().detach().numpy())
        plt.savefig("data.png")
        plt.clf()