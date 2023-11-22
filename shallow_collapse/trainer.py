import logging
logger = logging.getLogger(__name__)
from typing import Dict, Any
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

class Trainer():
    """
    Model trainer class with custom training loops
    Args:
        context: Dictionary of model training parameters
    """
    def __init__(self, context: Dict[str, Any]) -> None:
        self.context = context

    def compute_accuracy(self, pred, labels):
        pred = (pred >= 0).squeeze().type(torch.int64)
        logging.debug("pred: {}, labels: {}".format(pred, labels))
        acc = torch.mean((pred == labels).type(torch.float))
        return acc

    def plot_pred(self, pred):
        plt.plot(pred)
        plt.savefig("pred.png")
        plt.clf()

    def train(self, model, training_data):
        N = self.context["N"]
        BATCH_SIZE = self.context["BATCH_SIZE"]
        NUM_EPOCHS = self.context["NUM_EPOCHS"]
        X = training_data.X
        Y = training_data.Y
        loss_criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=0.0001,
            momentum=0,
            weight_decay=5e-4
        )
        num_batches = N//BATCH_SIZE
        logger.info("Number of batches: {}".format(num_batches))
        for iter in tqdm(range(NUM_EPOCHS)):
            for batch in range(num_batches):
                model.zero_grad()
                optimizer.zero_grad()
                x = X[ batch*(BATCH_SIZE):(batch+1)*BATCH_SIZE, : ]
                y = Y[ batch*(BATCH_SIZE):(batch+1)*BATCH_SIZE, : ]
                pred = model(x)
                loss = loss_criterion(pred, y)
                loss.backward()
                optimizer.step()
            if iter%10 == 0:
                print("iter: {} loss: {}".format(iter, loss.cpu().detach().numpy()))

        pred=model(X)
        acc = self.compute_accuracy(pred=pred, labels=training_data.labels)
        logger.info("accuracy: {}".format(acc.cpu().detach().numpy()))
        self.plot_pred(pred[training_data.perm_inv].detach().cpu().numpy())
