import sys
import logging
import torch
torch.manual_seed(3)
# - setting random seed to 1 leads to zero accuracy for 1 iter (the data is linearly separable
#   from the get go!). However, after 2,3 iters accuracy increases to 0.5
#   and after 4 iters, it becomes 1. This implies that the weight vectors might be rotating?
# - Setting random seed to 9, leads to accuracy 1 even after 1 epoch. However, observe that the data is linearly separable
#   from the get go!
from shallow_collapse.model import MLPModel
from shallow_collapse.data import Gaussian1D
from shallow_collapse.utils import MetricTracker
from shallow_collapse.trainer import Trainer

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    context = {
        "N": 100,
        "BATCH_SIZE": 100,
        "NUM_EPOCHS": 1000,
        "L": 2,
        "in_features": 1,
        "hidden_features": 128,
        "out_features": 1
    }
    logging.info("context: \n{}".format(context))
    training_data = Gaussian1D(context=context)
    model = MLPModel(context=context)
    tracker = MetricTracker(context=context)
    trainer = Trainer(context=context, tracker=tracker)
    logging.info("Model: {}".format(model))
    trainer.forward_pass_at_init(model=model, training_data=training_data)
    trainer.train(model=model, training_data=training_data, probe_ntk_features=True)

if __name__ == "__main__":
    main()
