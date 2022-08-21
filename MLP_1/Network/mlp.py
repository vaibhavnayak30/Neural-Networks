# import necessary libraries
import torch.nn as nn
from collections import OrderedDict


def get_training_model(inFeatures=4, hiddenDim=8, nbClasses=3):
    mlpModel = nn.Sequential(
        OrderedDict(
            [
                ("hidden_layer_1", nn.Linear(inFeatures, hiddenDim)),
                ("activation_1",   nn.ReLU()),
                ("output_layer",   nn.Linear(hiddenDim, nbClasses))
            ]))

    # return the sequential model
    return mlpModel
