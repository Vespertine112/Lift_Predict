from collections import OrderedDict
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def getLiftRegressionModel(inFeatures=4, outFeatures=1):
    mlp_model = nn.Sequential(
        OrderedDict(
            [
                ("hidden_layer_1", nn.Linear(inFeatures, 36)),
                ("activation_1", nn.ReLU()),
                ("hidden_layer_2", nn.Linear(36, 18)),
                ("activation_2", nn.ReLU()),
                ("hidden_layer_3", nn.Linear(18, 9)),
                ("activation_3", nn.ReLU()),
                ("hidden_layer_4", nn.Linear(9, 6)),
                ("activation_4", nn.ReLU()),
                ("hidden_layer_5", nn.Linear(6, 6)),
                ("activation_5", nn.ReLU()),
                ("hidden_layer_6", nn.Linear(6, 3)),
                ("activation_6", nn.ReLU()),
                ("output_layer", nn.Linear(3, outFeatures)),
            ]
        )
    )
    return mlp_model
