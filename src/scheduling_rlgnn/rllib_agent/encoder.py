import torch.nn as nn


class MLPEncoder(nn.Module):
    """
    Encoder class using a Multi-Layer Perceptron (MLP) architecture for the graph representation of the observations
    """
