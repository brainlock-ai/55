import torch
from torch.nn import Module

class SubModel(Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def split_cnv_model(model):
    """
    Split the CNV model into submodels based on its structure.
    """
    # Split convolutional features into smaller groups
    conv_splits = [
        SubModel(model.conv_features[:4]),  # First group of convolution layers
        SubModel(model.conv_features[4:8]),  # Second group (e.g., with pooling)
        SubModel(model.conv_features[8:])   # Remaining convolution layers
    ]

    # Split linear features into smaller groups
    linear_splits = [
        SubModel(model.linear_features[:3]),  # First group of linear layers
        SubModel(model.linear_features[3:])  # Final classification layers
    ]

    return conv_splits, linear_splits
