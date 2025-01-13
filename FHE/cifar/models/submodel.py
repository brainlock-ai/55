import torch
from torch.nn import Module
from .model import CNV_OUT_CH_POOL, INTERMEDIATE_FC_FEATURES

class SubModel(Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)  # Use ModuleList to store layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  # Pass input through each layer sequentially
        return x

def split_cnv_model(model):
    """
    Split the CNV model into submodels based on its structure.
    """
    # Split convolutional features into smaller groups
    conv_group_delimiters = [0]
    initial_identity_layer = True  #QuantIdentity
    
    for _, is_pool_enabled in CNV_OUT_CH_POOL[:-1]:
        delimiter = 3  # QuantConv2d, BatchNorm2d, QuantIdentity
        if initial_identity_layer:
            delimiter += 1
            initial_identity_layer = False
        if is_pool_enabled:
            delimiter += 2  # AvgPool2d, QuantIdentity
        conv_group_delimiters.append(conv_group_delimiters[-1] + delimiter)
    
    conv_group_delimiters.append(None)

    print(conv_group_delimiters)

    conv_splits = [
        SubModel(model.conv_features[
            conv_group_delimiters[i]:conv_group_delimiters[i+1]
        ]) for i in range(len(CNV_OUT_CH_POOL))
    ]

    #conv_splits = [
    #    SubModel(model.conv_features[:4]),  # First group of convolution layers
    #    SubModel(model.conv_features[4:8]),  # Second group (e.g., with pooling)
    #    SubModel(model.conv_features[8:])   # Remaining convolution layers
    #]

    # Split linear features into smaller groups
    linear_group_delimiters = [0]
    for _ in INTERMEDIATE_FC_FEATURES[:-1]:
        delimiter = 3  # QuantLinear, BatchNorm1d, QuantIdentity
        linear_group_delimiters.append(linear_group_delimiters[-1] + 3)

    linear_group_delimiters.append(None)

    print(linear_group_delimiters)

    linear_splits = [
        SubModel(model.linear_features[
            linear_group_delimiters[i]:linear_group_delimiters[i+1]
        ]) for i in range(len(INTERMEDIATE_FC_FEATURES))
    ]
    #linear_splits = [
    #    SubModel(model.linear_features[:3]),  # QuantLinear, BatchNorm1d, QuantIdentity
    #    SubModel(model.linear_features[3:])  # QuantLinear, BatchNorm1d, QuantIdentity,
    #]
    
    assert sum(len(conv_submodel.layers) for conv_submodel in conv_splits) == len(model.conv_features)
    
    assert sum(len(linear_submodel.layers) for linear_submodel in linear_splits) == len(model.linear_features)

    return conv_splits, linear_splits
