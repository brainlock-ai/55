# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Original file can be found at https://github.com/Xilinx/brevitas/blob/8c3d9de0113528cf6693c6474a13d802a66682c6/src/brevitas_examples/bnn_pynq/models/CNV.py

import torch
from brevitas.core.restrict_val import RestrictValueType
from brevitas.nn import QuantConv2d, QuantIdentity
from torch.nn import AvgPool2d, BatchNorm2d, Module, ModuleList

from .common import CommonActQuant, CommonWeightQuant


CNV_OUT_CH_POOL = [(64, False), (64, True), (128, False), (128, True), (256, False), (256, False)]
INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]
LAST_FC_IN_FEATURES = 512
LAST_FC_PER_OUT_CH_SCALING = False
POOL_SIZE = 2
KERNEL_SIZE = 3

class SyntheticCNV(Module):
    def __init__(self, weight_bit_width, act_bit_width, in_bit_width, in_ch=256):
        super(SyntheticCNV, self).__init__()

        self.features = ModuleList()

        # Quantized Activation
        self.features.append(
            QuantIdentity(  # for Q1.7 input format
                act_quant=CommonActQuant,
                return_quant_tensor=True,
                bit_width=in_bit_width,
                min_val=-1.0,
                max_val=1.0 - 2.0 ** (-7),
                narrow_range=False,
                restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            )
        )

        # Quantized Convolutional Layer
        self.features.append(
            QuantConv2d(
                kernel_size=KERNEL_SIZE,
                stride=1,
                padding=1,
                in_channels=in_ch,
                out_channels=in_ch,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width,
            )
        )

        # Batch Normalization
        self.features.append(BatchNorm2d(in_ch, eps=1e-4))

        # Quantized Activation
        self.features.append(
            QuantIdentity(
                act_quant=CommonActQuant,
                bit_width=weight_bit_width,
                min_val=-1.0,
                max_val=1.0 - 2.0 ** (-7),
                restrict_scaling_type=RestrictValueType.POWER_OF_TWO
            )
        )

    def clip_weights(self, min_val, max_val):
        for mod in self.features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        for mod in self.features:
            x = mod(x)
        return x


def synthetic_cnv(cfg):
    weight_bit_width = cfg.getint("QUANT", "WEIGHT_BIT_WIDTH")
    act_bit_width = cfg.getint("QUANT", "ACT_BIT_WIDTH")
    in_bit_width = cfg.getint("QUANT", "IN_BIT_WIDTH")
    in_channels = cfg.getint("MODEL", "IN_CHANNELS")
    net = SyntheticCNV(
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        in_bit_width=in_bit_width,
        in_ch=in_channels,
    )
    return net
