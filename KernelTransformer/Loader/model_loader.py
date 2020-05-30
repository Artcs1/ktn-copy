#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from cfg import LAYERS
from cfg import MODEL_DIR
from KTNLayer import KTNConv


INPUT_WIDTH = 640
FOV = 65.5

imgW = {
    "1a": INPUT_WIDTH,
    "1b": INPUT_WIDTH,
    "2a": INPUT_WIDTH / 2,
    "2b": INPUT_WIDTH / 2,
    "3a": INPUT_WIDTH / 2 ** 2,
    "3b": INPUT_WIDTH / 2 ** 2,
    "4a": INPUT_WIDTH / 2 ** 3,
    "4b": INPUT_WIDTH / 2 ** 3,
    "Pa": INPUT_WIDTH / 2 ** 3,
    "Pb": INPUT_WIDTH / 2 ** 3,
    "Da": INPUT_WIDTH / 2 ** 3,
    "Db": INPUT_WIDTH / 2 ** 3,
}

DILATIONS = {
    
    "1a": 1,
    "1b": 1,
    "2a": 1,
    "2b": 1,
    "3a": 1,
    "3b": 1,
    "4a": 1,
    "4b": 1,
    "Pa": 2,
    "Pb": 2,
    "Da": 2,
    "Db": 2,
}

TIED_WEIGHT = 5

TIED_WEIGHTS = {
    "1a": 1,
    "1b": TIED_WEIGHT,
    "2a": TIED_WEIGHT,
    "2b": TIED_WEIGHT,
    "3a": TIED_WEIGHT,
    "3b": TIED_WEIGHT,
    "4a": TIED_WEIGHT,
    "4b": TIED_WEIGHT,
    "Pa": TIED_WEIGHT,
    "Pb": TIED_WEIGHT,
    "Da": TIED_WEIGHT,
    "Db": TIED_WEIGHT,
}

ARCHS = {
    "1a": "bilinear",
    "1b": "residual",
    "2a": "residual",
    "2b": "residual",
    "3a": "residual",
    "3b": "residual",
    "4a": "residual",
    "4b": "residual",
    "Da": "residual",
    "Db": "residual",
    "Pa": "residual",
    "Pb": "residual",
}


class KTNNet(nn.Module):

    def __init__(self, dst, **kwargs):
        super(KTNNet, self).__init__()

        dst_i = LAYERS.index(dst) + 1
        src = kwargs.get("src", "pixel")
        if src == "pixel":
            src_i = 0
        else:
            src_i = LAYERS.index(src) + 1

        layers = []
        for layer in LAYERS[src_i:dst_i]:
            ktnconv = build_ktnconv(layer, **kwargs)
            layers.append(ktnconv)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        print("dudosaa in resuelta")
        print(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                print("quien pasa por aqui")
                x = F.relu(x)
        return x

    def update_group(self, group):
        for layer in self.layers:
            layer.update_group(group)


def load_src(target, network="pascal"):
    path = os.path.join(MODEL_DIR, "{}.pt".format(network))
    sys.stderr.write("Load source kernels for {0} from {1}\n".format(target, path))
    weights = torch.load(path)
    

    key = "conv{}.weight".format(target)
    kernel = weights[key]
    key = "conv{}.bias".format(target)
    bias = weights[key]
    return kernel, bias

def build_ktnconv(target, **kwargs):
    network = kwargs.get("network", "pascal")
    kernel, bias = load_src(target, network=network)

    sphereH = kwargs.get("sphereH", INPUT_WIDTH / 2)
    fov = kwargs.get("fov", FOV)
    iw = kwargs.get("imgW", imgW[target])
    dilation = kwargs.get("dilation", DILATIONS[target])
    tied_weights = kwargs.get("tied_weights", TIED_WEIGHTS[target])
    arch = kwargs.get("arch", ARCHS[target])
    if target == "1a":
        kernel_shape_type = "full"
    else:
        kernel_shape_type = "dilated"
    print(kernel_shape_type)
    sys.stderr.write("Build layer {0} with arch: {1}, tied_weights: {2}\n".format(target, arch, tied_weights))
    ktnconv = KTNConv(kernel,
                      bias,
                      sphereH=sphereH,
                      imgW=iw,
                      fov=fov,
                      dilation=dilation,
                      tied_weights=tied_weights,
                      arch=arch,
                      kernel_shape_type=kernel_shape_type)
    return ktnconv

def load_ktnnet(network, dst, **kwargs):
    ktnnet = KTNNet(dst, network=network, **kwargs)
        
    dst_i = LAYERS.index(dst)
    src = kwargs.get("src", "pixel")
    if src == "pixel":
        src_i = 0
    else:
        src_i = LAYERS.index(src) + 1
    layers = LAYERS[src_i:dst_i+1]

    ktn_state_dict = OrderedDict()
    transform = kwargs.get("transform", "pascal")
    for i, layer in enumerate(layers):
        model_name = "{0}{1}.transform.pt".format(transform, layer)
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.isfile(model_path):
            sys.stderr.write("Skip {}\n".format(model_path))
            continue
        sys.stderr.write("Load transformation from {}\n".format(model_path))
        ktn_state = torch.load(model_path)
        for name, params in ktn_state.iteritems():
            if "src_kernel" in name or "src_bias" in name:
                continue
            name = "layers.{0}.{1}".format(i, name)
            ktn_state_dict[name] = params

    # Use default parameters 
    for name, params in ktnnet.state_dict().iteritems():
        if name not in ktn_state_dict:
            ktn_state_dict[name] = params
    ktnnet.load_state_dict(ktn_state_dict)
    return ktnnet

