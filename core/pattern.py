from random import random

import numpy as np
import torch

from models.blocks import LinearBlock, ConvBlock, BasicBlock, Bottleneck


class BaseHook:
    """
    Base method for adding hooks of a network.
    """

    def __init__(self, model):
        """
        Initialization method
        :param model: The model to be added
        """
        self.model = model
        self.handles = []  # handles for registered hooks
        self.features = {}  # recorder for computed attributes
        self.counter = {}

    def set_up(self):
        """
        Remove all previous hooks and register hooks for each of t
        :return:
        """
        self.remove()
        [self.add_block_hook(block_name, block) for block_name, block in self.model.valid_blocks()]

    def add_block_hook(self, block_name, block):
        pass

    def remove(self):
        [handle.remove() for handle in self.handles]
        self.features = {}
        return

    def hook(self, block_name, module_name):
        def fn(layer, input_var, output_var):
            pass

        return fn


class BaseActHook(BaseHook):
    def add_block_hook(self, block_name, block):
        self.features[block_name] = {}
        if isinstance(block, (LinearBlock, ConvBlock)):
            self.features[block_name]['act'] = None
            self.handles += [block.Act.register_forward_hook(self.hook(block_name, 'act'))]
            self.counter[block_name] = 0
        if isinstance(block, (BasicBlock, Bottleneck)):
            self.features[block_name]['act'] = None
            self.handles += [block.act.register_forward_hook(self.hook(block_name, 'act'))]
            self.counter[block_name] = 0


class BaseWeightHook(BaseHook):
    def add_block_hook(self, block_name, block):
        self.features[block_name] = {}
        if isinstance(block, (LinearBlock, ConvBlock)):
            self.features[block_name]['lt'] = None
            self.handles += [block.LT.register_full_backward_hook(self.hook(block_name, 'lt'))]
            self.counter[block_name] = 0
        if isinstance(block, (BasicBlock, Bottleneck)):
            self.features[block_name]['conv2'] = None
            self.handles += [block.conv2.register_full_backward_hook(self.hook(block_name, 'conv2'))]
            self.counter[block_name] = 0


class GradientHook(BaseWeightHook):
    def hook(self, block_name, module_name):
        def fn(layer, input_var, output_var):
            self.counter[block_name] += 1
            if self.counter[block_name] % 10 == 0:
                return
            if self.features[block_name][module_name] is None:
                self.features[block_name][module_name] = output_var[0].detach().cpu().numpy()
            else:
                self.features[block_name][module_name] += output_var[0].detach().cpu().numpy()

        return fn

    def retrieve(self):
        return self.features


class PreActHook(BaseActHook):
    def hook(self, block_name, module_name):
        def fn(layer, input_var, output_var):
            self.counter[block_name] += 1
            if self.counter[block_name] % 10 == 0:
                return
            if self.features[block_name][module_name] is None:
                self.features[block_name][module_name] = input_var[0].detach().cpu().numpy()
            else:
                self.features[block_name][module_name] += input_var[0].detach().cpu().numpy()

        return fn

    def retrieve(self):
        global_pre_act = []
        for block_name, block in self.features.items():
            for layer_name, layer in block.items():
                global_pre_act.append(layer)
        return global_pre_act


class EntropyHook(BaseHook):
    """
    Entropy hook is a forward hood that computes the neuron entropy of the network.
    """

    def __init__(self, model, Gamma, ratio=1):
        """
        Initialization method.
        :param model: Pytorch model, which should be a sequential blocks
        :param Gamma: The breakpoint for a given activation function, i.e.
                        {0} separates ReLU and PReLU into two linear regions.
                        {-0.5, 0.5} separates Sigmoid and tanH into 2 semi-constant region and 1 semi-linear region.
        """
        super().__init__(model)
        self.Gamma = Gamma
        self.num_pattern = len(Gamma) + 1
        self.ratio = ratio

    def hook(self, block_name, layer_name):
        """

        :param block_name:
        :param layer_name:
        :return:
        """

        def fn(layer, input_var, output_var):
            """
            Count the frequency of each pattern
            """
            if random() > self.ratio:
                return
            input_var = input_var[0]
            pattern = get_pattern(input_var, self.Gamma)
            freq = np.array([(pattern == i).sum(axis=0) for i in range(self.num_pattern)], dtype=np.float32)
            if self.features[block_name][layer_name] is None:
                self.features[block_name][layer_name] = freq
            self.features[block_name][layer_name] += freq

        return fn

    def retrieve(self):
        global_entropy = {}
        for block_name, block in self.features.items():
            global_entropy[block_name] = {}
            for layer_name, layer in block.items():
                global_entropy[block_name][layer_name] = self.compute_entropy(layer)
        return global_entropy

    def compute_entropy(self, layer):
        layer /= layer.sum(axis=0).astype(np.float32)
        entropy = np.sum([-layer[0] * np.log(1e-8 + layer[0]) for i in range(self.num_pattern)], axis=0)
        return entropy


def get_pattern(input_var, Gamma):
    boundaries = torch.tensor(Gamma, device=input_var.device)
    return torch.bucketize(input_var, boundaries).cpu().numpy()
