import numpy as np
import torch
from typing import Union
from utils.objects import Device


def mode(x: Union[list, np.array], use_pytorch=False, device=Device.CPU):
    return Mode(x, use_pytorch, device).get_mode


class Mode:
    def __init__(self, x, use_pytorch, device):
        self.x = x
        self.x_partitions = []
        self.use_pytorch = use_pytorch
        self.device = device
        self.dict_list = []
        self.x_reduced = {}
        self.mode = []
        self.calculate_mode()

    def calculate_mode(self):
        self.partition_data()
        self.map()
        self.reduce()
        self.mode_list()

    def mode_list(self):
        max_count = max(self.x_reduced.values())
        for key_i in self.x_reduced.keys():
            if self.x_reduced[key_i] == max_count:
                self.mode.append(key_i)

    def partition_data(self):
        number_of_partitions = max(1, int(np.size(self.x) / 10))
        for partition_i in range(number_of_partitions):
            self.x_partitions.append(self.x[partition_i * 10:min(np.size(self.x), partition_i * 10 + 10)])

    def map(self):
        for partition_i in range(np.shape(self.x_partitions)[0]):
            partition_dict_i = {}
            for observation_i in self.x_partitions[partition_i]:
                if observation_i in partition_dict_i.keys():
                    partition_dict_i[observation_i] += 1
                else:
                    partition_dict_i[observation_i] = 1
            self.dict_list.append(partition_dict_i)

    def reduce(self):
        for dict_i in self.dict_list:
            for key_i in dict_i.keys():
                if key_i in self.x_reduced.keys():
                    self.x_reduced[key_i] += dict_i[key_i]
                else:
                    self.x_reduced[key_i] = dict_i[key_i]

    @property
    def get_mode(self):
        return self.mode

    @property
    def get_partitions(self):
        return self.x_partitions

    @property
    def get_map(self):
        return self.dict_list

    @property
    def get_reduce(self):
        return self.x_reduced
