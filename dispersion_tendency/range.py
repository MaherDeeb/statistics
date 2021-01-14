import numpy as np
import torch
from typing import Union
from utils.objects import Device


def data_range(x: Union[list, np.array], use_pytorch=False, device=Device.CPU):
    return Range(x, use_pytorch, device).get_range


class Range:
    def __init__(self, x, use_pytorch, device):
        self.x = x
        self.use_pytorch = use_pytorch
        self.device = device
        self.range = None
        self.calculate_range()

    def calculate_range(self):
        if self.use_pytorch:
            if self.device == Device.CPU:
                self.pytorch_cpu_range()
            elif self.device == Device.GPU:
                self.pytorch_gpu_range()
            elif self.device == Device.TPU:
                pass
        else:
            self.numpy_range()

    def numpy_range(self):
        self.range = np.max(self.x) - np.min(self.x)

    def pytorch_cpu_range(self):
        x_tensor = torch.tensor(self.x, device=Device.CPU.value)
        self.range = torch.max(x_tensor).values() - torch.min(x_tensor).values()

    def pytorch_gpu_range(self):
        x_tensor = torch.tensor(self.x, device=Device.GPU.value)
        self.range = torch.max(x_tensor).values() - torch.min(x_tensor).values()

    @property
    def get_range(self):
        return self.range
