import numpy as np
import torch
from typing import Union
from utils.objects import Device


def mean(x: Union[list, np.array], use_pytorch=False, device=Device.CPU):
    return Mean(x, use_pytorch, device).get_mean


class Mean:
    def __init__(self, x, use_pytorch, device):
        self.x = x
        self.use_pytorch = use_pytorch
        self.device = device
        self.mean = None
        self.calculate_mean()

    def calculate_mean(self):
        if self.use_pytorch:
            if self.device == Device.CPU:
                self.pytorch_cpu_mean()
            elif self.device == Device.GPU:
                self.pytorch_gpu_mean()
            elif self.device == Device.TPU:
                pass
        else:
            self.numpy_mean()

    def numpy_mean(self):
        self.mean = np.divide(np.sum(self.x), np.size(self.x))

    def pytorch_cpu_mean(self):
        x_tensor = torch.tensor(self.x, device=Device.CPU.value)
        self.mean = torch.divide(torch.sum(x_tensor), x_tensor.size()[0])

    def pytorch_gpu_mean(self):
        x_tensor = torch.tensor(self.x, device=Device.GPU.value)
        self.mean = torch.divide(torch.sum(x_tensor), x_tensor.size()[0])

    @property
    def get_mean(self):
        return self.mean
