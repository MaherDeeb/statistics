import numpy as np
import torch
from typing import Union
from utils.objects import Device


def median(x: Union[list, np.array], use_pytorch=False, device=Device.CPU):
    return Median(x, use_pytorch, device).get_median


class Median:
    def __init__(self, x, use_pytorch, device):
        self.x = x
        self.use_pytorch = use_pytorch
        self.device = device
        self.median = None
        self.calculate_median()

    def calculate_median(self):
        if self.use_pytorch:
            if self.device == Device.CPU:
                self.pytorch_cpu_median()
            elif self.device == Device.GPU:
                self.pytorch_gpu_median()
            elif self.device == Device.TPU:
                pass
        else:
            self.numpy_median()

    def numpy_median(self):
        self.x = np.sort(self.x)
        if self.is_odd():
            self.median = self.x[int((self.x.size - 1) / 2)]
        else:
            self.median = (self.x[int(self.x.size / 2)] + self.x[int(self.x.size / 2 + 1)]) / 2

    def pytorch_cpu_median(self):
        self.x = torch.sort(torch.tensor(self.x, device=Device.CPU.value)).values
        if self.is_odd():
            self.median = self.x[int((self.x.size()[0] - 1) / 2)]
        else:
            self.median = (self.x[int(self.x.size()[0] / 2)] + self.x[int(self.x.size()[0] / 2 + 1)]) / 2

    def pytorch_gpu_median(self):
        self.x = torch.sort(torch.tensor(self.x, device=Device.GPU.value)).values
        if self.is_odd():
            self.median = self.x[int((self.x.size()[0] - 1) / 2)]
        else:
            self.median = (self.x[int(self.x.size()[0] / 2)] + self.x[int(self.x.size()[0] / 2 + 1)]) / 2

    def is_odd(self):
        try:
            return False if not self.x.size % 2 else True
        except Exception as e:
            print(f"Error: {e}")
            return False if not self.x.size()[0] % 2 else True

    @property
    def get_median(self):
        return self.median
