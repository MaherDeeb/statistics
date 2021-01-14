import numpy as np
import torch
from typing import Union
from utils.objects import Device, Framework
import tensorflow as tf


def mean(x: Union[list, np.array], framework: Framework = Framework.Numpy, device=Device.CPU):
    return Mean(x, framework, device).get_mean


class Mean:
    def __init__(self, x, framework, device):
        self.x = x
        self.framework = framework
        self.device = device
        self.mean = None
        self.calculate_mean()

    def calculate_mean(self):
        if self.framework == Framework.Pytorch:
            if self.device == Device.CPU:
                self.pytorch_cpu_mean()
            elif self.device == Device.GPU:
                self.pytorch_gpu_mean()
            elif self.device == Device.TPU:
                pass
        elif self.framework == Framework.Tensorflow:
            self.tensorflow_mean()
        elif self.framework == Framework.Numpy:
            self.numpy_mean()

    def numpy_mean(self):
        self.mean = np.divide(np.sum(self.x), np.size(self.x))

    def pytorch_cpu_mean(self):
        x_tensor = torch.tensor(self.x, device=Device.CPU.value)
        self.mean = torch.divide(torch.sum(x_tensor), x_tensor.size()[0])

    def pytorch_gpu_mean(self):
        x_tensor = torch.tensor(self.x, device=Device.GPU.value)
        self.mean = torch.divide(torch.sum(x_tensor), x_tensor.size()[0])

    def tensorflow_mean(self):
        x_tensor = tf.constant(self.x, dtype=tf.float32)
        self.mean = tf.divide(tf.reduce_sum(x_tensor), tf.cast(x_tensor.shape, tf.float32)).numpy()[0]

    @property
    def get_mean(self):
        return self.mean
