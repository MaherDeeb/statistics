from enum import Enum


class Device(Enum):
    GPU = "cuda"
    CPU = "cpu"
    TPU = "tpu"


class Framework(Enum):
    Tensorflow = "Tensorflow"
    Pytorch = "Pytorch"
    Numpy = "Numpy"
