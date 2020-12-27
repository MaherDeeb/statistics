from enum import Enum


class Device(Enum):
    GPU = "cuda"
    CPU = "cpu"
    TPU = "tpu"
