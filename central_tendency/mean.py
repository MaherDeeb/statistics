import numpy as np
from typing import Union


def mean(x: Union[list, np.array]):

    return np.sum(x) / np.size(x)
