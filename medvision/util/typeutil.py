import collections
import numpy as np


def isarrayinstance(x):
    return isinstance(x, (collections.Sequence, np.ndarray))
