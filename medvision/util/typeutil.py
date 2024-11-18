from collections.abc import Sequence

import numpy as np


def isarrayinstance(x):
    return isinstance(x, (Sequence, np.ndarray))
