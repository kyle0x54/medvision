from collections import OrderedDict

import numpy as np


class AverageMeter:
    def __init__(self):
        self.val_history = OrderedDict()
        self.num_history = OrderedDict()
        self.output = OrderedDict()
        self.ready = False

    def clear(self):
        self.val_history.clear()
        self.num_history.clear()
        self.clear_output()

    def clear_output(self):
        self.output.clear()
        self.ready = False

    def update(self, variables, count=1):
        assert isinstance(variables, dict)
        for key, var in variables.items():
            if key not in self.val_history:
                self.val_history[key] = []
                self.num_history[key] = []
            self.val_history[key].append(var)
            self.num_history[key].append(count)

    def average(self, n=0):
        """Average latest n values or all values"""
        assert n >= 0
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.num_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg
        self.ready = True
