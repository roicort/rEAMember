import torch
import numpy as np
import random

class AssociativeMemory(torch.nn.Module):
    def __init__(self, n, m, xi=1, sigma=0.5, iota=1, kappa=1, device=None):
        """
        Parameters
        ----------
        n : int
            The size of the domain (of properties).
        m : int
            The size of the range (of representation).
        tolerance: int
            The number of mismatches allowed between the
            memory content and the cue.
        sigma:
            The standard deviation of the normal distribution
            used in remembering, as percentage of the number of
            characteristics. Default: None, in which case
            half the number of characteristics is used.
        """
        super().__init__()
        self._n = n
        self._m = m + 1
        self._t = xi
        self._absolute_max = 1023
        self._sigma = sigma * m
        self._iota = iota
        self._kappa = kappa
        self._scale = 1.0
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self._relation = torch.zeros((self._m, self._n), dtype=torch.int16, device=self.device)
        self._iota_relation = torch.zeros((self._m, self._n), dtype=torch.int16, device=self.device)
        self._entropies = torch.zeros(self._n, dtype=torch.float, device=self.device)
        self._means = torch.zeros(self._n, dtype=torch.float, device=self.device)
        self._updated = True

    @property
    def n(self):
        return self._n
    @property
    def m(self):
        return self._m - 1
    @property
    def relation(self):
        return self._relation[:self.m, :]
    @property
    def undefined(self):
        return self.m
    def validate(self, vector):
        v = torch.nan_to_num(torch.tensor(vector, device=self.device, dtype=torch.float), nan=self.undefined)
        v = torch.where((v > self.m) | (v < 0), torch.tensor(self.undefined, device=self.device), v)
        return v.to(torch.int)
    def vector_to_relation(self, vector):
        relation = torch.zeros((self._m, self._n), dtype=torch.bool, device=self.device)
        relation[vector, torch.arange(self.n, device=self.device)] = True
        return relation
    def register(self, vector):
        vector = self.validate(vector)
        r_io = self.vector_to_relation(vector)
        self._relation = torch.where(
            self._relation == self._absolute_max,
            self._relation, self._relation + r_io.to(torch.int16))
        self._updated = False
    def recognize(self, vector):
        vector = self.validate(vector)
        r_io = self.vector_to_relation(vector)
        recognized = torch.all(self._relation[vector, torch.arange(self.n, device=self.device)] > 0).item()
        return recognized
    def recall(self, vector):
        vector = self.validate(vector)
        if self.recognize(vector):
            return vector.cpu().numpy(), True
        else:
            return np.full(self.n, self.undefined), False
