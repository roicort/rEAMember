import torch
import numpy as np
from scipy.stats import norm
import random

    
class AssociativeMemory(torch.nn.Module):
    """ 
    A PyTorch version of the Associative Memory.
    """
    def __init__(self, n, m, xi=1, sigma=0.5, iota=1, kappa=1, device=None):
        """
        Parameters
        ----------
        n : int
            The size of the domain (of properties).
        m : int
            The size of the range (of representation).
        xi : int, optional
        sigma : float, optional
            The scaling factor for the relation. Default is 0.5.
        iota : int, optional
        kappa : int, optional
        device : torch.device, optional
            The device to use for the tensors. If None, uses the default device.
            Default is None.
        """
        super().__init__()
        self._n = n
        self._m = m + 1
        self._t = xi
        self._absolute_max = 1023
        self._sigma = sigma * m
        self._iota = iota
        self._kappa = kappa
        self.device = device

        def normpdf(x, mean, sd, scale=1.0):
            eps = 1e-8
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            mean = torch.as_tensor(mean, dtype=torch.float32, device=self.device)
            sd = torch.as_tensor(sd, dtype=torch.float32, device=self.device)
            scale = torch.as_tensor(scale, dtype=torch.float32, device=self.device)
            var = sd ** 2 + eps
            denom = torch.sqrt(torch.tensor(2.0, device=self.device) * torch.pi * var)
            num = torch.exp(-((x - mean) ** 2) / (2 * var))
            return num / (scale * denom)
        self.normpdf = normpdf

        self._scale = self.normpdf(0, 0, self._sigma)
        self._relation = torch.zeros((self._m, self._n), dtype=torch.int16, device=self.device)
        self._iota_relation = torch.zeros((self._m, self._n), dtype=torch.int16, device=self.device)
        self._entropies = torch.zeros(self._n, dtype=torch.float32, device=self.device)
        self._means = torch.zeros(self._n, dtype=torch.float32, device=self.device)
        self._updated = True         # A flag to know whether iota-relation, entropies and means are up to date.

    def __str__(self):
        return str(self._relation)
    
    @property
    def n(self):
        return self._n

    @property
    def m(self):
        return self._m

    @property
    def relation(self):
        return self._relation[:self.m,:]
    
    @property
    def absolute_max_value(self):
        return self._absolute_max

    @property
    def entropies(self):
        if not self._updated:
            self._updated = self.update()
        return self._entropies
    
    @property
    def entropy(self) -> float:
        """Return the entropy of the Associative Memory."""
        return torch.mean(self.entropies)

    @property
    def means(self):
        if not self._updated:
            self._updated = self.update()
        return self._means
    
    @property
    def mean(self):
        return torch.mean(self.means)

    @property
    def iota_relation(self):
        if not self._updated:
            self._updated = self.update()
        return self._iota_relation[:self.m,:]
    
    @property
    def max_value(self):
        maximum = torch.max(self.relation)
        return 1 if maximum == 0 else maximum
    
    @property
    def undefined(self):
        return self.m
    
    @property
    def sigma(self):
        return self._sigma / self.m
    
    @sigma.setter
    def sigma(self, s):
        self._sigma = abs(s*self.m)
        self._scale = self.normpdf(0, 0, self._sigma)
    
    @property
    def kappa(self):
        return self._kappa
    
    @kappa.setter
    def kappa(self, k):
        if (k < 0):
            raise ValueError('Kappa must be a non negative number.')
        self._kappa = k

    @property
    def iota(self):
        return self._iota
    
    @iota.setter
    def iota(self, i):
        if (i < 0):
            raise ValueError('Iota must be a non negative number.')
        self._iota = i
        self._updated = False

    def _update_entropies(self):
        totals = self.relation.sum(dim=0)
        totals = torch.where(totals == 0, torch.tensor(1.0, device=self.device), totals)
        matrix = self.relation / totals
        matrix = -matrix * torch.log2(torch.where(matrix == 0.0, torch.tensor(1.0, device=self.device), matrix))
        self._entropies = matrix.sum(dim=0)

    def _update_means(self):
        sums = torch.sum(self.relation, dim=0, dtype=torch.float32)
        counts = torch.count_nonzero(self.relation, dim=0)
        counts = torch.where(counts == 0, torch.tensor(1.0, device=self.device), counts)
        self._means = (sums / counts) / self.max_value

    def _update_iota_relation(self):
        for j in range(self._n):
            column = self._relation[:,j]
            sum = torch.sum(column)
            if sum == 0:
                self._iota_relation[:,j] = torch.zeros(self._m, dtype=torch.int16, device=self.device)
            else:
                count = torch.count_nonzero(column)
                mean = self.iota * sum / count
                self._iota_relation[:,j] = torch.where(column < mean, 0, column)

    def is_undefined(self, value):
        return value == self.undefined

    @property
    def updated(self):
        self._update_entropies()
        self._update_means()
        self._update_iota_relation()
        return True
    
    def vector_to_relation(self, vector):
        vector = vector.to(dtype=torch.long, device=self.device)
        relation = torch.zeros((self._m, self._n), dtype=torch.bool, device=self.device)
        relation[vector, torch.arange(self.n, device=self.device)] = True
        return relation
    
    def _normalize(self, column, mean, std, scale):
        norm = torch.tensor([self.normpdf(i, mean, std, scale) for i in range(self.m)], device=self.device)
        return norm * column
    
    def normalized(self, j, v):
        return self._normalize(self.relation[:, j], v, self._sigma, self._scale)
    
    # Choose a value for feature i.
    def choose(self, j, v):
        if self.is_undefined(v):
            column = self.relation[:,j]
        else:
            column = self._normalize(
                self.relation[:,j], v, self._sigma, self._scale)
        sum = column.sum()
        n = sum * torch.rand(1, device=self.device).item()
        for i in range(self.m):
            if n < column[i]:
                return i
            n -= column[i]
        return self.m - 1
    
    def _weights(self, vector):
        weights = []
        for i in range(self.n):
            w = 0 if self.is_undefined(vector[i]) \
                else self.relation[vector[i], i]
            weights.append(w)
        return torch.tensor(weights, device=self.device)
    
    def _weight(self, vector):
        return torch.mean(self._weights(vector).float()) / self.max_value
    
    def abstract(self, r_io) -> None:
        self._relation = torch.where(
            self._relation == self.absolute_max_value, 
            self._relation, self._relation + r_io)
        self._updated = False

    def containment(self, r_io):
        return ~r_io[:self.m, :] | self._iota_relation
    
    # Reduces a relation to a function
    def lreduce(self, vector):
        v = torch.tensor([self.choose(i, vector[i]) for i in range(self.n)], device=self.device)
        return v
    
    def validate(self, vector):
        if len(vector) != self.n:
            raise ValueError('Invalid size of the input data. Expected', self.n, 'and given', vector.size(0))
        vector = torch.as_tensor(vector, dtype=torch.float32, device=self.device)
        v = torch.nan_to_num(vector, nan=self.undefined)
        v = torch.where((v > self.m) | (v < 0), self.undefined, v)
        return v.to(torch.int16)
    
    def revalidate(self, vector):
        v = vector.to(torch.float32)
        return torch.where(v == float(self.undefined), torch.nan, v)

    # Operations

    def register(self, vector) -> None:
        vector = self.validate(vector)
        r_io = self.vector_to_relation(vector)
        self.abstract(r_io)

    def recognize(self, vector):
        vector = self.validate(vector)
        r_io = self.vector_to_relation(vector)
        r_io = self.containment(r_io)
        recognized = torch.count_nonzero(r_io[:self.m,:self.n] == False) <= self._t
        weight = self._weight(vector)
        recognized = recognized and (self.mean*self._kappa <= weight)
        return recognized, weight
    
    def mismatches(self, vector):
        vector = self.validate(vector)
        r_io = self.vector_to_relation(vector)
        r_io = self.containment(r_io)
        return torch.count_nonzero(r_io[:self.m,:self.n] == False)

    def recall(self, vector):
        vector = self.validate(vector)
        accept = self.mismatches(vector) <= self._t
        weight = self._weight(vector)
        accept = accept and (self.mean*self._kappa <= weight)
        if accept:
            r_io = self.lreduce(vector)
        else:
            r_io = torch.full((self.n,), self.undefined, dtype=torch.int16, device=self.device)
        r_io = self.revalidate(r_io)
        return r_io, accept, weight



