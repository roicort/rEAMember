import torch
import numpy as np
from scipy.stats import norm
import random

class AssociativeMemory():
    def __init__(self, n, m, xi=1, sigma=0.5, iota=1, kappa=1):
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

        def normpdf(x, mean, sd, scale=1.0):
            return norm.pdf(x, loc=mean, scale=sd) / scale
        
        self.normpdf = normpdf

        self._scale = self.normpdf(0, 0, self._sigma)

        # It is m+1 to handle partial functions.
        self._relation = np.zeros((self._m, self._n), dtype='int16')
        # Iota moderated relation
        self._iota_relation = np.zeros((self._m, self._n), dtype='int16')
        self._entropies = np.zeros(self._n, dtype=float)
        self._means = np.zeros(self._n, dtype=float)

        # A flag to know whether iota-relation, entropies and means
        # are up to date.
        self._updated = True
    def __str__(self):
        return str(self.relation)

    @property
    def n(self):
        return self._n

    @property
    def m(self):
        return self._m-1

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
        return np.mean(self.entropies)

    @property
    def means(self):
        if not self._updated:
            self._updated = self.update()
        return self._means

    @property
    def mean(self):
        return np.mean(self.means)

    @property
    def iota_relation(self):
        if not self._updated:
            self._updated = self.update()
        return self._iota_relation[:self.m,:]


    @property
    def max_value(self):
        # max_value is used as normalizer by dividing, so it
        # should not be zero.
        maximum = np.max(self.relation)
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

    def update(self):
        self._update_entropies()
        self._update_means()
        self._update_iota_relation()
        return True

    def _update_entropies(self):
        totals = self.relation.sum(axis=0)  # sum of cell values by columns
        totals = np.where(totals == 0, 1, totals)
        matrix = self.relation/totals
        matrix = -matrix*np.log2(np.where(matrix == 0.0, 1.0, matrix))
        self._entropies = matrix.sum(axis=0)

    def _update_means(self):
        sums = np.sum(self.relation, axis=0, dtype=float)
        counts = np.count_nonzero(self.relation, axis=0)
        counts = np.where(counts == 0, 1, counts)
        self._means = (sums/counts)/self.max_value

    def _update_iota_relation(self):
        for j in range(self._n):
            column = self._relation[:,j]
            sum = np.sum(column)
            if sum == 0:
                self._iota_relation[:,j] = np.zeros(self._m, dtype='int16')
            else:
                count = np.count_nonzero(column)
                mean = self.iota*sum/count
                self._iota_relation[:,j] = np.where(column < mean, 0, column)

    def is_undefined(self, value):
        return value == self.undefined

    def vector_to_relation(self, vector):
        relation = np.zeros((self._m, self._n), bool)
        relation[vector, range(self.n)] = True
        return relation

    def _normalize(self, column, mean, std, scale):            
        norm = np.array([self.normpdf(i, mean, std, scale) for i in range(self.m)])
        return norm*column

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
        n = sum*random.random()
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
        return np.array(weights)

    def _weight(self, vector):
        return np.mean(self._weights(vector)) / self.max_value

    def abstract(self, r_io) -> None:
        self._relation = np.where(
            self._relation == self.absolute_max_value, 
            self._relation, self._relation + r_io)
        self._updated = False

    def containment(self, r_io):
        return ~r_io[:self.m, :] | self.iota_relation

    # Reduces a relation to a function
    def lreduce(self, vector):
        v = np.array([self.choose(i, vector[i]) for i in range(self.n)])
        return v

    def validate(self, vector):
        """ It asumes vector is an array of floats, and np.nan
            is used to register an undefined value, but it also 
            considerers any negative number or out of range number
            as undefined.
        """
        if len(vector) != self.n:
            raise ValueError('Invalid size of the input data. Expected', self.n, 'and given', vector.size)
        v = np.nan_to_num(vector, copy=True, nan=self.undefined)
        v = np.where((v > self.m) | (v < 0), self.undefined, v)
        return v.astype('int')

    def revalidate(self, vector):
        v = vector.astype('float')
        return np.where(v == float(self.undefined), np.nan, v)

    def register(self, vector) -> None:
        vector = self.validate(vector)
        r_io = self.vector_to_relation(vector)
        self.abstract(r_io)

    def recognize(self, vector):
        vector = self.validate(vector)
        r_io = self.vector_to_relation(vector)
        r_io = self.containment(r_io)
        recognized = np.count_nonzero(r_io[:self.m,:self.n] == False) <= self._t
        weight = self._weight(vector)
        recognized = recognized and (self.mean*self._kappa <= weight)
        return recognized, weight

    def mismatches(self, vector):
        vector = self.validate(vector)
        r_io = self.vector_to_relation(vector)
        r_io = self.containment(r_io)
        return np.count_nonzero(r_io[:self.m,:self.n] == False)

    def recall(self, vector):
        vector = self.validate(vector)
        accept = self.mismatches(vector) <= self._t
        weight = self._weight(vector)
        accept = accept and (self.mean*self._kappa <= weight)
        if accept:
            r_io = self.lreduce(vector)
        else:
            r_io = np.full(self.n, self.undefined)
        r_io = self.revalidate(r_io)
        return r_io, accept, weight
    
class PytorchAssociativeMemory(torch.nn.Module):
    """ 
    A PyTorch version of the Associative Memory. (WIP)
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

        def normpdf(x, mean, sd, scale=1.0):
            var = sd ** 2
            denom = torch.sqrt(2 * torch.pi * var)
            num = torch.exp(-((x - mean) ** 2) / (2 * var))
            return num / (scale * denom)

        self._scale = normpdf(0, 0, self._sigma)
        self._relation = torch.zeros((self._m, self._n), dtype=torch.int16, device=device)
        self._iota_relation = torch.zeros((self._m, self._n), dtype=torch.int16, device=device)
        self._entropies = torch.zeros(self._n, dtype=torch.float32, device=device)
        self._means = torch.zeros(self._n, dtype=torch.float32, device=device)
        self._updated = True

    def __str__(self):
        return str(self._relation)
    
    @property
    def relation(self):
        return self._relation

    @property
    def iota_relation(self):
        return self._iota_relation

    @property
    def entropies(self):
        return self._entropies

    @property
    def means(self):
        return self._means

    @property
    def updated(self):
        return self._updated
    
    @updated.setter
    def updated(self, value):
        self._updated = value
    
