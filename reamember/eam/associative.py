import math
import numpy as np

# Copyright [2020] Luis Alberto Pineda Cortés, Gibrán Fuentes Pineda,
# Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# File originally create by Raul Peralta-Lozada.


class NumpyAssociativeMemory:
    def __init__(self, n: int, m: int,
        xi = 0, sigma=0.1,
        iota = 0, kappa=0, device=None):
        """
        Parameters
        ----------
        n : int
            The size of the domain (of properties).
        m : int
            The size of the range (of representation).
        """
        def normpdf(x, mean, sd, scale = 1.0):
            var = float(sd)**2
            denom = (2*math.pi*var)**.5
            num = math.exp(-(float(x)-float(mean))**2/(2*var))
            return num/(scale*denom)
        self._n = n
        # We need an extra feature to represent the 'undefined' value, so we use m+1 internally.
        self._m = m + 1
        self._xi = xi
        self._sigma = sigma
        self._sigma_scaled = sigma * m
        self._gauss_bank = None  # To be computed on demand for optimization
        self._iota = iota
        self._kappa = kappa
        self._scale = 1.0 / normpdf(0, 0, self._sigma_scaled)
        self._absolute_max = 2**16 - 1

        # It is m+1 to handle partial functions.
        self._relation = np.zeros((self._n, self._m), dtype=int)
        # Iota moderated relation
        self._iota_relation = np.zeros((self._n, self._m), dtype=int)
        self._entropies = np.zeros(self._n, dtype=float)
        self._means = np.zeros(self._n, dtype=float)

        # A flag to know whether iota-relation, entropies and means
        # are up to date.
        self._updated = self.update()
        print(
            f'Memory {{n: {self.n}, m: {self.m}, '
            + f'xi: {self.xi}, iota: {self.iota}, '
            + f'kappa: {self.kappa}, sigma: {self.sigma}}}, has been created'
        )

    @classmethod
    def from_relation(
        cls,
        relation: np.ndarray,
        xi=0,
        sigma=0.1,
        iota=0,
        kappa=0,
        device=None,
    ):
        n, m = relation.shape
        memory = cls(
            n=n,
            m=m,
            xi=xi,
            sigma=sigma,
            iota=iota,
            kappa=kappa,
            device=device,
        )
        memory._relation[:, :m] = np.asarray(relation)
        memory._updated = memory.update()
        return memory

    def __str__(self):
        return str(self.relation)

    @property
    def n(self):
        return self._n

    @property
    def m(self):
        return self._m - 1

    @property
    def relation(self):
        return self._relation[:, : self.m]

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
        return self._iota_relation[:, : self.m]

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
    def undefined_output(self):
        return np.full(self.n, np.nan)

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, s):
        if s < 0:
            raise ValueError('Sigma must be a non negative number.')
        self._sigma = s
        self._sigma_scaled = abs(s * self.m)
        self._scale = normpdf(0, 0, self._sigma_scaled)
        self._gauss_bank = None  # Invalidate precomputed bank

    def _get_gauss_bank(self):
        """Precomputes Gaussian windows for all possible cue values."""
        if self._gauss_bank is None:
            # Bank shape: (m + 1, m). Each row v is the Gaussian window for cue value v.
            v_indices = np.arange(self.m + 1)
            j_indices = np.arange(self.m)
            dist_sq = (j_indices[None, :] - v_indices[:, None]) ** 2
            self._gauss_bank = np.exp(-dist_sq / (2 * self._sigma_scaled**2))
            # The last row (v = undefined) results in no modulation (raw relation)
            self._gauss_bank[self.m, :] = 1.0
        return self._gauss_bank

    @property
    def kappa(self):
        return self._kappa

    @kappa.setter
    def kappa(self, k):
        if k < 0:
            raise ValueError('Kappa must be a non negative number.')
        self._kappa = k

    @property
    def iota(self):
        return self._iota

    @iota.setter
    def iota(self, i):
        if i < 0:
            raise ValueError('Iota must be a non negative number.')
        self._iota = i
        self._updated = False

    @property
    def xi(self):
        return self._xi

    @xi.setter
    def xi(self, x):
        if (x < 0) or (x > self.n):
            raise ValueError('Xi must be a non negative number.')
        self._xi = x
        self._updated = False

    def register(self, value) -> None:
        vector = self.validate(value)
        r_io = self.to_relation(vector)
        self.abstract(r_io)

    def batch_register(self, values) -> None:
        """
        Registers a batch of values (e.g., an entire dataset) at once.
        values: A 2D array of shape (number_of_samples, n)
        """
        values = np.asanyarray(values)
        values = self.batch_validate(values)

        # Aggregates counts for each feature-value pair
        # For each column (row, feature), we count how many times each value appears in the batch
        batch_counts = np.zeros((self._n, self._m), dtype=int)
        for i in range(self._n):
            # bincount is extremely fast for this operation
            batch_counts[i] = np.bincount(values[:, i], minlength=self._m)

        # Adds all counts to the relation and clip at the absolute maximum value
        # This replaces the iterative np.where calls
        new_relation = self._relation + batch_counts
        self._relation = np.clip(new_relation, 0, self.absolute_max_value).astype(int)

        # Flag that means/entropies need recalculation
        self._updated = False

    def recognize(self, cue, validate=True):
        recognized, weight = self.recog_weight(cue, validate)
        return recognized, weight

    def recog_weight(self, cue, validate=True):
        vector = self.validate(cue) if validate else cue
        recognized = self._mismatches(vector) <= self.xi
        weight = np.mean(self._weights(vector))
        recognized = recognized and (self.kappa * self.mean <= weight)
        return recognized, weight

    def recall(self, cue=None):
        if cue is None:
            cue = np.full(self.n, np.nan)
        r_io, recognized, weight = self.recall_weights(cue)
        return r_io, recognized, weight

    def recall_weights(self, cue, validate=True):
        vector = self.validate(cue) if validate else cue
        recognized, _ = self.recog_weight(vector, validate=False)
        r_io = self.produce(vector) if recognized else np.full(self.n, self.undefined)
        weight = np.mean(self._weights(r_io))
        r_io = self.revalidate(r_io)
        return r_io, recognized, weight

    def batch_recall(self, cues):
        """Vectorized recognition and conditional production."""
        cues = self.batch_validate(cues)

        if not self._updated:
            self._updated = self.update()  # Ensure iota_relation and mean are current
        features = np.arange(self.n)[None, :]

        # 2. Recognition Logic (using the input Cue)
        # Iota Condition: Check containment in thresholded relation
        matches = self._iota_relation[features, cues]
        is_mismatch = (matches == 0) & (cues != self.undefined)
        mismatches = np.sum(is_mismatch, axis=1)
        recognized_mask = mismatches <= self.xi

        # Kappa Condition: Check average weight of the cue
        cue_weights_per_feature = self._relation[features, cues].astype(float)
        cue_weights_per_feature = np.where(
            cues == self.undefined, 0.0, cue_weights_per_feature
        )
        cue_weights = np.mean(cue_weights_per_feature, axis=1)
        recognized_mask &= cue_weights >= (self.kappa * self.mean)

        # 3. Production: ONLY for recognized cues
        memories = np.full(cues.shape, self.undefined, dtype=int)
        if np.any(recognized_mask):
            rec_indices = np.where(recognized_mask)[0]
            memories[rec_indices] = self.batch_produce(cues[rec_indices])

        # 4. Weight Calculation: Based on RECOVERED memories (consistency with recall())
        # Note: Samples that were not recognized will have a weight of 0.0
        mem_weights_per_feature = self._relation[features, memories].astype(float)
        mem_weights_per_feature = np.where(
            memories == self.undefined, 0.0, mem_weights_per_feature
        )
        final_weights = np.mean(mem_weights_per_feature, axis=1)
        return memories, recognized_mask, final_weights

    def abstract(self, r_io) -> None:
        self._relation = np.where(
            self._relation == self.absolute_max_value,
            self._relation,
            self._relation + r_io,
        )
        self._updated = False

    def _mismatches(self, vector):
        r_io = self.to_relation(vector)
        r_io = self.containment(r_io)
        return np.count_nonzero(r_io[: self.n, : self.m] == 0)

    def containment(self, r_io):
        return ~r_io[:, : self.m] | self.iota_relation

    def produce(self, cue):
        j_indices = np.arange(self.m)

        # Identify which features have defined values, and
        # get float values of the relation for optimization with nump.
        defined_mask = ~self.is_undefined(cue)
        weights = self.relation.astype(float)

        if np.any(defined_mask):
            # Calculate Gaussian windows for all features simultaneously
            dist_sq = (j_indices[None, :] - cue[:, None]) ** 2
            gauss = np.exp(-dist_sq / (2 * self._sigma_scaled**2))

            # Modulate only the rows with defined cues
            weights[defined_mask] *= gauss[defined_mask]

        # Generate random values scaled by the total weight of each row
        cumsum_weights = weights.cumsum(axis=1)
        totals = cumsum_weights[:, -1]
        r = np.random.rand(self.n) * totals

        # Find the first index where cumsum exceeds the random value
        v = (cumsum_weights < r[:, None]).sum(axis=1)

        # Handle rows with zero total weight (fallback to undefined)
        v = np.where(totals == 0, self.undefined, v)
        return v

    def batch_produce(self, cues):
        """Optimized constructive retrieval using precomputed Gaussian bank."""
        S, n = cues.shape
        m = self.m

        # Use precomputed bank indexing: (S, n, m)
        # This replaces np.exp calculation for every sample
        gauss = self._get_gauss_bank()[cues]

        # Apply to relation: (n, m) broadcast to (S, n, m)
        weights = self._relation[None, :, :m] * gauss

        # Sampling logic
        cumsum_weights = weights.cumsum(axis=2)
        totals = cumsum_weights[:, :, -1]
        r = np.random.rand(S, n) * totals

        v = (cumsum_weights < r[:, :, None]).sum(axis=2)
        return np.where(totals == 0, m, v)

    def _normalize(self, column, mean, std, scale):
        norm = np.array([normpdf(i, mean, std, scale) for i in range(self.m)])
        return norm * column

    def to_relation(self, cue):
        relation = np.zeros((self._n, self._m), dtype=bool)
        relation[range(self.n), cue] = True
        return relation

    def validate(self, cue):
        """It asumes vector is an array of floats, and np.nan
        may be used to register an undefined value. Values out of
        range are clipped to the closest valid value.
        """
        cue = np.asanyarray(cue)
        if cue.shape[-1] != self.n:
            raise ValueError(
                'Invalid size of the input data. '
                + f'Expected {self.n} and given {cue.size}'
            )
        v = np.clip(cue, 0, self.m - 1)
        v = np.nan_to_num(v, copy=True, nan=self.undefined)
        return v.astype('int')

    def batch_validate(self, cues):
        if cues.shape[-1] != self.n:
            raise ValueError(
                'Invalid size of the input data. '
                + f'Expected {self.n} and given {cues.shape[-1]}'
            )
        # Runs the validation for the whole batch
        # (Updated to handle 2D inputs)
        cues = np.clip(cues, 0, self.m - 1)
        cues = np.nan_to_num(cues, copy=True, nan=self.undefined).astype(int)
        return cues

    def revalidate(self, vector):
        v = vector.astype('float')
        return np.where(v == float(self.undefined), np.nan, v)

    def _weight(self, vector):
        return np.mean(self._weights(vector))

    def _weights(self, vector):
        # Use advanced indexing to pull all weights in one step
        mask = ~self.is_undefined(vector)
        weights = np.zeros(self.n)
        # np.arange(self.n)[mask] ensures we only index valid rows
        weights[mask] = self.relation[np.arange(self.n)[mask], vector[mask]]
        return weights

    def is_undefined(self, value):
        return value == self.undefined

    def update(self):
        self._update_entropies()
        self._update_means()
        self._update_iota_relation()
        return True

    def _update_entropies(self):
        totals = self.relation.sum(axis=1)  # sum of cell values by columns
        totals = np.where(totals == 0, 1, totals)
        matrix = self.relation / totals[:, None]
        matrix = -matrix * np.log2(np.where(matrix == 0.0, 1.0, matrix))
        self._entropies = matrix.sum(axis=1)

    def _update_means(self):
        sums = np.sum(self.relation, axis=1, dtype=float)
        counts = np.count_nonzero(self.relation, axis=1)
        counts = np.where(counts == 0, 1, counts)
        self._means = sums / counts

    def _update_iota_relation(self):
        # Calculates the sum and the count of non-zero entries per column
        sums = self._relation.sum(axis=1, keepdims=True)
        counts = np.count_nonzero(self._relation, axis=1).reshape(-1, 1)

        # Avoid division by zero for empty rows
        counts = np.where(counts == 0, 1, counts)
        thresholds = self.iota * sums / counts

        # Apply thresholding to the entire table at once
        self._iota_relation = np.where(self._relation < thresholds, 0, self._relation)