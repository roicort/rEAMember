import torch
import math
import numpy as np
import random

class TorchAssociativeMemory(torch.nn.Module):
    """
    A PyTorch version of the Associative Memory.
    """

    def __init__(self, n, m, xi=0, sigma=0.1, iota=0, kappa=0, device=None):
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
            var = sd**2 + eps
            denom = torch.sqrt(torch.tensor(2.0, device=self.device) * torch.pi * var)
            num = torch.exp(-((x - mean) ** 2) / (2 * var))
            return num / (scale * denom)

        self.normpdf = normpdf

        self._scale = self.normpdf(0, 0, self._sigma)
        self._relation = torch.zeros(
            (self._m, self._n), dtype=torch.float32, device=self.device
        )
        self._entropies = torch.zeros(self._n, dtype=torch.float32, device=self.device)
        self._means = torch.zeros(self._n, dtype=torch.float32, device=self.device)
        self._iota_relation = torch.zeros(
            (self._m, self._n), dtype=torch.float32, device=self.device
        )
        self._kernel_cache = None
        self._updated = True  # A flag to know whether iota-relation, entropies and means are up to date.
        self._update_kernel_cache()

    def __str__(self):
        return str(self._relation)

    @property
    def n(self):
        return self._n

    @property
    def m(self):
        return self._m-1

    @property
    def relation(self):
        return self._relation[: self.m, :]

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
        """Return the iota-moderated relation."""
        if not self._updated:
            self._updated = self.update()
        return self._iota_relation[: self.m, :]

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
        self._sigma = abs(s * self.m)
        self._scale = self.normpdf(0, 0, self._sigma)
        self._update_kernel_cache()

    @property
    def kappa(self):
        return self._kappa

    @kappa.setter
    def kappa(self, k):
        if k < 0:
            raise ValueError("Kappa must be a non negative number.")
        self._kappa = k

    @property
    def iota(self):
        return self._iota

    @iota.setter
    def iota(self, i):
        if i < 0:
            raise ValueError("Iota must be a non negative number.")
        self._iota = i
        self._updated = False

    def _update_entropies(self):
        """
        Update the entropies of the relation.
        """
        totals = self.relation.sum(dim=0)
        totals = torch.where(totals == 0, torch.tensor(1.0, device=self.device), totals)
        matrix = self.relation / totals
        matrix = -matrix * torch.log2(
            torch.where(matrix == 0.0, torch.tensor(1.0, device=self.device), matrix)
        )
        self._entropies = matrix.sum(dim=0)

    def _update_means(self):
        sums = torch.sum(self.relation, dim=0, dtype=torch.float32)
        counts = torch.count_nonzero(self.relation, dim=0)
        counts = torch.where(counts == 0, torch.tensor(1.0, device=self.device), counts)
        self._means = (sums / counts) / self.max_value

    def _update_iota_relation(self):
        """
        Update the iota-moderated relation.
        """
        columns = self._relation
        sums = columns.sum(dim=0)
        counts = torch.count_nonzero(columns, dim=0)
        means = torch.where(counts == 0, 0, self.iota * sums / counts)
        mask = columns < means.unsqueeze(0)
        self._iota_relation = torch.where(mask, torch.zeros_like(columns), columns)

    def is_undefined(self, value):
        return value == self.undefined

    def update(self):
        self._update_entropies()
        self._update_means()
        self._update_iota_relation()
        return True

    def _update_kernel_cache(self):
        row_indices = torch.arange(self.m, device=self.device, dtype=torch.float32).unsqueeze(1)
        means = torch.arange(self.m, device=self.device, dtype=torch.float32).unsqueeze(0)
        kernel = self.normpdf(row_indices, means, self._sigma, self._scale)
        undefined_kernel = torch.ones((self.m, 1), dtype=torch.float32, device=self.device)
        self._kernel_cache = torch.cat([kernel, undefined_kernel], dim=1)

    def validate_batch(self, vectors):
        if torch.is_tensor(vectors):
            vectors = vectors.to(device=self.device)
            if vectors.ndim == 1:
                vectors = vectors.unsqueeze(0)
            if vectors.ndim != 2 or vectors.size(1) != self.n:
                raise ValueError(
                    f"Invalid shape of the input data. Expected [batch, {self.n}] and given {tuple(vectors.shape)}"
                )
            if torch.is_floating_point(vectors):
                vectors = torch.nan_to_num(vectors, nan=float(self.undefined))
                vectors = torch.where(
                    (vectors > self.m) | (vectors < 0),
                    torch.full_like(vectors, float(self.undefined)),
                    vectors,
                )
                return vectors.to(torch.int16)

            undefined_fill = torch.full_like(vectors, self.undefined)
            vectors = torch.where((vectors > self.m) | (vectors < 0), undefined_fill, vectors)
            return vectors.to(torch.int16)

        vectors = torch.as_tensor(vectors, dtype=torch.float32, device=self.device)
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(0)
        if vectors.ndim != 2 or vectors.size(1) != self.n:
            raise ValueError(
                f"Invalid shape of the input data. Expected [batch, {self.n}] and given {tuple(vectors.shape)}"
            )
        vectors = torch.nan_to_num(vectors, nan=float(self.undefined))
        vectors = torch.where(
            (vectors > self.m) | (vectors < 0),
            torch.full_like(vectors, float(self.undefined)),
            vectors,
        )
        return vectors.to(torch.int16)

    def vector_to_relation(self, vector):
        """
        Convert a vector to a relation.
        Parameters
        ----------
        vector : torch.Tensor
            A 1D tensor of shape [n] representing the input vector.
        Returns
        -------
        torch.Tensor
            A 2D tensor of shape [m, n] representing the relation.
        """
        return self.vectors_to_relation(self.validate_batch(vector)).squeeze(0)

    def vectors_to_relation(self, vectors):
        vectors = self.validate_batch(vectors).to(dtype=torch.long)
        batch_size = vectors.size(0)
        relation = torch.zeros(
            (batch_size, self._m, self._n), dtype=torch.bool, device=self.device
        )
        batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1)
        column_idx = torch.arange(self.n, device=self.device).unsqueeze(0)
        relation[batch_idx, vectors, column_idx] = True
        return relation

    def _normalize(self, columns, means, std, scale):
        row_indices = torch.arange(self.m, device=self.device).unsqueeze(1) # Crear un vector columna de índices: [m, 1]
        norm_weights = self.normpdf(row_indices, means, std, scale) # Calcular los pesos para todas las columnas a la vez usando broadcasting
        return norm_weights * columns # Multiplicar todas las columnas por sus respectivos pesos a la vez

    def normalized(self, j, v):
        column = self.relation[:, j].float().unsqueeze(1)
        mean = torch.as_tensor([v], dtype=torch.float32, device=self.device)
        return self._normalize(column, mean, self._sigma, self._scale).squeeze(1)

    def _kernel_values(self, vectors):
        if not torch.is_tensor(vectors):
            vectors = torch.as_tensor(vectors, device=self.device)
        else:
            vectors = vectors.to(device=self.device)
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(0)
        vectors = vectors.to(dtype=torch.long)
        kernels = self._kernel_cache[:, vectors]
        return kernels.permute(1, 0, 2)

    def _weights(self, vector):
        # Vector debe estar en el dispositivo correcto y ser tipo long
        vector = torch.as_tensor(vector, dtype=torch.long, device=self.device)
        # Creamos una máscara para los valores definidos
        mask = (vector != self.undefined)
        # Inicializamos los pesos en cero
        weights = torch.zeros(self.n, dtype=self._relation.dtype, device=self.device)
        # Solo asignamos los pesos donde el valor no es undefined
        idx = torch.arange(self.n, device=self.device)
        weights[mask] = self._relation[vector[mask], idx[mask]]
        return weights

    def _weight(self, vector):
        return torch.mean(self._weights(vector).float()) / self.max_value

    def _weights_batch(self, vectors):
        vectors = self.validate_batch(vectors).to(dtype=torch.long)
        mask = vectors != self.undefined
        column_idx = torch.arange(self.n, device=self.device).unsqueeze(0)
        weights = self._relation[vectors, column_idx]
        return torch.where(mask, weights, torch.zeros_like(weights))

    def _weight_batch(self, vectors):
        return torch.mean(self._weights_batch(vectors).float(), dim=1) / self.max_value

    def _gather_iota_values(self, vectors):
        vectors = self.validate_batch(vectors).to(dtype=torch.long)
        defined_mask = vectors != self.undefined
        safe_vectors = torch.where(defined_mask, vectors, torch.zeros_like(vectors))
        column_idx = torch.arange(self.n, device=self.device).unsqueeze(0)
        gathered = self.iota_relation[safe_vectors, column_idx]
        return gathered, defined_mask

    def _mismatches_and_weights(self, vectors):
        gathered, defined_mask = self._gather_iota_values(vectors)
        mismatches = torch.sum(torch.logical_and(defined_mask, gathered == 0), dim=1)
        weights = self._weight_batch(vectors)
        return mismatches, weights

    def abstract(self, r_io) -> None:
        r_io = r_io.to(dtype=self._relation.dtype)
        self._relation = torch.where(
            self._relation == self.absolute_max_value,
            self._relation,
            torch.clamp(self._relation + r_io, max=float(self.absolute_max_value)),
        )
        self._updated = False

    def abstract_batch(self, vectors) -> None:
        vectors = self.validate_batch(vectors).to(dtype=torch.long)
        rows = vectors.reshape(-1)
        cols = torch.arange(self.n, device=self.device).repeat(vectors.size(0))
        updates = torch.ones_like(rows, dtype=self._relation.dtype)
        delta = torch.zeros((self._m, self._n), dtype=self._relation.dtype, device=self.device)
        delta.index_put_((rows, cols), updates, accumulate=True)
        self._relation = torch.where(
            self._relation == self.absolute_max_value,
            self._relation,
            torch.clamp(self._relation + delta, max=float(self.absolute_max_value)),
        )
        self._updated = False

    def containment(self, r_io):
        return ~r_io[: self.m, :] | self.iota_relation

    def containment_batch(self, r_io):
        return torch.logical_or(
            ~r_io[:, : self.m, :], self.iota_relation.bool().unsqueeze(0)
        )

    def lreduce(self, vector):
        """Reduces a relation to a function, using the input vector to guide the sampling."""
        return self.lreduce_batch(vector).squeeze(0)

    def lreduce_batch(self, vectors):
        vectors = self.validate_batch(vectors)
        batch_size = vectors.size(0)
        result = torch.empty((batch_size, self.n), dtype=torch.int16, device=self.device)
        relation = self.relation
        chunk_size = max(1, min(self.n, 64))

        for start in range(0, self.n, chunk_size):
            end = min(start + chunk_size, self.n)
            vector_chunk = vectors[:, start:end]
            column_chunk = relation[:, start:end].unsqueeze(0)
            kernel_values = self._kernel_values(vector_chunk)
            weighted_columns = column_chunk * kernel_values

            totals = weighted_columns.sum(dim=1)
            totals = torch.where(totals == 0, torch.ones_like(totals), totals)
            random_thresholds = totals * torch.rand(
                totals.shape, dtype=weighted_columns.dtype, device=self.device
            )
            cumsum = torch.cumsum(weighted_columns, dim=1).transpose(1, 2).contiguous()
            sampled = torch.searchsorted(
                cumsum,
                random_thresholds.unsqueeze(-1),
                right=False,
            ).squeeze(-1)
            result[:, start:end] = torch.clamp(sampled, max=self.m - 1).to(torch.int16)

        return result

    def validate(self, vector):
        validated = self.validate_batch(vector)
        if validated.size(0) != 1:
            raise ValueError(
                "Invalid size of the input data. Expected",
                self.n,
                "and given",
                validated.size(),
            )
        return validated.squeeze(0)
    
    def revalidate(self, vector):
        v = vector.to(torch.float32)
        return torch.where(v == float(self.undefined), torch.nan, v)

    # Operations

    def register(self, vector) -> None:
        self.register_batch(vector)

    def register_batch(self, vectors) -> None:
        self.abstract_batch(vectors)

    def recognize(self, vector):
        recognized, weight = self.recognize_batch(vector)
        return recognized.squeeze(0), weight.squeeze(0)

    def recognize_batch(self, vectors):
        vectors = self.validate_batch(vectors)
        mismatches, weights = self._mismatches_and_weights(vectors)
        recognized = mismatches <= self._t
        recognized = torch.logical_and(recognized, self.mean * self._kappa <= weights)
        return recognized, weights

    def mismatches(self, vector):
        return self.mismatches_batch(vector).squeeze(0)

    def mismatches_batch(self, vectors):
        vectors = self.validate_batch(vectors)
        mismatches, _ = self._mismatches_and_weights(vectors)
        return mismatches

    def recall(self, vector):
        recalled, accepted, weight = self.recall_batch(vector)
        return recalled.squeeze(0), accepted.squeeze(0), weight.squeeze(0)

    def recall_batch(self, vectors):
        vectors = self.validate_batch(vectors)
        mismatches, weights = self._mismatches_and_weights(vectors)
        accepted = torch.logical_and(
            mismatches <= self._t, self.mean * self._kappa <= weights
        )
        recalled = self.lreduce_batch(vectors)
        fallback = torch.full_like(recalled, self.undefined)
        recalled = torch.where(accepted.unsqueeze(1), recalled, fallback)
        return self.revalidate(recalled), accepted, weights