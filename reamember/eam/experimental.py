import torch


def _gaussian_bank(m: int, sigma_scaled: float, device: torch.device) -> torch.Tensor:
    if sigma_scaled == 0:
        bank = torch.zeros((m + 1, m), dtype=torch.float32, device=device)
        bank[torch.arange(m, device=device), torch.arange(m, device=device)] = 1.0
        bank[m, :] = 1.0
        return bank

    v_indices = torch.arange(m + 1, device=device, dtype=torch.float32).unsqueeze(1)
    j_indices = torch.arange(m, device=device, dtype=torch.float32).unsqueeze(0)
    dist_sq = (j_indices - v_indices) ** 2
    bank = torch.exp(-dist_sq / (2 * sigma_scaled**2))
    bank[m, :] = 1.0
    return bank

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
        self._xi = xi
        self._absolute_max = 2**16 - 1
        self._sigma = sigma
        self._sigma_scaled = sigma * m
        self._iota = iota
        self._kappa = kappa
        self.device = device or torch.device("cpu")

        self._scale = 1.0
        self._relation = torch.zeros(
            (self._n, self._m), dtype=torch.float32, device=self.device
        )
        self._entropies = torch.zeros(self._n, dtype=torch.float32, device=self.device)
        self._means = torch.zeros(self._n, dtype=torch.float32, device=self.device)
        self._iota_relation = torch.zeros(
            (self._n, self._m), dtype=torch.float32, device=self.device
        )
        self._gauss_bank = None
        self._updated = self.update()

    def __str__(self):
        return str(self.relation)

    @classmethod
    def from_relation(
        cls,
        relation,
        xi=0,
        sigma=0.1,
        iota=0,
        kappa=0,
        device=None,
    ):
        relation = torch.as_tensor(relation, dtype=torch.float32, device=device)
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
        memory._relation[:, :m] = relation
        memory._updated = memory.update()
        return memory

    @property
    def n(self):
        return self._n

    @property
    def m(self):
        return self._m-1

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
        return self._iota_relation[:, : self.m]

    @property
    def max_value(self):
        maximum = torch.max(self.relation)
        return 1 if maximum == 0 else maximum

    @property
    def undefined(self):
        return self.m

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, s):
        if s < 0:
            raise ValueError("Sigma must be a non negative number.")
        self._sigma = s
        self._sigma_scaled = abs(s * self.m)
        self._gauss_bank = None

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

    @property
    def xi(self):
        return self._xi

    @xi.setter
    def xi(self, x):
        if (x < 0) or (x > self.n):
            raise ValueError("Xi must be a non negative number.")
        self._xi = x
        self._updated = False

    def _update_entropies(self):
        """
        Update the entropies of the relation.
        """
        totals = self.relation.sum(dim=1)
        totals = torch.where(totals == 0, torch.tensor(1.0, device=self.device), totals)
        matrix = self.relation / totals.unsqueeze(1)
        matrix = -matrix * torch.log2(
            torch.where(matrix == 0.0, torch.tensor(1.0, device=self.device), matrix)
        )
        self._entropies = matrix.sum(dim=1)

    def _update_means(self):
        sums = torch.sum(self.relation, dim=1, dtype=torch.float32)
        counts = torch.count_nonzero(self.relation, dim=1)
        counts = torch.where(counts == 0, torch.tensor(1.0, device=self.device), counts)
        self._means = sums / counts

    def _update_iota_relation(self):
        """
        Update the iota-moderated relation.
        """
        sums = self._relation.sum(dim=1, keepdim=True)
        counts = torch.count_nonzero(self._relation, dim=1).reshape(-1, 1)
        counts = torch.where(counts == 0, torch.tensor(1, device=self.device), counts)
        thresholds = self.iota * sums / counts
        self._iota_relation = torch.where(
            self._relation < thresholds,
            torch.zeros_like(self._relation),
            self._relation,
        )

    def is_undefined(self, value):
        return value == self.undefined

    def update(self):
        self._update_entropies()
        self._update_means()
        self._update_iota_relation()
        return True

    def _get_gauss_bank(self):
        if self._gauss_bank is None:
            self._gauss_bank = _gaussian_bank(self.m, self._sigma_scaled, self.device)
        return self._gauss_bank

    def validate_batch(self, vectors):
        vectors = vectors.to(device=self.device) if torch.is_tensor(vectors) else torch.as_tensor(vectors, device=self.device)
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(0)
        if vectors.ndim != 2 or vectors.size(1) != self.n:
            raise ValueError(
                f"Invalid shape of the input data. Expected [batch, {self.n}] and given {tuple(vectors.shape)}"
            )

        if torch.is_floating_point(vectors):
            vectors = torch.clamp(vectors, 0, self.m - 1)
            vectors = torch.nan_to_num(vectors, nan=float(self.undefined))
        else:
            vectors = torch.clamp(vectors, 0, self.m - 1)
        return vectors.to(torch.long)

    def batch_validate(self, vectors):
        return self.validate_batch(vectors)

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
        vector = self.validate(vector)
        relation = torch.zeros((self._n, self._m), dtype=torch.bool, device=self.device)
        relation[torch.arange(self.n, device=self.device), vector] = True
        return relation

    def vectors_to_relation(self, vectors):
        vectors = self.validate_batch(vectors).to(dtype=torch.long)
        batch_size = vectors.size(0)
        relation = torch.zeros(
            (batch_size, self._n, self._m), dtype=torch.bool, device=self.device
        )
        batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1)
        feature_idx = torch.arange(self.n, device=self.device).unsqueeze(0)
        relation[batch_idx, feature_idx, vectors] = True
        return relation

    def _normalize(self, column, mean, std, scale):
        bank = _gaussian_bank(self.m, float(std), self.device)[: self.m, : self.m]
        mean_idx = int(mean.item()) if torch.is_tensor(mean) else int(mean)
        return bank[mean_idx] * column

    def normalized(self, j, v):
        return self._normalize(self.relation[j], v, self._sigma_scaled, self._scale)

    def _weights(self, vector):
        vector = torch.as_tensor(vector, dtype=torch.long, device=self.device)
        mask = vector != self.undefined
        weights = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        row_idx = torch.arange(self.n, device=self.device)
        weights[mask] = self.relation[row_idx[mask], vector[mask]].float()
        return weights

    def _weight(self, vector):
        return torch.mean(self._weights(vector))

    def _weights_batch(self, vectors):
        vectors = torch.as_tensor(vectors, dtype=torch.long, device=self.device)
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(0)
        mask = vectors != self.undefined
        safe_vectors = torch.where(mask, vectors, torch.zeros_like(vectors))
        feature_idx = torch.arange(self.n, device=self.device).unsqueeze(0)
        weights = self.relation[feature_idx, safe_vectors]
        return torch.where(mask, weights, torch.zeros_like(weights, dtype=torch.float32))

    def _weight_batch(self, vectors):
        return torch.mean(self._weights_batch(vectors).float(), dim=1)

    def _gather_iota_values(self, vectors):
        vectors = self.validate_batch(vectors).to(dtype=torch.long)
        defined_mask = vectors != self.undefined
        safe_vectors = torch.where(defined_mask, vectors, torch.zeros_like(vectors))
        feature_idx = torch.arange(self.n, device=self.device).unsqueeze(0)
        gathered = self.iota_relation[feature_idx, safe_vectors]
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
        rows = torch.arange(self.n, device=self.device).repeat(vectors.size(0))
        cols = vectors.reshape(-1)
        updates = torch.ones_like(cols, dtype=self._relation.dtype)
        delta = torch.zeros((self._n, self._m), dtype=self._relation.dtype, device=self.device)
        delta.index_put_((rows, cols), updates, accumulate=True)
        self._relation = torch.where(
            self._relation == self.absolute_max_value,
            self._relation,
            torch.clamp(self._relation + delta, max=float(self.absolute_max_value)),
        )
        self._updated = False

    def containment(self, r_io):
        return ~r_io[:, : self.m] | self.iota_relation.bool()

    def containment_batch(self, r_io):
        return torch.logical_or(
            ~r_io[:, :, : self.m], self.iota_relation.bool().unsqueeze(0)
        )

    def produce(self, cue):
        cue = self.validate(cue)
        return self.batch_produce(cue.unsqueeze(0)).squeeze(0)

    def batch_produce(self, cues):
        cues = self.batch_validate(cues)
        gauss = self._get_gauss_bank()[cues]
        weights = self._relation[:, : self.m].unsqueeze(0) * gauss
        cumsum_weights = weights.cumsum(dim=2)
        totals = cumsum_weights[:, :, -1]
        random_thresholds = torch.rand(
            totals.shape, dtype=weights.dtype, device=self.device
        ) * totals
        sampled = torch.searchsorted(
            cumsum_weights.contiguous(),
            random_thresholds.unsqueeze(-1),
            right=False,
        ).squeeze(-1)
        return torch.where(totals == 0, torch.full_like(sampled, self.m), sampled)

    def lreduce(self, vector):
        return self.produce(vector)

    def lreduce_batch(self, vectors):
        return self.batch_produce(vectors)

    def validate(self, vector):
        validated = self.validate_batch(vector)
        if validated.size(0) != 1:
            raise ValueError(
                f"Invalid size of the input data. Expected {self.n} and given {validated.numel()}"
            )
        return validated.squeeze(0)
    
    def revalidate(self, vector):
        v = vector.to(torch.float32)
        return torch.where(v == float(self.undefined), torch.nan, v)

    # Operations

    def register(self, vector) -> None:
        vector = self.validate(vector)
        r_io = self.vector_to_relation(vector)
        self.abstract(r_io)

    def batch_register(self, vectors) -> None:
        self.abstract_batch(vectors)

    def register_batch(self, vectors) -> None:
        self.batch_register(vectors)

    def recog_weight(self, cue, validate=True):
        vector = self.validate(cue) if validate else cue
        recognized = self._mismatches(vector) <= self.xi
        weight = self._weight(vector)
        recognized = bool(recognized and (self.kappa * self.mean <= weight))
        return recognized, weight

    def recognize(self, vector):
        return self.recog_weight(vector)

    def batch_recog_weights(self, vectors):
        vectors = self.batch_validate(vectors)
        if not self._updated:
            self._updated = self.update()
        mismatches, weights = self._mismatches_and_weights(vectors)
        recognized = mismatches <= self.xi
        recognized = torch.logical_and(recognized, self.kappa * self.mean <= weights)
        return recognized, weights

    def recognize_batch(self, vectors):
        return self.batch_recog_weights(vectors)

    def _mismatches(self, vector):
        r_io = self.to_relation(vector)
        r_io = self.containment(r_io)
        return torch.count_nonzero(r_io[:, : self.m] == 0)

    def mismatches(self, vector):
        return self._mismatches(self.validate(vector))

    def mismatches_batch(self, vectors):
        vectors = self.batch_validate(vectors)
        mismatches, _ = self._mismatches_and_weights(vectors)
        return mismatches

    def recall(self, cue=None):
        if cue is None:
            cue = torch.full((self.n,), float("nan"), dtype=torch.float32, device=self.device)
        r_io, recognized, weight = self.recall_weights(cue)
        return r_io, recognized, weight

    def recall_weights(self, cue, validate=True):
        vector = self.validate(cue) if validate else cue
        recognized, _ = self.recog_weight(vector, validate=False)
        r_io = self.produce(vector) if recognized else torch.full((self.n,), self.undefined, dtype=torch.long, device=self.device)
        weight = self._weight(r_io)
        r_io = self.revalidate(r_io)
        return r_io, recognized, weight

    def batch_recall(self, cues):
        cues = self.batch_validate(cues)

        if not self._updated:
            self._updated = self.update()
        features = torch.arange(self.n, device=self.device).unsqueeze(0)

        matches = self._iota_relation[features, cues]
        is_mismatch = torch.logical_and(matches == 0, cues != self.undefined)
        mismatches = torch.sum(is_mismatch, dim=1)
        recognized_mask = mismatches <= self.xi

        cue_weights_per_feature = self._relation[features, cues].float()
        cue_weights_per_feature = torch.where(
            cues == self.undefined,
            torch.zeros_like(cue_weights_per_feature),
            cue_weights_per_feature,
        )
        cue_weights = torch.mean(cue_weights_per_feature, dim=1)
        recognized_mask = torch.logical_and(
            recognized_mask,
            cue_weights >= (self.kappa * self.mean),
        )

        memories = torch.full_like(cues, self.undefined)
        if torch.any(recognized_mask):
            rec_indices = torch.nonzero(recognized_mask, as_tuple=False).squeeze(1)
            memories[rec_indices] = self.batch_produce(cues[rec_indices])

        mem_weights_per_feature = self._relation[features, memories].float()
        mem_weights_per_feature = torch.where(
            memories == self.undefined,
            torch.zeros_like(mem_weights_per_feature),
            mem_weights_per_feature,
        )
        final_weights = torch.mean(mem_weights_per_feature, dim=1)
        return memories, recognized_mask, final_weights

    def recall_batch(self, vectors):
        memories, accepted, _ = self.batch_recall(vectors)
        weights = torch.mean(self._weights_batch(memories).float(), dim=1)
        return self.revalidate(memories), accepted, weights