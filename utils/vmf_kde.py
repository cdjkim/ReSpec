import torch
import torch.nn.functional as F
import math
from torch.special import i0, i1
import scipy.special
import numpy as np


class VonMisesFisherKDE:
    def __init__(self, data, kappa=None, device='cpu'):
        self.device = device
        # Normalize data to unit vectors
        self.data = F.normalize(data, p=2, dim=1).to(self.device)  # Shape: (n_samples, dim)
        self.n, self.dim = self.data.shape

        # Estimate kappa if not provided
        if kappa is None:
            self.kappa = self._estimate_kappa()
            # self.kappa = torch.tensor(1.0)
        else:
            self.kappa = kappa

    @torch.no_grad()
    def _estimate_kappa(self):
        # Estimate the concentration parameter kappa using the mean resultant length.
        R = self.data.sum(dim=0)
        R_norm = R.norm()
        r = R_norm / self.n
        print("r: ", r)

        self.r = r

        kappa = (r * (self.dim - r**2)) / (1 - r**2)

        return kappa

    @torch.no_grad()
    def density(self, query_points, exclude_idx=None):
        # Estimate the log density at the given query points, up to additive constant
        # Normalize query points
        query = F.normalize(query_points, p=2, dim=1).to(self.device)  # Shape: (m, dim)

        # Compute cosine similarity between query points and data points
        # This is equivalent to the dot product since all vectors are unit norm
        # Resulting shape: (m, n)
        if exclude_idx is None:
            data = self.data
        elif isinstance(exclude_idx, int):
            data = torch.cat([self.data[:exclude_idx], self.data[exclude_idx+1:]], dim=0)
        else:
            raise NotImplementedError
        similarity = torch.matmul(query, data.t())  # (m, n)

        # Compute the log density estimate, up to constant
        log_density = torch.logsumexp(self.kappa * similarity, dim=1) - math.log(len(data))
        return log_density

