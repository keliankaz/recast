import torch
from torch.distributions import constraints

from .distribution import Distribution


class GutenbergRichter(Distribution):
    arg_constraints = {"b": constraints.positive}

    def __init__(self, b, mag_min=2.0, mag_max=10):
        self.b = b
        self.mag_min = mag_min
        self.mag_max = mag_max
        batch_shape = b.shape
        super().__init__(batch_shape, validate_args=False)

    def log_hazard(self, x):
        return torch.zeros_like(x)

    def log_survival(self, x):
        return torch.zeros_like(x)

    def log_prob(self, x):
        # TODO: We ignore the NLL for the magnitude distribution.
        return torch.zeros_like(x)

    def rsample(self, sample_shape=torch.Size()):
        shape = torch.Size(sample_shape) + self.batch_shape
        u = torch.empty(shape, device=self.b.device, dtype=self.b.dtype).uniform_()
        return self.b.reciprocal().neg() * torch.log10(
            -u * (10 ** (-self.b * self.mag_min) - 10 ** (-self.b * self.mag_max))
            + 10 ** (-self.b * self.mag_min)
        )
