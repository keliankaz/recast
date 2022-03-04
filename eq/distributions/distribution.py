import torch
from torch.distributions import Distribution as TorchDistribution


class Distribution(TorchDistribution):
    def log_hazard(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the logarithm of the hazard function.

        The hazard function h(x) is defined as h(x) = p(x) / S(x), where p(x) is the
        probability density function (PDF) and S(x) is the survival function (SF)
        defined as S(x) = \int_{0}^{x} p(u) du.

        Args:
            x: Input.

        Returns:
            log_h: log h(x), same shape as the input x.
        """
        raise NotImplementedError

    def log_survival(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the logarithm of the survival function.

        The survival function S(x) corresponds to Pr(X >= x) and can be computed as
        S(x) = \int_{0}^{x} p(u) du, where p(x) is the PDF.

        Args:
            x: Input.

        Returns:
            log_S: log S(x), same shape as the input x.
        """
        raise NotImplementedError

    def sample_conditional(
        self, lower_bound: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Sample from the distribution conditioned on the fact that x > lower_bound."""
        raise NotImplementedError
