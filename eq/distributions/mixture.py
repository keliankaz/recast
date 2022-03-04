import torch
from torch.distributions import Categorical
from torch.distributions import MixtureSameFamily as TorchMixtureSameFamily

from .distribution import Distribution


class MixtureSameFamily(TorchMixtureSameFamily, Distribution):
    def __init__(
        self, mixture_distribution, component_distribution, validate_args=False
    ):
        super(MixtureSameFamily, self).__init__(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
            validate_args=False,
        )

    def log_hazard(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_prob(x) - self.log_survival(x)

    def log_survival(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad(x)
        log_sf_x = self.component_distribution.log_survival(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_sf_x + mix_logits, dim=-1)

    def sample_conditional(self, lower_bound, sample_shape=torch.Size()):
        with torch.no_grad():
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape

            # Since we know that the sample x > lower_bound, we have to adjust the
            # mixing probabilities as p(z_i = k) * Pr(x >= lower_bound | z_i = k)
            # mixture samples [n, B]
            conditional_mix_probs = (
                self.mixture_distribution.probs
                * self.component_distribution.log_survival(lower_bound).exp()
            )
            mix_sample = Categorical(probs=conditional_mix_probs).sample(sample_shape)
            mix_shape = mix_sample.shape

            # component samples [n, B, k, E]
            comp_samples = self.component_distribution.sample_conditional(
                lower_bound, sample_shape
            )

            # Gather along the k dimension
            mix_sample_r = mix_sample.reshape(
                mix_shape + torch.Size([1] * (len(es) + 1))
            )
            mix_sample_r = mix_sample_r.repeat(
                torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es
            )

            samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
            return samples.squeeze(gather_dim)
