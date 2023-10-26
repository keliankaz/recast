from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import eq
import eq.distributions as dist

from eq.models.tpp_model import TPPModel


class RecurrentTPP(TPPModel):
    """Neural TPP model with an recurrent encoder.

    Args:
        input_magnitude: Should magnitude be used as model input?
        predict_magnitude: Should the model predict the magnitude?
        num_extra_features: Number of extra features to use as input.
        context_size: Size of the RNN hidden state.
        num_components: Number of mixture components in the output distribution.
        rnn_type: Type of the RNN. Possible choices {'GRU', 'RNN'}
        dropout_proba: Dropout probability.
        tau_mean: Mean inter-event times in the dataset.
        mag_mean: Mean earthquake magnitude in the dataset.                             # FLAG
        richter_b: Fixed b value of the Gutenberg-Richter distribution for magnitudes.
        mag_completeness: Magnitude of completeness of the catalog.
        learning_rate: Learning rate used in optimization.
    """

    def __init__(
        self,
        input_magnitude: bool = True,
        predict_magnitude: bool = True,
        num_extra_features: Optional[int] = None,
        context_size: int = 32,
        num_components: int = 32,
        rnn_type: str = "GRU",
        dropout_proba: float = 0.5,
        tau_mean: float = 1.0,
        richter_b: float = 1.0,
        learning_rate: float = 5e-2,
    ):
        super().__init__()
        self.input_magnitude = input_magnitude
        self.predict_magnitude = predict_magnitude
        self.num_extra_features = num_extra_features
        self.context_size = context_size
        self.num_components = num_components
        self.register_buffer("tau_mean", torch.tensor(tau_mean, dtype=torch.float64))
        self.register_buffer("log_tau_mean", self.tau_mean.log())
        self.register_buffer("richter_b", torch.tensor(richter_b, dtype=torch.float64))
        self.learning_rate = learning_rate

        # Decoder for the time distribution
        self.num_time_params = 3 * self.num_components
        self.hypernet_time = nn.Linear(context_size, self.num_time_params)

        # RNN input features
        if self.input_magnitude:
            # Decoder for magnitude
            self.num_mag_params = 1  # (1 rate)
            self.hypernet_mag = nn.Linear(context_size, self.num_mag_params)

        if rnn_type not in ["RNN", "GRU"]:
            raise ValueError(
                f"rnn_type must be one of ['RNN', 'GRU'] " f"(got {rnn_type})"
            )
        self.num_rnn_inputs = (
            1 + int(self.input_magnitude) + (
                0  if self.num_extra_features is None
                else self.num_extra_features
            )
        )

        self.rnn = getattr(nn, rnn_type)(
            self.num_rnn_inputs,
            context_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_proba)

    def encode_time(self, inter_times):
        # inter_times has shape (...)
        # output has shape (..., 1)
        log_tau = torch.log(torch.clamp_min(inter_times, 1e-10)).unsqueeze(-1)
        return log_tau - self.log_tau_mean

    def encode_magnitude(self, mag, mag_completeness: Union[float,torch.tensor]):
        # mag has shape (...)
        # mag_completeness 
        # output has shape (..., 1)
        if type(mag) is float:
            out = mag.unsqueeze(-1) - mag_completeness
        else:
            out = (mag - mag_completeness.unsqueeze(1)).unsqueeze(-1)
        return out

    def encode_extra_features(self, extra_feat):
        # Place holder for any encoding of extra features that may be needed
        # e.g. normalization, log-transform, aggregation, etc.
        # extra_feat has shape (..., num_extra_features)
        # output has shape (..., num_extra_features)
        return extra_feat

    def get_context(self, batch):
        """Get context embedding for each event in the batch of padded sequences.

        Returns:
            context: Context vectors, shape (batch_size, seq_len, context_size)
        """
        feat_list = [self.encode_time(batch.inter_times)]
        if self.input_magnitude:
            feat_list.append(self.encode_magnitude(batch.mag, batch.mag_bounds[:,0]))
        if self.num_extra_features is not None:
            feat_list.append(self.encode_extra_features(batch.extra_feat))
        features = torch.cat(feat_list, dim=-1)

        rnn_output = self.rnn(features)[0][:, :-1, :]
        output = F.pad(rnn_output, (0, 0, 1, 0))  # (B, L, C)
        return self.dropout(output)  # (B, L, C)

    def get_inter_time_dist(self, context):
        """Get the distribution over the inter-event times given the context."""
        params = self.hypernet_time(context)
        # Very small params may lead to numerical problems, clamp to avoid this
        # params = clamp_preserve_gradients(params, -6.0, np.inf)
        scale, shape, weight_logits = torch.split(
            params,
            [self.num_components, self.num_components, self.num_components],
            dim=-1,
        )
        scale = F.softplus(scale.clamp_min(-5.0))
        shape = F.softplus(shape.clamp_min(-5.0))
        weight_logits = F.log_softmax(weight_logits, dim=-1)
        component_dist = dist.Weibull(scale=scale, shape=shape)
        mixture_dist = Categorical(logits=weight_logits)
        return dist.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            component_distribution=component_dist,
        )

    def get_magnitude_dist(self, context, mag_completeness):
        """Returns the GutenberRichter distribution for each context.  """
        log_rate = self.hypernet_mag(context).squeeze(-1)  # (B, L)
        b = self.richter_b * torch.ones_like(log_rate)
        mag_min = mag_completeness.unsqueeze(1) * torch.ones_like(b[0,:])                     # FLAG
        return dist.GutenbergRichter(b=b, mag_min=mag_min)

    def nll_loss(self, batch: eq.data.Batch) -> torch.Tensor:
        """
        Compute negative log-likelihood (NLL) for a batch of event sequences.

        Args:
            batch: Batch of padded event sequences.

        Returns:
            nll: NLL of each sequence, shape (batch_size,)
        """
        context = self.get_context(batch)  # (B, L, C)
        # Inter-event times
        inter_time_dist = self.get_inter_time_dist(context)
        log_pdf = inter_time_dist.log_prob(batch.inter_times.clamp_min(1e-10))  # (B, L)
        log_like = (log_pdf * batch.mask).sum(-1)

        # Survival time from last event until t_end
        arange = torch.arange(batch.batch_size)
        last_surv_context = context[arange, batch.end_idx, :]
        last_surv_dist = self.get_inter_time_dist(last_surv_context)
        last_log_surv = last_surv_dist.log_survival(
            batch.inter_times[arange, batch.end_idx]
        )
        log_like = log_like + last_log_surv.squeeze(-1)  # (B,)

        # Remove survival time from t_prev to t_nll_start
        if torch.any(batch.t_nll_start != batch.t_start):
            prev_surv_context = context[arange, batch.start_idx, :]
            prev_surv_dist = self.get_inter_time_dist(prev_surv_context)
            prev_surv_time = batch.inter_times[arange, batch.start_idx] - (
                batch.arrival_times[arange, batch.start_idx] - batch.t_nll_start
            )
            prev_log_surv = prev_surv_dist.log_survival(prev_surv_time)
            log_like = log_like - prev_log_surv

        return -log_like / (batch.t_end - batch.t_nll_start)  # (B,)

    def sample(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[eq.data.Sequence] = None,
        return_sequences: bool = False,
        mag_completeness: Optional[float] = None 
    ) -> Union[eq.data.Batch, List[eq.data.Sequence]]:
        """Simulate a batch of event sequences from the model.

        Args:
            batch_size: Number of sequences to generate.
            duration: Length of the interval on which to simulate the TPP.
            t_start: Start of the interval on which to simulate the TPP.
            past_seq: If provided, events are sampled conditioned on the past sequence.
            return_sequences: If True, returns samples as List[eq.data.Sequence].
                If False, returns samples as eq.data.Batch.

        Returns:
            batch: Sequences generated from the model.

        """
        raise NotImplementedError() # temporary hide this function
        if self.input_magnitude != self.predict_magnitude:
            raise ValueError(
                "Sampling is impossible if input_magnitude != predict_magnitude"
            )
        if self.num_extra_features is not None:
            raise ValueError("Sampling is not currently supported for extra features")

        if past_seq is not None:
            t_start = past_seq.t_end
            past_batch = eq.data.Batch.from_list([past_seq])
            if mag_completeness is not None:
                assert mag_completeness == past_seq.mag_bounds[0] # TODO: use the mag_completeness to truncate the output. 
            else:
                mag_completeness = past_seq.mag_bounds[0]
            current_state = self.get_context(past_batch)[:, [-1], :]  # (1, 1, C)
            current_state = current_state.expand(batch_size, -1, -1)
            time_remaining = past_seq.t_end - past_seq.arrival_times[-1]
        else:
            current_state = torch.zeros(batch_size, 1, self.context_size)
            time_remaining = None
        t_end = t_start + duration

        inter_times = torch.empty(batch_size, 0, device=self.device)
        if self.predict_magnitude:
            magnitudes = torch.empty(batch_size, 0, device=self.device)
        else:
            magnitudes = None

        generated = False
        while not generated:
            inter_time_dist = self.get_inter_time_dist(current_state)
            if time_remaining is None:
                next_inter_times = inter_time_dist.sample()  # (B, 1)
            else:
                next_inter_times = inter_time_dist.sample_conditional(
                    lower_bound=time_remaining
                )  # (B, 1)
                next_inter_times -= time_remaining
                time_remaining = None
            next_inter_times.clamp_max_(
                t_end - t_start
            )  # BUG: (?) creates samples of len 1 instead of zero
            inter_times = torch.cat([inter_times, next_inter_times], dim=1)  # (B, L)
            # Prepare RNN input
            rnn_input_list = [self.encode_time(next_inter_times)]

            if self.predict_magnitude:
                mag_dist = self.get_magnitude_dist(current_state)
                next_mag = mag_dist.sample()  # (B, 1)                                  # FLAG
                magnitudes = torch.cat([magnitudes, next_mag], dim=1)  # (B, L)
                rnn_input_list.append(self.encode_magnitude(next_mag, mag_completeness))                  # FLAG

            with torch.no_grad():
                reached = inter_times.sum(-1).min()
                generated = reached >= t_end - t_start
                rnn_input = torch.cat(rnn_input_list, dim=-1)
            current_state = self.rnn(
                rnn_input, current_state.transpose(0, 1).contiguous()
            )[0]
            current_state = self.dropout(current_state)  # (B, 1, C)

        duration = t_end - t_start
        unclipped_arrival_times = inter_times.cumsum(-1)  # (B, L)
        padding_mask = (
            unclipped_arrival_times >= duration
        )  # BUG: ATTEMPTED FIX (> to >=) - clipped values should be masked out
        inter_times = torch.masked_fill(inter_times, padding_mask, 0.0)
        end_idx = (1 - padding_mask.long()).sum(-1)
        last_surv_time = duration - inter_times.sum(-1)
        inter_times[torch.arange(batch_size), end_idx] = last_surv_time
        batch = eq.data.Batch(
            inter_times=inter_times,
            arrival_times=inter_times.cumsum(-1),
            t_start=torch.full([batch_size], t_start, device=self.device).float(),
            t_end=torch.full([batch_size], t_end, device=self.device).float(),
            t_nll_start=torch.full([batch_size], t_start, device=self.device).float(),
            mask=padding_mask.float(),
            start_idx=torch.zeros(batch_size, device=self.device).long(),
            end_idx=end_idx,
            mag=magnitudes,
        )
        if return_sequences:
            return batch.to_list()
        else:
            return batch

    def evaluate_intensity(
        self,
        sequence: eq.data.Sequence,
        num_grid_points: int = 100,
        eps: float = 1e-4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = eq.data.Batch.from_list([sequence])
        context = self.get_context(batch).squeeze(0)  # (L, C)
        inter_time_dist = self.get_inter_time_dist(context)

        # Evaluate each hazard function at times x = [eps, ..., tau_i]
        x = batch.inter_times * torch.linspace(eps, 1, num_grid_points)[:, None]
        intensity = inter_time_dist.log_hazard(x).T.reshape(-1).exp()

        # Shift the inter-event times x to get the global times
        offsets = torch.cat([torch.tensor([0.0]), sequence.arrival_times])
        grid = (x + offsets).T.reshape(-1)
        return grid, intensity

    def evaluate_compensator(
        self, sequence: eq.data.Sequence, num_grid_points: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = eq.data.Batch.from_list([sequence])
        context = self.get_context(batch).squeeze(0)  # (L, C)
        inter_time_dist = self.get_inter_time_dist(context)

        # Evaluate each log survival function at times x = [eps, ..., tau_i]
        x = batch.inter_times * torch.linspace(1e-4, 1, num_grid_points)[:, None]
        log_surv = inter_time_dist.log_survival(x)
        # Compute the cumulative sum of log survival functions to get the compensator
        surv_offsets = torch.cat(
            [torch.tensor([0.0]), log_surv[-1].cumsum(dim=-1)[:-1]]
        )
        compensator = -(log_surv + surv_offsets).T.reshape(-1)

        # Shift the inter-event times x to get the global times
        offsets = torch.cat([torch.tensor([0.0]), sequence.arrival_times])
        grid = (x + offsets).T.reshape(-1)
        return grid, compensator
