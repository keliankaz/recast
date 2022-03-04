import math
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from scipy.stats import poisson
from tqdm.auto import trange

import eq
from eq.data.batch import get_mask, pad_sequence

from .tpp_model import TPPModel


def branching_ratio(k=0.001, b=1, alpha=1, M_min=0, M_max=10):
    """Compute branching ratio of the ETAS model (Sornette & Werner)."""
    if b == alpha:
        branching_ratio = (
            k * b * np.log(10) * (M_max - M_min) / (1 - 10 ** (-b * (M_max - M_min)))
        )
    else:
        branching_ratio = k * b / (b - alpha)
        branching_ratio *= 1 - 10 ** (
            -(b - alpha) * (M_max - M_min) / (1 - 10 ** (-b * (M_max - M_min)))
        )
    if branching_ratio > 1:
        print("Branching ratio: ", branching_ratio)
    return branching_ratio


def gen_mag(shape=1, b=1, M_min=0, M_max=10):
    """Draw sample from the Gutenberg-Richter distribution."""
    u = np.random.random(shape)
    mag = (
        -1
        / b
        * np.log10(-u * (10 ** (-b * M_min) - 10 ** (-b * M_max)) + 10 ** (-b * M_min))
    )
    return mag


def productivity(m, k, alpha=1, Mc=3):
    """Compute productivity of the earthquake with given magnitude."""
    return k * 10 ** (alpha * (m - Mc))


def omori_int(T1, T2, c, p):
    """Integral of Omori's law from T1 to T2.

    Used for the finite catalog correction in the productivity estimate (Brodsky 2011).
    """
    if p == 1:
        return np.log(T2 + c) - np.log(T1 + c)
    else:
        return ((T2 + c) ** (1 - p) - (T1 + c) ** (1 - p)) / (1 - p)


def omori_inv(T1, T2, c, p, size=1, t_max=1e10):
    """Draw sample from Omori's law using inverse transform."""
    u = np.random.random(size=size)
    F = lambda tau: omori_int(0, tau, c, p) / omori_int(0, t_max, c, p)
    u_prime = u * (F(T2) - F(T1)) + F(T1)
    return (
        (u_prime * omori_int(0, t_max, c, p) + c ** (1 - p) / (1 - p)) * (1 - p)
    ) ** (1 / (1 - p)) - c


class ETAS(TPPModel):
    """Epidemic-type aftershock sequence model (Ogata, 1988).

    Args:
        omori_p_init: Initial value of the p paramater of Omori's law.
        omori_c_init: Initial value of the c paramater of Omori's law.
        base_rate_init: Initial value of the background (immigrant) intensity.
        productivity_k_init: Initial value of the productivty parameter k.
        productivity_alpha_init: Initial value of the productivty parameter alpha.
        richter_b: Fixed b value of the Gutenberg-Richter distribution for magnitudes.
        mag_completeness: Magnitude of completeness.
        report_params: Whether to report the model parameters in the PyTorch Lightning
            progress bar during training.
        learning_rate: Learning rate use for optimization.
    """

    def __init__(
        self,
        omori_p_init: float = 1.08,
        omori_c_init: float = 0.1,
        base_rate_init: float = 0.02,
        productivity_k_init: float = 0.0073,
        productivity_alpha_init: float = 1.0,
        richter_b: float = 1.0,
        mag_completeness: float = 2.0,
        report_params: bool = True,
        learning_rate: float = 5e-2,
    ):
        super().__init__()
        self.log_p = nn.Parameter(torch.tensor(math.log(omori_p_init)))
        self.log_c = nn.Parameter(torch.tensor(math.log(omori_c_init)))
        self.log_mu = nn.Parameter(torch.tensor(math.log(base_rate_init)))
        self.log_k = nn.Parameter(torch.tensor(math.log(productivity_k_init)))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(productivity_alpha_init)))
        self.register_buffer("M_c", torch.tensor(mag_completeness))
        self.register_buffer("b", torch.tensor(richter_b))
        self.report_params = report_params
        self.learning_rate = learning_rate

    @property
    def p(self):
        return torch.exp(self.log_p)

    @property
    def c(self):
        return torch.exp(self.log_c)

    @property
    def mu(self):
        return torch.exp(self.log_mu)

    @property
    def k(self):
        return torch.exp(self.log_k)

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def nll_loss(self, batch: eq.data.Batch) -> torch.Tensor:
        """
        Compute negative log-likelihood (NLL) for a batch of event sequences.

        Args:
            batch: Batch of padded event sequences.

        Returns:
            nll: NLL of each sequence, shape (batch_size,)
        """
        t = batch.arrival_times
        # t_select - arrival times of events for which intensity must be computed, shape (B, S)
        # (where S = L if t_start == t_nll_start, and S <= L otherwise)
        t_select, intensity_mask = masked_select_per_row(t, batch.mask)
        # delta_t[0, i, j] = t_i - t_j
        delta_t = t_select.unsqueeze(-1) - t.unsqueeze(-2)  # (B, S, L)
        # prev_mask[0, i, j] = float(t_i < t_j)
        prev_mask = (delta_t > 0).float()  # (B, S, L)

        # Logarithm of the intensity
        # omori[0, i, j] = contribution of event t_j on intensity at time t_i
        omori = (delta_t * prev_mask + self.c).pow(-self.p)  # (B, S, L)
        # productivity[0, j] = expected number of aftershocks after event t_j
        productivity = self.k * 10 ** (self.alpha * (batch.mag - self.M_c))  # (B, L)
        log_intensity = (
            torch.log(
                (omori * productivity.unsqueeze(-2) * prev_mask).sum(-1) + self.mu
            )
            * intensity_mask
        ).sum(-1)

        # Integrated intensity
        one_minus_p = 1 - self.p
        t_end = batch.t_end.unsqueeze(-1)  # (B, 1)
        t_nll_start = batch.t_nll_start.unsqueeze(-1)  # (B, 1)
        # omori_int[0, j] = integral of the omori law from max(t_j, t_nll_start) to t_end
        omori_int = (
            (t_end - t + self.c).pow(one_minus_p)
            - ((t_nll_start - t).clamp_min(0.0) + self.c).pow(one_minus_p)
        ) / one_minus_p  # (B, L)
        survival_mask = get_mask(
            batch.inter_times,
            start_idx=torch.zeros_like(batch.start_idx),
            end_idx=batch.end_idx,
        )
        integral = (omori_int * productivity * survival_mask).sum(-1)
        integral += (batch.t_end - batch.t_nll_start) * self.mu
        return (-log_intensity + integral) / (batch.t_end - batch.t_nll_start)  # (B,)

    def training_step(self, batch, batch_idx):
        loss = self.nll_loss(batch).mean()
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
        )
        if self.report_params:
            for param_name in ["p", "c", "mu", "k", "alpha"]:
                self.log(
                    f"params/{param_name}",
                    getattr(self, param_name).item(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=batch.batch_size,
                )
        return loss

    def sample_thinning(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[eq.data.Sequence] = None,
        random_state: int = 123,
        max_length: int = 50_000,
        n_jobs: int = -1,
        return_sequences: bool = False,
        verbose: bool = False,
    ) -> Union[eq.data.Batch, List[eq.data.Sequence]]:
        """Generate a sample from the model (conditional or unconditional).

        Uses the thinning algorithm, which can be much slower than the default branching
        process sampler `sample`, especially for long sequences.

        Args:
            batch_size: number of sequences to generate.
            duration: length of the interval on which to simulate the tpp.
            t_start: start of the interval on which to simulate the tpp.
            past_seq: if provided, events are sampled conditioned on the past sequence.
            random_state: random seed, specified for reproducibility.
            max_length: if not none, discards samples with more than this many events.
                prevents runaway explosive sequences.
            n_jobs: Number of jobs that run sampling in parallel. -1 uses all cores.
            return_sequences: if true, returns samples as list[eq.data.sequence].
                if false, returns samples as eq.data.batch.
        """
        p, c, mu, k, alpha = [
            param.cpu().detach().numpy()
            for param in [self.p, self.c, self.mu, self.k, self.alpha]
        ]

        def get_intensity(t: float, t_past: np.ndarray, mag_past: np.ndarray):
            """Compute the intesity at time t given past events and magnitudes."""
            omori = (t - t_past + c) ** (-p)
            productivity = k * 10 ** (alpha * (mag_past - float(self.M_c)))
            return (omori * productivity).sum() + mu

        def bernoulli(success_proba: float):
            if success_proba < 0 or success_proba > 1:
                raise ValueError("Success probability must be in [0, 1] range")
            return np.random.uniform() < success_proba

        # Need to pass t_start as an argument due to weird Python interpreter behavior
        def sample_single_seq(t_start, seed):
            np.random.seed(seed)
            if past_seq is not None:
                # Recompute the arrival times in float64 precision
                past_tau = past_seq.inter_times.cpu().numpy().copy()
                arrival_times = np.cumsum(past_tau[:-1]) + past_seq.t_start
                magnitudes = past_seq.mag.cpu().numpy().copy()
                t_start = float(past_seq.t_end)
            else:
                arrival_times = np.array([], dtype=np.float64)
                magnitudes = np.array([], dtype=np.float64)
                t_start = t_start
            t_current = t_start
            t_end = t_start + duration
            # upper bound on the intensity - used to generate candidate events
            upper_bound = get_intensity(t_current, arrival_times, magnitudes)
            tau_current = 0.0
            inter_times = []
            while True:
                tau = np.random.exponential(1.0 / upper_bound)
                tau_current += tau
                t_current = t_current + tau
                if t_current > t_end:
                    break

                lambda_current = get_intensity(t_current, arrival_times, magnitudes)
                p_accept = lambda_current / upper_bound
                if verbose:
                    print(
                        f"\nCandidate event at {t_current:.3f}, acceptance prob = {p_accept:.2f}",
                        end="",
                    )
                if bernoulli(p_accept):
                    arrival_times = np.append(arrival_times, t_current)
                    magnitudes = np.append(
                        magnitudes,
                        gen_mag(b=float(self.b), M_min=float(self.M_c)),
                    )
                    inter_times.append(tau_current)
                    tau_current = 0.0
                    if verbose:
                        print(f" -> accepted (mag = {magnitudes[-1]:.2f})", end="")
                # Update the upper bound for the next event
                upper_bound = get_intensity(t_current, arrival_times, magnitudes)
                if len(inter_times) > max_length:
                    print(
                        "Stopping generation since max_length exceeded (likely explosive process)."
                    )
                    return None

            # Use max to avoid numerical errors
            inter_times = np.append(inter_times, max(duration - np.sum(inter_times), 0))
            valid_idx = (arrival_times > t_start) & (arrival_times <= t_end)
            return dict(
                inter_times=inter_times,
                t_start=t_start,
                mag=magnitudes[valid_idx],
            )

        sequences = []
        # Keep generating sequences in groups of size (batch_size - len(sequences))
        # until batch_size is reached. Some sequences might be too long because of
        # explosiveness - these are filtered out
        while len(sequences) < batch_size:
            num_seq_to_generate = batch_size - len(sequences)
            # Random seed is passed as an agument to the sampling function to ensure reproducibility
            new_sequences = Parallel(n_jobs=n_jobs)(
                delayed(sample_single_seq)(t_start, seed)
                for seed in trange(random_state, num_seq_to_generate + random_state)
            )
            filtered = [
                eq.data.Sequence(**seq) for seq in new_sequences if seq is not None
            ]
            sequences.extend(filtered)

        if return_sequences:
            return sequences
        else:
            return eq.data.Batch.from_list(sequences)

    def sample(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[eq.data.Sequence] = None,
        random_state: int = 123,
        max_length: Optional[int] = 50_000,
        t_max: float = 1e10,  # maximum duration of the aftershock sequence. Important for p close to 1.
        n_jobs: int = -1,
        return_sequences: bool = False,
    ) -> Union[eq.data.Batch, List[eq.data.Sequence]]:
        """Generate a sample from the model (conditional or unconditional).

        Args:
            batch_size: Number of sequences to generate.
            duration: Length of the interval on which to simulate the TPP.
            t_start: Start of the interval on which to simulate the TPP.
            past_seq: If provided, events are sampled conditioned on the past sequence.
            random_state: Random seed, specified for reproducibility.
            max_length: If not None, discards samples with more than this many events.
                Prevents runaway explosive sequences.
            t_max: Maximum time since parent at which an aftershock can be produced.
            n_jobs: Number of jobs that run sampling in parallel. -1 uses all cores.
            return_sequences: If True, returns samples as List[eq.data.Sequence].
                If False, returns samples as eq.data.Batch.

        Returns:
            batch: Sequences generated from the model.
        """
        p, c, mu, k, alpha, b, M_c = [
            param.cpu().detach().numpy()
            for param in [self.p, self.c, self.mu, self.k, self.alpha, self.b, self.M_c]
        ]
        if past_seq is not None:
            t_start = float(past_seq.t_end)
        else:
            t_start = t_start

        # Determine the branching ratio (and assert that it is smaller than one)
        branch = branching_ratio(k=k, b=b, alpha=alpha, M_min=M_c, M_max=10)
        if branch > 1:
            raise ValueError(
                f"The process is explosive: branching ratio {branch:.2f} is > 1."
            )

        def sample_single_seq(seed):
            np.random.seed(seed)
            if past_seq is not None:
                # Recompute the arrival times in float64 precision
                past_tau = past_seq.inter_times.cpu().numpy().copy()
                arrival_times = np.cumsum(past_tau[:-1]) + past_seq.t_start
                magnitudes = past_seq.mag.cpu().numpy().copy()
                parent_catalog = np.column_stack((arrival_times, magnitudes))
            else:
                arrival_times = np.array([], dtype=np.float64)
                magnitudes = np.array([], dtype=np.float64)
                parent_catalog = []

            t_end = t_start + duration

            # Background events are sampled from a poisson distribution with mean mu*T
            Nback = poisson.rvs(mu * (duration))  # number of background events

            # background events occur randomly in the time domain
            background_events = [np.random.uniform(t_start, t_end, Nback).T]
            background_events.append(gen_mag(shape=Nback, b=b, M_min=M_c))

            # Now iteratively add generations of aftershocks
            # The background and pre-existing catalog define the first parent catalog
            background_catalog = np.column_stack(background_events)
            parent_catalog = (
                np.vstack((parent_catalog, background_catalog))
                if len(parent_catalog) > 0
                else background_catalog
            )
            offspring_catalog = []
            whole_catalog = []
            generation = 0

            num_generated = Nback
            while True:
                whole_catalog.append(parent_catalog)

                # Determine the number of offspring each parent will have:
                k_prime = k * omori_int(0, t_max, c, p)
                prod = productivity(parent_catalog[:, -1], k_prime, alpha, M_c)

                # Determine how many of these events will be within the forecast interval:

                # Explanation:
                # This deserves some explanation. Condering the kernel k10**(alpha(M-M_c))*(t+c)**-p
                # We can recast the above expression as: N * p(t) where N is the number of aftershocks and p(t)
                # is a PDF for the arrival time (which integrates to 1).
                # Exanding the above expression yields:
                # N                   *           p(t)
                # k'*10**(alpha(M-Mc)) * (t+c)**-p / int((t+c)**-p)
                # where k' = k*int((t+c)**-p)
                TAU1 = t_start - parent_catalog[:, 0]
                TAU1[TAU1 < 0] = 0
                TAU2 = t_end - parent_catalog[:, 0]
                prod_in_interval = (
                    prod * omori_int(TAU1, TAU2, c, p) / omori_int(0, t_max, c, p)
                )

                # to streamline things we only consider the events that do have aftershocks
                N_aftershock = np.random.poisson(prod_in_interval)
                I = N_aftershock != 0
                short_parent_catalog = parent_catalog[I, :]
                short_N_aftershock = N_aftershock[I]

                for ieq, iNaft, itau1, itau2 in zip(
                    short_parent_catalog, short_N_aftershock, TAU1[I], TAU2[I]
                ):

                    # The cdf follows of the time distributions follows the integral of
                    # omori's law normalized by the integral out to infinity. Provided the
                    # temporal decay (p-value) is greater than 1 (p>1), the integral converges
                    # the inverse cdf can be used to generate random times:

                    t_parent = ieq[0]  # parent event time

                    dti = omori_inv(itau1, itau2, c, p, size=iNaft, t_max=t_max)
                    t_aftershock = t_parent + dti  # new arrival time

                    aftershock_catalog = []
                    aftershock_catalog.append(t_aftershock)

                    # ...and magnitudes
                    m_aftershock = gen_mag(iNaft, b=b, M_min=M_c)
                    aftershock_catalog.append(m_aftershock)

                    aftershock_catalog = np.column_stack(aftershock_catalog)

                    # Tack on the aftershock sequence to the catalog of offsprings
                    offspring_catalog.append(aftershock_catalog)

                if not offspring_catalog:
                    break

                # make offspring catalog
                offspring_catalog = np.vstack(offspring_catalog)
                generation += 1
                parent_catalog = offspring_catalog
                offspring_catalog = []
                # Stop generation if the event sequence is too long
                num_generated += len(parent_catalog)
                if max_length is not None and num_generated > max_length:
                    print(f"Exceeded {max_length} events, discarding sequence")
                    return None

            whole_catalog = np.vstack(whole_catalog)
            whole_catalog = whole_catalog[whole_catalog[:, 0].argsort()]

            arrival_times = whole_catalog[:, 0]
            magnitudes = whole_catalog[:, -1]

            valid_idx = (arrival_times > t_start) & (arrival_times <= t_end)
            fc_arrival_times = arrival_times[valid_idx]
            fc_magnitudes = magnitudes[valid_idx]

            inter_times = np.diff(fc_arrival_times, prepend=t_start, append=t_end)
            return eq.data.Sequence(
                inter_times=inter_times,
                t_start=t_start,
                mag=fc_magnitudes,
            )

        sequences = []
        # Keep generating sequences in groups of size (batch_size - len(sequences)) until batch_size is reached
        # Some sequences might be too long because of explosiveness - these are filtered out
        np.random.seed(random_state)
        starting_seed = np.random.randint(0, 100000)
        while len(sequences) < batch_size:
            num_seq_to_generate = batch_size - len(sequences)
            new_sequences = Parallel(n_jobs=n_jobs)(
                delayed(sample_single_seq)(seed)
                for seed in trange(starting_seed, starting_seed + num_seq_to_generate)
            )
            # Filter out explosive sequences
            filtered = [seq for seq in new_sequences if seq is not None]
            sequences.extend(filtered)
            starting_seed += num_seq_to_generate

        if return_sequences:
            return sequences
        else:
            return eq.data.Batch.from_list(sequences)


def masked_select_per_row(matrix, mask):
    """Perform masked select on each row, and return the result as a padded tensor.

    Args:
        matrix: 2-d tensor from which values must be selected, shape [M, N]
        mask: Boolean matrix indicating what entries must be selected, shape [M, N]

    Returns:
        new_matrix: 2-d tensor, where each row contains the selected entries from the
            respective row of matrix + padding.
        new_mask: Float mask indicating what entries correspond to actual values
            (new_mask[i, j] = 1 => new_matrix[i, j] is not padding).

    Example:
        >>> matrix = torch.tensor([
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
            ])
        >>> mask = torch.tensor([
                [0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1],
            ])
        >>> selected, new_mask = masked_select_per_row(matrix, mask)
        >>> print(selected)
        tensor([[1, 2, 3],
                [8, 9, 0]])
        >>> print(new_mask)
        tensor([[1., 1., 1.],
                [1., 1., 0.]])
    """
    assert matrix.shape == mask.shape and matrix.ndim == 2
    selected_rows = []
    for matrix_row, mask_row in zip(matrix, mask.bool()):
        selected_rows.append(matrix_row.masked_select(mask_row))

    new_matrix = pad_sequence(selected_rows)
    new_mask = pad_sequence([torch.ones_like(s) for s in selected_rows])
    return new_matrix, new_mask.float()
