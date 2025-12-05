from eq.data import Sequence, Batch
from copy import deepcopy
import random
import torch
from typing import List

AUGMENTATION_REGISTRY = {}


def register(name: str):
    def decorator(fn):
        AUGMENTATION_REGISTRY[name] = fn
        return fn

    return decorator


@register("jitter")
def jitter(
    seq: Sequence, std: float = 0.1, key: str = "mag", enforce_bounds: bool = True
) -> Sequence:
    """Jitter the values of a mark.

    Note that if enforce_bounds is True, the jitter is clamped to the bounds of the mark following the `ContinousMarks` class.
    """

    seq = deepcopy(seq)
    seq[key] += torch.abs(torch.normal(0, std, seq[key].shape))
    if enforce_bounds:
        seq[key] = torch.clamp(
            seq[key], min=getattr(seq, key + "_bounds")[0], max=getattr(seq, key + "_bounds")[1]
        )

    return seq


@register("jitter_time")
def jitter_time(seq: Sequence, std: float = 1e-5) -> Sequence:
    """Jitter the time of a sequence. Default std is 1e-5 days, which is approximately 1 second."""
    seq = deepcopy(seq)
    arrival_times = seq.arrival_times64 + torch.abs(
        torch.normal(0, std, seq.arrival_times64.shape)
    )
    arrival_times = torch.clamp(arrival_times, min=seq.t_start, max=seq.t_end)
    arrival_times, sorted_idx = arrival_times.sort()
    inter_times = seq.compute_inter_times(arrival_times, seq.t_start, seq.t_end)

    remaining_attr = {}
    for key in seq.keys():
        if "_bounds" not in key and key not in seq.default_sequence_attrs:
            remaining_attr[key] = seq[key][sorted_idx]

    bounds = {}
    for key, value in seq.items():
        if "_bounds" in key:
            bounds[key] = value

    return Sequence(
        inter_times=inter_times,
        t_start=seq.t_start,
        t_end=seq.t_end,
        t_nll_start=seq.t_nll_start,
        **remaining_attr,
        **bounds,
    )


@register("superimpose")
def superimpose(seq: Sequence, seq_bank: List[Sequence]) -> Sequence:
    """Superimpose a random sequence from `seq_back` onto `seq`.

    The resulting sequence preserves the bounds of `seq`.
    
    Note that if the location of events will be considered the superimpose may give rise to issues.
    """

    seq = deepcopy(seq)
    dtype = seq.inter_times.dtype

    other = random.choice(seq_bank)

    assert (
        other.t_end - other.t_start >= seq.t_end - seq.t_nll_start
    ), "The duration of the other sequence needs to longer than interval must be longer than the nll interval of seq"

    # randomly choose a start time for other that ensures that the whole interval between seq.t_nll_start and seq.t_end is covered by other.
    seq_shift = -seq.t_start

    min_shift = max(
        -other.t_start, (other.t_end - other.t_start) - (seq.t_end - seq.t_start)
    )
    max_shift = (seq.t_nll_start - seq.t_start) - other.t_start

    random_shift = torch.rand(1) * (max_shift - min_shift) + min_shift

    other_arrival_times = (
        other.inter_times.cumsum(dim=-1, dtype=torch.float64)[:-1]
        + other.t_start
        + random_shift
    )
    seq_arrival_times = (
        seq.inter_times.cumsum(dim=-1, dtype=torch.float64)[:-1]
        + seq.t_start
        + seq_shift
    )

    combined_arrival_times, sorted_idx = torch.cat(
        [seq_arrival_times, other_arrival_times]
    ).sort()

    inter_times = torch.diff(
        combined_arrival_times,
        prepend=torch.tensor([0.0], dtype=torch.float64),
        append=torch.tensor(
            [max(seq.t_end + seq_shift, other.t_end + random_shift)],
            dtype=torch.float64,
        ),
    ).to(dtype)

    remaining_attr = {}
    for key in seq.keys():
        if "_bounds" not in key and key not in seq.default_sequence_attrs:
            remaining_attr[key] = torch.cat([seq[key], other[key]])[
                sorted_idx
            ]  # Should I account for other potential dimensions here?

    bounds = {}
    for key, value in seq.items():
        if "_bounds" in key:
            bounds[key] = value

    combined_sequence = Sequence(
        inter_times=inter_times,
        t_start=0.0,
        t_end=max(seq.t_end + seq_shift, other.t_end + random_shift),
        t_nll_start=seq.t_nll_start + seq_shift,
        **remaining_attr,
        **bounds,
    )

    return combined_sequence.get_subsequence(
        seq.t_start + seq_shift, seq.t_end + seq_shift
    )


@register("sub_radius")
def sub_radius(seq: Sequence, fraction_range: list[float] = [0.5, 1.0], radius_km: float = 1000) -> Sequence:
    """Subset the sequence to a spatial radius of `radius` kilometers."""
    
    assert "distance_km" in seq.keys(), "distance_km is not a key in the sequence"
    new_radius_km = radius_km * random.uniform(fraction_range[0], fraction_range[1])
    indices = seq.distance_km <= new_radius_km
    
    return seq.subsequence_by_index(indices)



def build_augmentations(specs):
    """Build a list of augmentations from a list of specifications.

    Example:
    >>> specs = [("jitter", {"key": "mag", "std": 0.1}), ("jitter", {"key": "time", "std": 0.2, "enforce_bounds": False})]
    >>> build_augmentations(specs)
    """
    aug_list = []
    for name, kwargs in specs:
        aug_list.append(
            lambda x, fn=AUGMENTATION_REGISTRY[name], kw=kwargs: fn(x, **kw)
        )
    return aug_list


class AugmentationCollator:
    def __init__(self, specs):
        self.aug_list = build_augmentations(specs)

    def __call__(self, seq_list: List[Sequence]) -> Batch:
        """returns an augmented batch given a list of sequences"""

        for aug in self.aug_list:
            seq_list = [aug(seq) for seq in seq_list]

        return Batch.from_list(seq_list)
