from eq.data import Sequence, Batch
from copy import deepcopy
import random
import torch
from typing import List

#
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
            seq[key], min=seq[key + "_bounds"][0], max=seq[key + "_bounds"][1]
        )

    return seq


@register("superimpose")
def superimpose(seq: Sequence, seq_bank: List[Sequence]) -> Sequence:
    """Superimpose a random sequence from `seq_back` onto `seq`.

    The resulting sequence preserves the bounds of `seq`.
    """

    seq = deepcopy(seq)
    other = random.choice(seq_bank)
    other = other.get_subsequence(
        max(seq.t_start, other.t_start), 
        min(seq.t_end, other.t_end)
    )
    
    arrival_times, sorted_idx = torch.cat(
        [seq.arrival_times, other.arrival_times]
    ).sort()

    inter_times = torch.diff(
        arrival_times, 
        prepend=torch.tensor([seq.t_start], device=seq.arrival_times.device, dtype=seq.arrival_times.dtype), 
        append=torch.tensor([seq.t_end], device=seq.arrival_times.device, dtype=seq.arrival_times.dtype)
    )

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

    return Sequence(
        inter_times=inter_times,
        t_start=seq.t_start,
        t_end=seq.t_end,
        t_nll_start=seq.t_nll_start,
        **remaining_attr,
        **bounds,
    )

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
