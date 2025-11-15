from typing import List, Optional

import numpy as np
import torch

from .dot_dict import DotDict
from .sequence import Sequence


class Batch(DotDict):
    """Batch of padded variable-length sequences.

    Should always be created from a list with Batch.from_list.

    In addition to basic information (such as arrival times) a Batch object contains
    other information that can be useful when computing the NLL of a TPP model, such as
    inter-event times, indices of first and last events in each sequence, and a
    mask indicating which entries correspond to actual events (and not to padding).

    See batch.keys() for the list of all available attributes.

    Attributes:
        inter_times: Inter-event times padded with zeros, shape [batch_size, seq_len]
        arrival_times: Padded arrival times, shape [batch_size, seq_len]
        t_start: Start of the observed interval for each sequence, shape [batch_size]
        t_end: End of the observed interval for each sequence, shape [batch_size]
        t_nll_start: Time from which the NLL is computed for each sequence.
            Defaults to t_start, shape [batch_size]
        mask: Binary mask indicating for which events the NLL must be computed,
            shape [batch_size, seq_len]
        start_idx: Index of the first event in each sequence, for which NLL must
            be computed, shape [batch_size]
        end_idx: Index of the last inter-event time in each sequence
            (survival time from last event t_N to t_end), shape [batch_size]
        **kwargs: Additional attributes associated with each event (e.g., magnitude,
            location), each with shape [batch_size, seq_len, ...].

    """

    default_batch_attrs = {
        "arrival_times",
        "inter_times",
        "t_start",
        "t_end",
        "t_nll_start",
        "start_idx",
        "end_idx",
        "mask",
    }

    @staticmethod
    def from_list(sequences: List[Sequence]) -> "Batch":
        """Construct a batch from a list of variable-length sequences."""
        batch_size = len(sequences)
        dtype = sequences[0].arrival_times.dtype
        device = sequences[0].arrival_times.device
        padded_seq_len = max(len(seq.inter_times) for seq in sequences)

        inter_times = pad_sequence(
            [seq.inter_times for seq in sequences],
            padding_value=0,
            max_len=padded_seq_len,
        )

        t_start = torch.zeros(batch_size, dtype=dtype, device=device)
        t_end = torch.zeros(batch_size, dtype=dtype, device=device)
        t_nll_start = torch.zeros(batch_size, dtype=dtype, device=device)
        end_idx = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i, seq in enumerate(sequences):
            t_start[i] = seq.t_start
            t_end[i] = seq.t_end
            t_nll_start[i] = seq.t_nll_start
            end_idx[i] = len(seq.arrival_times)

        # Get index of the first event that happened after t_nll_start
        arrival_times = torch.cumsum(inter_times, dim=-1) + t_start[:, None]
        start_idx = get_start_idx(arrival_times, t_nll_start)
        mask = get_mask(inter_times, start_idx, end_idx)

        # Handle other attributes (e.g., marks, locations) 
        other_attr_names = [
            k
            for k in sequences[0].keys()
            if k not in sequences[0].default_sequence_attrs and 'bounds' not in k
        ]
        other_attr = {}
        for name in other_attr_names:
            values = [seq[name] for seq in sequences]
            # Tensors are padded into shape (batch_size, padded_seq_len, ...)
            other_attr[name] = pad_sequence(
                values, padding_value=0, max_len=padded_seq_len
            )
            
        # Handle bounds (e.g. mark_bounds, mark_nll_bounds)
        other_bounds = {}
        other_attr_names = [
            k
            for k in sequences[0].keys()
            if 'bounds' in k
        ]
        for k in other_attr_names:
            other_bounds[k] = torch.zeros([batch_size,2], dtype=dtype, device=device)
            for i, seq in enumerate(sequences):
                other_bounds[k][i,:] = seq[k]
        

        return Batch(
            inter_times=inter_times,
            arrival_times=arrival_times,
            t_start=t_start,
            t_end=t_end,
            t_nll_start=t_nll_start,
            mask=mask,
            start_idx=start_idx,
            end_idx=end_idx,
            **other_attr,
            **other_bounds,
        )

    @property
    def batch_size(self):
        return self.arrival_times.shape[0]

    def __len__(self):
        return self.batch_size

    @property
    def seq_len(self):
        return self.arrival_times.shape[1]

    def get_sequence(self, idx: int) -> Sequence:
        length = int(self.end_idx[idx])
        inter_times = self.inter_times[idx, : length + 1].clone()
        t_start = float(self.t_start[idx])
        t_nll_start = float(self.t_nll_start[idx])
        other_attr = {}
        for k in self.keys():
            if k not in self.default_batch_attrs:
                if 'bounds' in k: 
                    other_attr[k] = self[k][idx]
                else:
                    other_attr[k] = self[k][idx, :length].clone()
        return Sequence(
            inter_times=inter_times,
            t_start=t_start,
            t_nll_start=t_nll_start,
            **other_attr,
        )

    def to_list(self) -> List[Sequence]:
        """Convert a batch into a list of variable-length sequences."""
        return [self.get_sequence(idx) for idx in range(self.batch_size)]


def get_start_idx(
    arrival_times: torch.Tensor, t_nll_start: torch.Tensor
) -> torch.Tensor:
    """Get index of the first event that happened after t_nll_start."""
    x = torch.masked_fill(arrival_times, arrival_times <= t_nll_start[:, None], np.inf)
    return x.argmin(-1)


def get_mask(
    inter_times: torch.Tensor,
    start_idx: torch.Tensor,
    end_idx: torch.Tensor,
) -> torch.Tensor:
    """Get a binary mask indicating which arrival times NLL must be computed."""
    arange = torch.arange(inter_times.shape[1], device=inter_times.device)[None, :]
    mask = (start_idx[:, None] <= arange) & (arange < end_idx[:, None])
    return mask.float()


def pad_sequence(
    sequences: List[torch.Tensor],
    padding_value: float = 0,
    max_len: Optional[int] = None,
) -> torch.Tensor:
    """Pad a list of variable length Tensors with `padding_value`."""
    dtype = sequences[0].dtype
    device = sequences[0].device
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len) + trailing_dims

    out_tensor = torch.empty(*out_dims, dtype=dtype, device=device).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :length, ...] = tensor

    return out_tensor
