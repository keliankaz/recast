import warnings
from typing import Optional, Union

import numpy as np
import torch

from eq.data.dot_dict import DotDict


class ContinuousMarks:
    """Mark associated with an event.

    TPP marks seem to have a recurring pattern. Namely, they involve a set of values
    that exist within a bounds. In many cases, it is useful to evaluate the negative log-likelihood (NLL) on a
    subset of the bounds.

    Note that ContinuousMarks does not actually check whether values are within bounds.

    Examples:
        - Magnitude: The bounds is [Mc,Mmax] but the NLL may be evaluated on the interval [Mmin, Mmax].
        - Location: The bounds delimited by a convex hull specified by set of knots. The NLL may be evaluated in a subregion.
        - Time: The bounds is [t_start, t_end] but the NLL may be evaluated on the interval [t_nll_start, t_end].

    Shared among all marks is that the boundss has dimensions R^d x R^n where d is the number of
    dimensions associated with the marks and n >= d+1 to bound the marks bounds.

    Args:
        values: Values associated with the mark.
        bounds: bounds associated with the mark.
        nll_bounds: bounds on which the NLL is evaluated.

    Example:
        >>> marks = ContinuousMarks([4.0, 2.1, 2.5, 2.9], bounds=[2.0, 5.0], nll_bounds=[2.5, 5.0])
        >>> marks.nll_bounds
        tensor([2.5000, 5.0000])

        # Simply input the mark into a sequence instead of the values alone
        >>> seq = Sequence(inter_times=[1.0, 2.0, 3.0], t_start=0.0, mag=marks)
        >>> seq.mag_nll_bounds
        tensor([2.5000, 5.0000])

    """

    def __init__(
        self,
        values: Union[torch.Tensor, np.ndarray, list],
        bounds: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
        nll_bounds: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
    ):
        self.values = values
        self.bounds = bounds

        if nll_bounds is None:
            self.nll_bounds = bounds
        else:
            self.nll_bounds = nll_bounds

        self._validate_args()

    def _validate_args(self):
        assert (
            np.atleast_2d(self.bounds).shape[0] == np.atleast_2d(self.values).shape[0]
        ), "bounds must share the same dimension as the values"
        assert (
            np.atleast_2d(self.bounds).shape[1]
            >= np.atleast_2d(self.values).shape[0] + 1
        ), "bounds must have at least one more dimension than the values"


class Sequence(DotDict):
    """Sequence of events (potentially with marks).

    Args:
        inter_times: Inter-event times, including the last survival time t_end - t_N.
            shape [num_events + 1]
        t_start: Start of the observed time interval.
        t_nll_start: The negative log-likelihood (NLL) will be evaluated on the interval
            (t_nll_start, t_end]. Used when evaluating predictive performance of a TPP
            model given past events. Defaults to t_start.
        **kwargs: Additional dynamical attributes associated with each event in the
            sequence (e.g., magnitude, location), each with shape [num_events, ...].

    Example:
        >>> t_start = 10.0
        >>> arrival_times = np.array([11.1, 11.5, 12.1, 14.2])
        >>> # If you don't know t_end, just set t_end = arrival_times[-1]
        >>> t_end = 20.0
        >>> mag = np.array([4.0, 2.1, 2.5, 2.9])
        >>> loc = np.array([
            [33.1, -115.0],
            [33.2, -115.3],
            [33.1, -116.2],
            [32.3, -115.2],
        ])
        >>> inter_times = np.diff(arrival_times, prepend=[t_start], append=[t_end])
        >>> seq = Sequence(inter_times, t_start=0.0, mag=mag, loc=loc)
    """

    default_sequence_attrs = {
        "arrival_times",
        "arrival_times64",  # used for computation of arrival times in double precision
        "inter_times",
        "t_start",
        "t_end",
        "t_nll_start",
    }

    def __init__(
        self,
        inter_times: Union[torch.Tensor, np.ndarray, list],
        t_start: float = 0.0,
        t_nll_start: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.inter_times = torch.flatten(
            torch.as_tensor(inter_times, dtype=torch.float32)
        )
        if self.inter_times.dtype not in [torch.float32, torch.float64]:
            raise ValueError(
                f"inter_times must be of type torch.float32 or torch.float64 (got {self.inter_times.dtype})"
            )

        self.t_start = float(t_start)
        self.t_end = float(self.inter_times.sum().item() + self.t_start)
        if t_nll_start is None:
            t_nll_start = t_start
        self.t_nll_start = float(t_nll_start)

        for key, value in kwargs.items():
            if isinstance(value, ContinuousMarks):
                self[key] = torch.as_tensor(value.values)
                self[key + "_bounds"] = torch.as_tensor(value.bounds)
                self[key + "_nll_bounds"] = torch.as_tensor(value.nll_bounds)
            else:
                self[key] = torch.as_tensor(value)

        self._validate_args()
        # Move all tensors to the same device as inter_times
        self.to(self.inter_times.device)

    @property
    def arrival_times64(self):
        # interevent_times is float32 on MPS
        inter = self.inter_times.to("cpu", dtype=torch.float64)
        return inter.cumsum(dim=-1)[:-1] + self.t_start

    @property
    def arrival_times(self):
        return self.arrival_times64.to(
            self.inter_times.device, dtype=self.inter_times.dtype
        )

    @property
    def num_events(self):
        return len(self.arrival_times)

    @property
    def num_nll_events(self):
        """Number of events in the interval where the NLL is computed."""
        return (self.arrival_times >= self.t_nll_start).sum().item()

    def __len__(self):
        return self.num_events

    @staticmethod
    def compute_inter_times(
        arrival_times: Union[np.ndarray, list, torch.Tensor],
        t_start: float,
        t_end: float,
    ) -> np.ndarray:

        return np.diff(arrival_times, prepend=[t_start], append=[t_end])

    def get_subsequence(self, start: float, end: float) -> "Sequence":
        """Select a subset of events in the interval [start, end]."""
        if start < self.t_start or end > self.t_end:
            raise ValueError(
                f"start must be >= {self.t_start} and end must be <= {self.t_end}"
            )
        mask = (self.arrival_times64 >= start) & (self.arrival_times64 <= end)

        new_arrival_times = self.arrival_times64[mask]
        if len(new_arrival_times) > 0:
            last_inter_time = torch.tensor(
                [end - new_arrival_times[-1]],
                device=self.inter_times.device,
                dtype=self.inter_times.dtype,
            )
            new_inter_times = torch.cat([self.inter_times[:-1][mask], last_inter_time])
            first_inter_time = new_arrival_times[0] - start
            new_inter_times[0] = first_inter_time
        else:
            new_inter_times = torch.tensor(
                [end - start],
                device=self.inter_times.device,
                dtype=self.inter_times.dtype,
            )

        # Deal with other sequence attributes
        other_attr = {}
        for key, value in self.items():
            if "_bounds" not in key and key not in self.default_sequence_attrs:
                other_attr[key] = value[mask].contiguous()

        bounds = {}
        for key, value in self.items():
            if "_bounds" in key:
                bounds[key] = value

        return Sequence(
            inter_times=new_inter_times,
            t_start=start,
            t_nll_start=max(self.t_nll_start, start),
            **other_attr,
            **bounds,
        )

    def subsequence_by_index(self, indices: Union[list[int], torch.Tensor, int]) -> "Sequence":
        """Select a subset of events by index or mask."""
        
        if isinstance(indices, int):
            indices = [indices]
        new_arrival_times = self.arrival_times64[indices]
        
        new_inter_times = self.compute_inter_times(new_arrival_times, self.t_start, self.t_end)
        
        other_attr = {}
        for key, value in self.items():
            if "_bounds" not in key and key not in self.default_sequence_attrs:
                other_attr[key] = value[indices].contiguous()

        bounds = {}
        for key, value in self.items():
            if "_bounds" in key:
                bounds[key] = value
    
        return Sequence(
            inter_times=new_inter_times,
            t_start=self.t_start,
            t_nll_start=self.t_nll_start,
            **other_attr,
            **bounds,
        )

    def state_dict(self) -> dict:
        # These attributes are computed from inter_times and t_start, no need to save them to disk
        inferred_attributes = ["arrival_times", "t_end"]
        return {k: v for (k, v) in self.items() if k not in inferred_attributes}

    def _validate_args(self):
        """Check if the event sequence is valid and the shapes are correct."""
        if not (0 <= self.t_start <= self.t_nll_start <= self.t_end):
            raise ValueError(
                "It should hold 0 <= t_start <= t_nll_start < t_end. "
                f"Received {self.t_start}, {self.t_nll_start}, {self.t_end}."
            )

        if torch.any(self.inter_times < 0):
            raise ValueError(
                "Negative inter-event times detected, this isn't a valid event sequence."
            )
        num_zero_inter_times = (self.inter_times[:-1] == 0).sum()
        if num_zero_inter_times > 0:
            warnings.warn(
                f"Found {num_zero_inter_times} zero inter-event times in the sequence. "
                f"This violates fundamental assumptions of TPP models and may lead to "
                f"incorrect log-likelihood values."
            )

        for key, value in self.items():
            if (
                key not in self.default_sequence_attrs and value.shape[0] != len(self)
            ) and ("_bounds" not in key and value.shape[1] != 2):
                raise ValueError(
                    f"Attribute {key} must have shape [{len(self)}, ...] (got {list(value.shape)})"
                )
