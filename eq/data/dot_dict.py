import warnings
from typing import Any, Optional

import numpy as np
import torch


def size_repr(key: str, value: Any, indent=0) -> str:
    """String containing the size / shape of an object (e.g. a tensor, array)."""
    if isinstance(value, torch.Tensor) and value.dim() == 0:
        out = value.item()
    elif isinstance(value, torch.Tensor):
        out = f"{str(list(value.size()))}"
    elif isinstance(value, list):
        out = f"[{len(value)}]"
    elif isinstance(value, float):
        out = f"{np.round(value, decimals=3)}"
    else:
        out = str(value)

    return f"{' ' * indent}{key}: {out}"


def _is_float(tensor: torch.Tensor):
    """Check if torch.Tensor is of type torch.float32 or torch.float64."""
    return (
        tensor.dtype == torch.float32
        or tensor.dtype == torch.float64
        or tensor.dtype == torch.float16
    )


class DotDict:
    """Dictionary where elements can be accessed as dict.entry."""

    def __init__(self, data: Optional[dict] = None, **kwargs):
        self.__dict__["_data"] = {}
        if data is not None:
            self._data.update(data)
        if kwargs:
            self._data.update(kwargs)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __getattr__(self, key):
        if "_data" not in self.__dict__:
            raise AttributeError(
                f"{self.__class__.__name__} hasn't been initialized properly. "
                "Use super().__init__() before getting/setting attributes."
            )
        if key not in self._data:
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute '{key}'"
            )
        return self._data[key]

    def __setattr__(self, key, value):
        if "_data" not in self.__dict__:
            raise AttributeError(
                f"{self.__class__.__name__} hasn't been initialized properly. "
                "Use super().__init__() before getting/setting attributes."
            )
        self._data[key] = value

    def __delattr__(self, key):
        del self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default=default)

    def _is_valid_key(self, key):
        return key[0] != "_" and self[key] is not None

    def keys(self):
        return [k for k in self._data.keys() if self._is_valid_key(k)]

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def __iter__(self):
        for key in sorted(self.keys()):
            yield key, self[key]

    def __contains__(self, key):
        return key in self.keys()

    def apply_(self, function):
        """Apply function to all attributes."""
        for key, value in self._data.items():
            if isinstance(value, DotDict):
                self[key] = value.apply_(function)
            else:
                self[key] = function(value)
        return self

    def to(self, device, non_blocking=False):
        """Move all tensors to the specified device."""

        def to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(device=device, non_blocking=non_blocking)
            else:
                return x

        return self.apply_(to_device)

    def cpu(self, non_blocking=False):
        """Move all tensors to CPU."""
        return self.to("cpu", non_blocking=non_blocking)

    def cuda(self, non_blocking=False):
        """Move all tensors to GPU."""
        return self.to("cuda", non_blocking=non_blocking)

    def double(self):
        """Convert all float tensors to torch.float64."""

        def to_double(x):
            if isinstance(x, torch.Tensor) and _is_float(x):
                return x.double()
            else:
                return x

        return self.apply_(to_double)

    def float(self):
        """Convert all float tensors to torch.float32."""

        def to_float(x):
            if isinstance(x, torch.Tensor) and _is_float(x):
                return x.float()
            else:
                return x

        return self.apply_(to_float)

    def __repr__(self):
        cls = self.__class__.__name__
        info = [size_repr(key, value, indent=2) for key, value in self.items()]
        info = ",\n".join(info)
        if len(info) > 0:
            return f"{cls}(\n{info}\n)"
        else:
            return f"{cls}()"

    @property
    def device(self):
        devices = [v.device for v in self.values() if isinstance(v, torch.Tensor)]
        num_devices = len(set(devices))
        if num_devices > 1:
            warnings.warn(
                f"Data tensors found on {num_devices} different devices. "
                "Move all tensors to the same device with `.to(device)`"
            )
        return devices[0]
