from pathlib import Path
from typing import List, Union

import torch
import torch.utils.data

from .batch import Batch
from .sequence import Sequence


class InMemoryDataset(torch.utils.data.Dataset):
    """Dataset represented by a list of event sequences stored in memory."""

    def __init__(self, sequences: List[Sequence]):
        if any(not isinstance(seq, Sequence) for seq in sequences):
            raise ValueError("sequences must be a list of eq.data.Sequence")
        self.sequences = sequences

    def __getitem__(self, key: int) -> Sequence:
        return self.sequences[key]

    def __len__(self):
        return len(self.sequences)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"

    def __add__(self, other: "InMemoryDataset") -> "InMemoryDataset":
        return InMemoryDataset(self.sequences + other.sequences)

    @staticmethod
    def load_from_disk(path: Union[str, Path]) -> "InMemoryDataset":
        data = torch.load(path)
        sequences = [Sequence(**seq) for seq in data]
        return InMemoryDataset(sequences=sequences)

    def save_to_disk(self, path: Union[str, Path]):
        torch.save([seq.state_dict() for seq in self.sequences], path)

    def apply_(self, function):
        """Apply function to all sequences in the dataset."""
        self.sequences = [function(seq) for seq in self.sequences]
        return self

    def to(self, device):
        """Move all sequences in the dataset to the specified device."""

        def to_device(seq: Sequence) -> Sequence:
            return seq.to(device=device)

        return self.apply_(to_device)

    def get_dataloader(self, batch_size=1, shuffle=False, **kwargs):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=Batch.from_list,
            **kwargs,
        )
