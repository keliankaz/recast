import io
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import requests
import torch

from eq.data import Catalog, InMemoryDataset, Sequence, ContinuousMarks, default_catalogs_dir

from .utils import train_val_test_split_sequence


def trim(x_min: float, x_max: float, percentile: float) -> Tuple[float, float]:
    """Decrease length of the interval by percentile from each side."""
    length = x_max - x_min
    return (x_min + length * percentile, x_min + length * (1.0 - percentile))


class White(Catalog):
    url = "https://data.mendeley.com/public-files/datasets/7ywkdx7c62/files/7de0df7a-a90a-4307-a090-3b00c8c1235d/file_downloaded"

    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "White",
        mag_completeness: float = 0.6,
        train_start_ts: pd.Timestamp = pd.Timestamp("2009-01-01"),
        val_start_ts: pd.Timestamp = pd.Timestamp("2014-01-01"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2017-01-01"),
    ):
        metadata = {
            "name": "White",
            "freq": "1D",
            "mag_roundoff_error": 0.01,
            "mag_completeness": mag_completeness,
            "start_ts": pd.Timestamp("2008-01-01"),
            "end_ts": pd.Timestamp("2021-01-01"),
        }
        super().__init__(root_dir=root_dir, metadata=metadata)

        self.full_sequence = InMemoryDataset.load_from_disk(
            self.root_dir / "full_sequence.pt"
        )[0]

        self.metadata["train_start_ts"] = pd.Timestamp(train_start_ts)
        self.metadata["val_start_ts"] = pd.Timestamp(val_start_ts)
        self.metadata["test_start_ts"] = pd.Timestamp(test_start_ts)
        seq_train, seq_val, seq_test = train_val_test_split_sequence(
            seq=self.full_sequence,
            start_ts=self.metadata["start_ts"],
            train_start_ts=self.metadata["train_start_ts"],
            val_start_ts=self.metadata["val_start_ts"],
            test_start_ts=self.metadata["test_start_ts"],
        )

        self.train = InMemoryDataset([seq_train])
        self.val = InMemoryDataset([seq_val])
        self.test = InMemoryDataset([seq_test])

    @property
    def required_files(self):
        return ["full_sequence.pt", "metadata.pt"]

    def generate_catalog(self):
        print("Downloading...")
        stream = requests.get(self.url).content
        raw_df = pd.read_csv(
            io.StringIO(stream.decode("utf-8")),
            delim_whitespace=True,
            index_col="event_id",
        )
        # hack: swapped column names in the Mendeley file
        raw_df[["time", "Ml"]] = raw_df[["Ml", "time"]]
        raw_df["time"] = pd.to_datetime(raw_df["time"])

        # Select events with magnitude above Mc
        large_mag_df = raw_df.loc[raw_df["Ml"] > self.metadata["mag_completeness"]]

        # Remove events on the boundary - sensors are noisy there
        lat_range = [large_mag_df["latitude"].min(), large_mag_df["latitude"].max()]
        lon_range = [large_mag_df["longitude"].min(), large_mag_df["longitude"].max()]

        new_lat_range = trim(lat_range[0], lat_range[1], 0.05)
        new_lon_range = trim(lon_range[0], lon_range[1], 0.05)

        indicator = [
            (large_mag_df["latitude"] > new_lat_range[0])
            & (large_mag_df["latitude"] < new_lat_range[1])
            & (large_mag_df["longitude"] > new_lon_range[0])
            & (large_mag_df["longitude"] < new_lon_range[1])
        ]
        subset_df = large_mag_df.loc[indicator[0]].copy()
        subset_df.sort_values(by="time", inplace=True)

        # Save full event sequence as InMemoryDataset
        start_ts = self.metadata["start_ts"]
        end_ts = self.metadata["end_ts"]
        assert subset_df.time.min() > start_ts
        assert subset_df.time.max() < end_ts

        t_start = 0.0
        t_end = (end_ts - start_ts) / pd.Timedelta("1 day")
        arrival_times = ((subset_df.time - start_ts) / pd.Timedelta("1 day")).values
        inter_times = np.diff(arrival_times, prepend=[t_start], append=[t_end])
        mag = subset_df.Ml.values

        seq = Sequence(
            inter_times=torch.as_tensor(inter_times, dtype=torch.float32),
            t_start=t_start,
            mag=ContinuousMarks(mag,[self.metadata["mag_completeness"],10]),
        )
        dataset = InMemoryDataset(sequences=[seq])

        dataset.save_to_disk(self.root_dir / "full_sequence.pt")
