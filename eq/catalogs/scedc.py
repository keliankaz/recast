import io
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import requests
import torch

from eq.data import Catalog, InMemoryDataset, Sequence, ContinuousMarks, default_catalogs_dir

from .utils import train_val_test_split_sequence

COL_NAMES = [
    "date",
    "time",
    "ET",
    "GT",
    "magnitude",
    "M",
    "latitude",
    "longitude",
    "depth",
    "Q",
    "EVID",
    "NPH",
    "NGRM",
]


class SCEDC(Catalog):
    url = "https://service.scedc.caltech.edu/ftp/catalogs/SCEC_DC/"

    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "SCEDC",
        mag_completeness: float = 2.0,
        train_start_ts: pd.Timestamp = pd.Timestamp("1985-01-01"),
        val_start_ts: pd.Timestamp = pd.Timestamp("2005-01-01"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2014-01-01"),
    ):
        metadata = {
            "name": f"SCEDC",
            "freq": "1D",
            "mag_roundoff_error": 0.1,
            "mag_completeness": mag_completeness,
            "start_ts": pd.Timestamp("1981-01-01"),
            "end_ts": pd.Timestamp("2020-01-01"),
        }

        super().__init__(root_dir=root_dir, metadata=metadata)

        # Load the full sequence
        self.full_sequence = InMemoryDataset.load_from_disk(
            self.root_dir / "full_sequence.pt"
        )[0]

        # Split full sequence into train / val / test parts
        if train_start_ts is None:
            train_start_ts = metadata["start_ts"]
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

        raw_df = []

        year_range = range(
            self.metadata["start_ts"].year,
            self.metadata["end_ts"].year,
        )

        for iyear in year_range:
            stream = requests.get(url="{}{}.catalog".format(self.url, iyear)).content

            raw_df.append(
                pd.read_csv(
                    io.StringIO(stream.decode("utf-8")),
                    delim_whitespace=True,
                    header=0,
                    names=COL_NAMES,
                    comment="#",
                    index_col=False,
                )
            )

        raw_df = pd.concat(raw_df, ignore_index=True)

        # workaround to deal with seconds going up to 60.0
        raw_df["date_time"] = pd.to_datetime(raw_df["date"]) + pd.to_timedelta(
            raw_df["time"]
        )

        raw_df.sort_values(by=["date_time"], inplace=True)
        subset_df = raw_df.loc[raw_df["magnitude"] > self.metadata["mag_completeness"]]

        start_ts = self.metadata["start_ts"]
        end_ts = self.metadata["end_ts"]

        assert subset_df.date_time.min() > start_ts
        assert subset_df.date_time.max() < end_ts

        t_start = 0.0
        t_end = (end_ts - start_ts) / pd.Timedelta("1 day")

        arrival_times = (
            (subset_df.date_time - start_ts) / pd.Timedelta("1 day")
        ).values
        inter_times = np.diff(arrival_times, prepend=[t_start], append=[t_end])
        mag = subset_df["magnitude"].values
        depth = subset_df["depth"].values
        seq = Sequence(
            inter_times=torch.as_tensor(inter_times, dtype=torch.float32),
            mag=ContinuousMarks(mag,[self.metadata["mag_completeness"],10]),
        #     extra_feat=torch.as_tensor(depth, dtype=torch.float32).unsqueeze(-1) # depth has shape N x 1
         )
        dataset = InMemoryDataset(sequences=[seq])
        dataset.save_to_disk(self.root_dir / "full_sequence.pt")
