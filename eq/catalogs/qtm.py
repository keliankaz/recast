import io
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import requests
import torch

from eq.data import Catalog, InMemoryDataset, Sequence, ContinuousMarks, default_catalogs_dir

from .utils import train_val_test_split_sequence

LAT_RANGE = {
    "SaltonSea": [32.5, 33.3],
    "SanJacinto": [33.0, 34.0],
}

LON_RANGE = {
    "SaltonSea": [-116.0, -115.0],
    "SanJacinto": [-117.0, -116.0],
}

VALID_REGIONS = set(LAT_RANGE.keys())


class QTM(Catalog):
    url = "https://service.scedc.caltech.edu/ftp/QTMcatalog/qtm_final_12dev.hypo"

    def __init__(
        self,
        root_dir: Union[str, Path],
        region: str = "SanJacinto",
        mag_completeness: float = 1.0,
        train_start_ts: pd.Timestamp = pd.Timestamp("2009-01-01"),
        val_start_ts: pd.Timestamp = pd.Timestamp("2014-01-01"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2016-01-01"),
    ):
        if region not in VALID_REGIONS:
            raise ValueError(
                f"Invalid region {region}, must be one of {VALID_REGIONS}."
            )

        lat_range = LAT_RANGE[region]
        lon_range = LON_RANGE[region]

        metadata = {
            "name": f"QTM{region}",
            "region": region,
            "freq": "1D",
            "mag_roundoff_error": 0.01,
            "mag_completeness": mag_completeness,
            "lat_range": lat_range,
            "lon_range": lon_range,
            "start_ts": pd.Timestamp("2008-01-01"),
            "end_ts": pd.Timestamp("2018-01-01"),
        }
        super().__init__(root_dir=root_dir, metadata=metadata)

        # Load the full sequence
        self.full_sequence = InMemoryDataset.load_from_disk(
            self.root_dir / "full_sequence.pt"
        )[0]

        # Split full sequence into train / val / test parts
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
        raw_df = pd.read_csv(io.StringIO(stream.decode("utf-8")), delim_whitespace=True)
        raw_df["date"] = pd.to_datetime(
            raw_df[["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"]]
        )
        raw_df.sort_values(by=["date"], inplace=True)

        print("Processing...")
        # Select relevant events from the dataframe
        indicator = (
            (raw_df.LATITUDE > self.metadata["lat_range"][0])
            & (raw_df.LATITUDE < self.metadata["lat_range"][1])
            & (raw_df.LONGITUDE > self.metadata["lon_range"][0])
            & (raw_df.LONGITUDE < self.metadata["lon_range"][1])
            & (raw_df.MAGNITUDE > self.metadata["mag_completeness"])
        )
        zone_df = raw_df.loc[indicator, :]

        timestamps = zone_df.date.to_numpy()
        mag = zone_df["MAGNITUDE"].values

        # Compute inter-event times
        start_ts = np.datetime64(self.metadata["start_ts"])
        end_ts = np.datetime64(self.metadata["end_ts"])
        assert timestamps.min() > start_ts
        assert timestamps.max() < end_ts

        arrival_times = (timestamps - start_ts) / pd.Timedelta("1 day")
        t_start = 0.0
        t_end = (end_ts - start_ts) / pd.Timedelta("1 day")
        inter_times = np.diff(arrival_times, prepend=[t_start], append=[t_end])
        seq = Sequence(
            inter_times=torch.as_tensor(inter_times, dtype=torch.float32),
            t_start=t_start,
            mag=ContinuousMarks(mag,[self.metadata["mag_completeness"],10]),
        )
        full_sequence = InMemoryDataset(sequences=[seq])
        full_sequence.save_to_disk(self.root_dir / "full_sequence.pt")


class QTMSanJacinto(QTM):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "QTMSanJacinto",
        mag_completeness: float = 1.0,
        val_start_ts: pd.Timestamp = pd.Timestamp("2014-01-01"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2016-01-01"),
    ):
        super().__init__(
            root_dir=root_dir,
            region="SanJacinto",
            mag_completeness=mag_completeness,
            val_start_ts=val_start_ts,
            test_start_ts=test_start_ts,
        )


class QTMSaltonSea(QTM):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "QTMSaltonSea",
        mag_completeness: float = 1.0,
        val_start_ts: pd.Timestamp = pd.Timestamp("2014-01-01"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2016-01-01"),
    ):
        super().__init__(
            root_dir=root_dir,
            region="SaltonSea",
            mag_completeness=mag_completeness,
            val_start_ts=val_start_ts,
            test_start_ts=test_start_ts,
        )
