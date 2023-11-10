# %%
from pathlib import Path
from typing import Union
import os
import warnings

from eq.data import Catalog, InMemoryDataset, Sequence, ContinuousMarks, default_catalogs_dir
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import datetime
import numpy as np
from tqdm import tqdm

from sklearn.neighbors import BallTree

import torch

# %%
EARTH_RADIUS_KM = 6378.1


class ANSS_MultiCatalog(Catalog):
    """Multiple catalogs dowloaded from ANSS using obspy."""

    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "ANSS_MultiCatalog",
        num_sequences: int = 1000,
        max_length=10000,
        train_frac: float = 0.6,
        train_daterange: list[pd.Timestamp, pd.Timestamp] = [
            pd.Timestamp("1990-01-01"),
            pd.Timestamp("2010-01-01"),
        ],
        val_frac: float = 0.2,
        val_daterange: list[pd.Timestamp, pd.Timestamp] = [
            pd.Timestamp("2010-01-01"),
            pd.Timestamp("2015-01-01"),
        ],
        test_frac: float = 0.2,
        test_daterange: list[pd.Timestamp, pd.Timestamp] = [
            pd.Timestamp("2015-01-01"),
            pd.Timestamp("2020-01-01"),
        ],
        t_end_days: float = 365,
        radius_kilometers: float = 1000,
        mag_completeness: float = 5.0,
        minimum_mainshock_mag: float = 7.0,
        random_state: int = 123,
    ):
        metadata = {
            "name": "ANSS_MultiCatalog",
            "num_sequences": num_sequences,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "test_frac": test_frac,
            "t_end": t_end_days,
            "max_length": max_length,
            "radius": radius_kilometers,
            "mag_completeness": mag_completeness,
            "minimum_mainshock_mag": minimum_mainshock_mag,
            "random_state": random_state,
            "train_daterange": train_daterange,
            "val_daterange": val_daterange,
            "test_daterange": test_daterange,
        }

        super().__init__(root_dir=root_dir, metadata=metadata)

        self.train = InMemoryDataset.load_from_disk(self.root_dir / "train.pt")
        self.val = InMemoryDataset.load_from_disk(self.root_dir / "val.pt")
        self.test = InMemoryDataset.load_from_disk(self.root_dir / "test.pt")

    @property
    def required_files(self):
        return ["train.pt", "val.pt", "test.pt", "metadata.pt"]

    def get_catalog_batch(
        self,
        whole_global_df,
        batch_size: int = None,
        global_start_time: pd.Timestamp = None,
        global_end_time: pd.Timestamp = None,
    ) -> InMemoryDataset:
        
        global_df = whole_global_df.loc[
            (whole_global_df.time > global_start_time) & 
            (whole_global_df.time < global_end_time)
        ]

        major_earthquakes_df = global_df.loc[
            global_df.mag > self.metadata["minimum_mainshock_mag"]
        ]

        tree = BallTree(
            np.deg2rad(global_df[["lat", "lon"]].values), metric="haversine"
        )

        sequences = []
        for i in tqdm(range(batch_size)):
            # randomly sample an event from the global catalog of major earthquakes
            event = major_earthquakes_df.sample(n=1)

            # randomly shift the window of observation around the 'mainshock' in consideration
            # Note that we need to deal with the annoying edge cases.
            time_shift = np.random.uniform(0, self.metadata["t_end"])
            start_time = max(
                [
                    event.time.item() - datetime.timedelta(days=time_shift),
                    global_start_time,
                ]
            )

            end_time = min(
                [
                    event.time.item()
                    + (datetime.timedelta(days=self.metadata["t_end"] - time_shift)),
                    global_end_time,
                ]
            )

            space_index = tree.query_radius(
                np.deg2rad(event[["lat", "lon"]].values),
                r=self.metadata["radius"] / EARTH_RADIUS_KM,
                return_distance=False,
            )[0]

            local_df = global_df.iloc[space_index]

            local_df = local_df.loc[
                (local_df.time > start_time) & (local_df.time < end_time)
            ]

            t_start = 0.0
            t_end = min(
                (global_end_time - start_time) / pd.Timedelta("1 day"),
                self.metadata["t_end"],
            )

            local_df = local_df.sort_values("time", ascending=[True])

            arrival_times = (
                (local_df.time - start_time) / pd.Timedelta("1 day")
            ).values
            inter_times = np.diff(arrival_times, prepend=[t_start], append=[t_end])
            mag = local_df.mag.values
            mag = ContinuousMarks(mag, [self.metadata["mag_completeness"], 10])

            sequences.append(
                Sequence(
                    inter_times=torch.as_tensor(inter_times, dtype=torch.float32),
                    t_start=t_start,
                    mag=mag,
                )
            )

        return InMemoryDataset(sequences=sequences)

    def generate_catalog(self):
        """
        Generate earthquake sequences from the Global ComCat catalog from ANSS.
        based on the metadata provided during object initialization, and save them to disk in
        three separate PyTorch datasets for training, validation, and testing. The generated
        earthquake sequences are used to train a deep learning model for earthquake forecasting.

        This catalog is assembled in a two step process, first we querry a glabal catalog of
        magnitude 6 or greater. Next, we querry data around these major earthquakes.

        Returns:
            None
        """

        dates = np.concatenate(
            [
                self.metadata["train_daterange"], 
                self.metadata["val_daterange"],
                self.metadata["test_daterange"],
            ]
        )
        start_date = min(dates)
        end_date = max(dates)
        
        raw_dir = str(self.root_dir) + '_raw'
        filename = raw_dir + '/anss.csv'
        
        querry = dict(
            starttime=UTCDateTime(start_date),
            endtime=UTCDateTime(end_date),
            magnitudetype="MW",
            minmagnitude=self.metadata["mag_completeness"],
        )
        
        if not (
            os.path.exists(filename)
            and np.load(
                os.path.splitext(filename)[0] + "_metadata.npy", allow_pickle=True
            ).item() == querry
        ):
            print("Downloading...")
            
            os.mkdir(raw_dir)
            
            # Use obspy api to ge  events from the IRIS earthquake client
            client=Client("IRIS")
            cat = client.get_events(**querry)

            # Write the earthquakes to a file
            f = open(filename, "w")
            f.write("time,lat,lon,depth,mag\n")
            for event in cat:
                loc = event.preferred_origin()
                lat = loc.latitude
                lon = loc.longitude
                dep = loc.depth
                time = loc.time.matplotlib_date
                mag = event.preferred_magnitude().mag
                f.write("{},{},{},{},{}\n".format(time, lat, lon, dep, mag))
            f.close()

            # Save querry to metadatafile
            np.save(os.path.splitext(filename)[0] + "_metadata.npy", querry) 
        else:
            print(f"Using existing catalog: {filename}")
        
        catalog_df = pd.read_csv(filename, na_values="None")
        catalog_df["time"] = pd.to_datetime(catalog_df["time"], unit='d')

        set_names = ["train", "val", "test"]
        for i_set in set_names:
            print(i_set)
            n = int(self.metadata["num_sequences"] * self.metadata[f"{i_set}_frac"])
            dataset = self.get_catalog_batch(catalog_df, n, *self.metadata[f"{i_set}_daterange"])
            dataset.save_to_disk((self.root_dir / f"{i_set}.pt"))
        print("Success!")

    @staticmethod
    def obspy2pd(cat):
        times = []
        lats = []
        lons = []
        deps = []
        magnitudes = []
        magnitudestype = []
        for event in cat:
            if len(event.origins) != 0 and len(event.magnitudes) != 0:
                times.append(event.origins[0].time.datetime)
                lats.append(event.origins[0].latitude)
                lons.append(event.origins[0].longitude)
                deps.append(event.origins[0].depth)
                magnitudes.append(event.magnitudes[0].mag)
                magnitudestype.append(event.magnitudes[0].magnitude_type)

        df = pd.DataFrame(
            {
                "time": times,
                "lat": lats,
                "lon": lons,
                "depth": deps,
                "mag": magnitudes,
                "type": magnitudestype,
            },
        )

        return df.sort_values(by=["time"])


# %%
if __name__ == "__main__":
    catalog = ANSS_MultiCatalog()

# %%
