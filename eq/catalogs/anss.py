from pathlib import Path
from typing import Union
import os
import warnings

from eq.data import (
    Catalog,
    InMemoryDataset,
    Sequence,
    default_catalogs_dir,
    ContinuousMarks,)
import pandas as pd
from obspy.clients.fdsn import Client
import datetime
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import BallTree
import torch


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
        include_depth: bool = True
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
            "include_depth": include_depth,
        }
        
        # set a private variable for the magnitude completeness of the ANSS catalog to avoid
        # downloading the global catalog multiple times. The detault behavior downloads the global catalog
        # at M4.5 once and than uses this file for all subsequent calls, if the file exists.
        self.__anss_mag_completeness = 4.5
        if metadata["mag_completeness"] < self.__anss_mag_completeness:
            warnings.warn(f"The magnitude completeness of the ANSS catalog (~{self.__anss_mag_completeness}) is higher than the specified magnitude of completeness {metadata['mag_completeness']}")
            self.__mag_completeness = metadata["mag_completeness"]
        else: 
            self.__mag_completeness = self.__anss_mag_completeness

    
        super().__init__(root_dir=root_dir, metadata=metadata)
        

        self.train = InMemoryDataset.load_from_disk(self.root_dir / "train.pt")
        self.val = InMemoryDataset.load_from_disk(self.root_dir / "val.pt")
        self.test = InMemoryDataset.load_from_disk(self.root_dir / "test.pt")

    @property
    def required_files(self):
        return ["train.pt", "val.pt", "test.pt", "metadata.pt"]

    def get_and_save_catalog(
        self,
        filename: Union[str, Path] = "_temp_local_catalog.csv",
        starttime: str = "2019-01-01",
        endtime: str = "2020-01-01",
        latitude_range: list[float] = [-90, 90],
        longitude_range: list[float] = [-180, 180],
        minimum_magnitude: float = 4.5,
        default_client_name: str = "IRIS",
        reload: bool = True,
    ) -> pd.DataFrame:
        """
        Gets earthquake catalog for the specified region and minimum event
        magnitude and writes the catalog to a file.

        By default, events are retrieved from the NEIC PDE catalog for recent
        events and then the ISC catalog when it becomes available. These default
        results include only that catalog's "primary origin" and
        "primary magnitude" for each event.
        """

        if longitude_range[1] > 180:
            longitude_range[1] = 180
            warnings.warn("Longitude range exceeds 180 degrees. Setting to 180.")

        if longitude_range[0] < -180:
            longitude_range[0] = -180
            warnings.warn("Longitude range exceeds -180 degrees. Setting to -180.")

        if latitude_range[1] > 90:
            latitude_range[1] = 90
            warnings.warn("Latitude range exceeds 90 degrees. Setting to 90.")

        if latitude_range[0] < -90:
            latitude_range[0] = -90
            warnings.warn("Latitude range exceeds -90 degrees. Setting to -90.")

        client_name = default_client_name

        querry = dict(
            starttime=starttime,
            endtime=endtime,
            minmagnitude=minimum_magnitude,
            minlatitude=latitude_range[0],
            maxlatitude=latitude_range[1],
            minlongitude=longitude_range[0],
            maxlongitude=longitude_range[1],
        )

        if not (
            reload is False
            and os.path.exists(filename)
            and np.load(
                os.path.splitext(filename)[0] + "_metadata.npy", allow_pickle=True
            ).item()
            == querry
        ):
            warnings.warn(f"Reloading {filename}")

            # Use obspy api to ge  events from the IRIS earthquake client
            client = Client(client_name)
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
            warnings.warn(f"Using existing {filename}")

        df = pd.read_csv(filename, na_values="None")

        # remove rows with NaN values, reset index and provide a warning is any rows were removed
        if df.isna().values.any():
            warnings.warn(
                f"{sum(sum(df.isna().values))} NaN values found in catalog. Removing rows with NaN values."
            )
            df = df.dropna()
            df = df.reset_index(drop=True)

        df.depth = df.depth / 1000  # convert depth from m to km
        
        df["time"] = pd.to_datetime(pd.to_datetime(df["time"], unit="d"))

        return df

    def get_catalog_batch(
        self,
        batch_size: int = 1,
        global_df: pd.DataFrame = None,
        global_mainshock_df: pd.DataFrame = None,
        tree: BallTree = None,
        start_time: pd.Timestamp = None,
        end_time: pd.Timestamp = None,
        global_start_time: pd.Timestamp = None,
    ) -> InMemoryDataset:
        """Builds a batch of earthquake sequences from the provided global
        catalog of earthqukes.

        Each sequence is a space-time window around a major earthquake
        (M greater than self.metadata["minimum_mainshock_mag"]) in the global catalog. The catalog spans from the global start time to an randomly selected end time. The window selected for training (t_nll_start to t_end) is selected so as to remain withing the specified time range (start_time to end_time) and randomly shifted around the mainshock.

        """

        assert global_start_time <= start_time
        assert (end_time - start_time) / pd.Timedelta(days=1) >= self.metadata[
            "t_end"
        ] * 2, "The time range is too short to generate the sequences with duration self.metadata['t_end'], allowing for a random time shift"


        global_mainshock_df = global_mainshock_df.loc[
            (
                global_mainshock_df.time
                <= end_time - pd.Timedelta(days=self.metadata["t_end"])
            )
            & (
                global_mainshock_df.time
                >= start_time + pd.Timedelta(days=self.metadata["t_end"])
            )
        ]

        sequences = []
        for i in tqdm(range(batch_size)):
            # randomly sample an event from the global catalog of major earthquakes
            event = global_mainshock_df.sample(n=1)

            # randomly shift the window of observation around the 'mainshock' in consideration
            # Note that we need to deal with the annoying edge cases.

            # -----------------------|                                set     (*: mainshock)                   |
            # global_start_time ---- |start_time --- sequence_start_time <-*-> sequence_end_time ----- end_time|
            # global_start_time ---- |sequence_start_time <-*-> sequence_end_time -------------------- end_time|
            # global_start_time ---- |start_time -------------------sequence_start_time <-*-> sequence_end_time|

            # Note that the variables get confusing here:
            # self.metadata["t_end"] is the total length of the sequence (in days)
            # start_time is the start time of the set (as a timestamp)
            # end_time is the end time of the set (as a timestamp)
            # global_start_time is the start time of the global catalog (as a timestamp)
            # sequence_start_time is the start time of the sequence (as a timestamp)
            # sequence_end_time is the end time of the sequence (as a timestamp)
            # t_start is the start time of the sequence (a float in days - starting from 0.0)
            # number_of_days_before_nll is the number of days before the NLL interval (a float in days - starting from 0.0)
            # total_number_of_days is the total number of days in the sequence (a float in days - starting from 0.0)

            time_shift = np.random.uniform(0, self.metadata["t_end"])

            sequence_nll_start_time = event.time.item() - datetime.timedelta(
                days=time_shift
            )
            sequence_end_time = event.time.item() + (
                datetime.timedelta(days=self.metadata["t_end"] - time_shift)
            )

            space_index, distances = tree.query_radius(
                np.deg2rad(event[["lat", "lon"]].values),
                r=self.metadata["radius"] / EARTH_RADIUS_KM,
                return_distance=True,
            )
            
            space_index = space_index[0]
            distances = distances[0]
            distances = distances * EARTH_RADIUS_KM

            local_df = global_df.iloc[space_index]
            
            local_df = local_df.assign(distance_km=distances)

            local_df = local_df.loc[local_df.time < sequence_end_time]

            total_number_of_days = (
                sequence_end_time - global_start_time
            ) / pd.Timedelta("1 day")
            
            number_of_days_before_nll = (
                sequence_nll_start_time - global_start_time
            ) / pd.Timedelta("1 day")

            local_df = local_df.sort_values("time", ascending=[True])

            arrival_times = (
                (local_df.time - global_start_time) / pd.Timedelta("1 day")
            ).values
            inter_times = np.diff(
                arrival_times, prepend=[0.0], append=[total_number_of_days]
            )
            mag = local_df.mag.values

            mag_bounds = torch.as_tensor(
                [self.metadata["mag_completeness"], 10.0], dtype=torch.float32
            )

            mag_marks = ContinuousMarks(
                values=torch.as_tensor(mag, dtype=torch.float32),
                bounds=mag_bounds,
            )

            if arrival_times[-1] < number_of_days_before_nll:
                # we need to pad the sequence with zeros
                print(
                    " there is an issue with the sequence, the last event is before the NLL interval"
                )

            sequences.append(
                Sequence(
                    inter_times=torch.as_tensor(inter_times, dtype=torch.float32),
                    t_start=0.0,
                    mag=mag_marks,
                    t_nll_start=number_of_days_before_nll,
                    lat=torch.as_tensor(local_df.lat.values, dtype=torch.float32),
                    lon=torch.as_tensor(local_df.lon.values, dtype=torch.float32),
                    depth=torch.as_tensor(local_df.depth.values, dtype=torch.float32),
                    distance_km=torch.as_tensor(local_df.distance_km.values, dtype=torch.float32),
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
        
        
        # create the raw directory if it doesn't exist
        (self.root_dir.parent / "raw").mkdir(parents=True, exist_ok=True)
        
        print("Downloading/loading...")
        global_catalog_df = self.get_and_save_catalog(
            filename=self.root_dir.parent / "raw" / "anss_global_catalog.csv",
            starttime=self.metadata["train_daterange"][0].strftime("%Y-%m-%d"),
            endtime=self.metadata["test_daterange"][1].strftime("%Y-%m-%d"),
            minimum_magnitude=self.__mag_completeness,
            reload=False,
        )
        
        mainshock_df = global_catalog_df.loc[
            global_catalog_df.mag > self.metadata["minimum_mainshock_mag"]
        ]
        global_tree = BallTree(
            np.deg2rad(global_catalog_df[["lat", "lon"]].values), metric="haversine"
        )

        set_names = ["train", "val", "test"]
        for i_set in set_names:
            print(i_set)

            dataset = self.get_catalog_batch(
                batch_size=int(
                    self.metadata["num_sequences"] * self.metadata[f"{i_set}_frac"]
                ),
                global_df=global_catalog_df,
                global_mainshock_df=mainshock_df,
                tree=global_tree,
                start_time=self.metadata[f"{i_set}_daterange"][0],
                end_time=self.metadata[f"{i_set}_daterange"][1],
                global_start_time=self.metadata["train_daterange"][0],
            )

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
