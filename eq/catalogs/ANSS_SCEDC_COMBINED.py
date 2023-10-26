
# %%
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import eq
from eq.data import Catalog, InMemoryDataset, Sequence
from eq.catalogs import SCEDC
from eq.catalogs import ANSS_MultiCatalog

class CombinedCatalog(Catalog):
    def __init__(self, 
            root_dir: str,
            anss_num_sequences: int = 100,
            anss_t_end_days: float = 365,
            anss_radius_kilometers: float = 1000,
            anss_mag_completeness: float = 5.0,
            scedc_mag_completeness: float = 2.0,
            minimum_mainshock_mag: float = 7.0,
            random_state: int = 123,
        ):
        # Initialize with metadata for the combined catalog
        metadata = {
            "name": "ANSS_MultiCatalog",
            "anss_num_sequences": anss_num_sequences,
            "t_end": anss_t_end_days,
            "radius": anss_radius_kilometers,
            "anss mag_completeness": anss_mag_completeness,
            "scedc mag_completeness": scedc_mag_completeness,
            "minimum_mainshock_mag": minimum_mainshock_mag,
            "random_state": random_state,
        }

        super().__init__(root_dir=root_dir, metadata=metadata)

        # Load the sequences from both catalogs
        SCEDC = eq.catalogs.SCEDC(mag_completeness = self.metadata["scedc mag_completeness"])
        ANSS = eq.catalogs.ANSS_MultiCatalog(num_sequences= self.metadata["anss_num_sequences"], radius_kilometers=self.metadata["radius"], minimum_mainshock_mag=self.metadata["minimum_mainshock_mag"])

        self.scedc_train = SCEDC.train
        self.scedc_val = SCEDC.val
        self.scedc_test = SCEDC.test
        self.anss_train = ANSS.train
        self.anss_val = ANSS.val
        self.anss_test = ANSS.test

        # Combine the sequences into a single catalog
        combined_train = InMemoryDataset(
            sequences=self.scedc_train.sequences + self.anss_train.sequences
        )
        combined_val = InMemoryDataset(
            sequences=self.scedc_val.sequences + self.anss_val.sequences
        )
        combined_test = InMemoryDataset(
            sequences=self.scedc_test.sequences + self.anss_test.sequences
        )

        self.train = combined_train
        self.val = combined_val
        self.test = combined_test

    @property
    def required_files(self):
        return []

    def generate_catalog(self):
        # This method is not used for the combined catalog
        pass