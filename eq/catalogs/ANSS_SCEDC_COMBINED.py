
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
    def __init__(self, catalogs):
        # Initialize with metadata for the combined catalog
        metadata = {
            "catalogs": [catalog.__class__.__name__ for catalog in catalogs]
            # You might want to add more metadata or information about the combined catalogs here
        }

        super().__init__(root_dir="combined", metadata=metadata)

        combined_train_sequences = []
        for catalog in catalogs:
            combined_train_sequences.extend(catalog.train)

        combined_val_sequences = []
        for catalog in catalogs:
            combined_val_sequences.extend(catalog.val)
        
        combined_test_sequences = []
        for catalog in catalogs:
            combined_test_sequences.extend(catalog.test)

        combined_train_data = InMemoryDataset(sequences=combined_train_sequences)
        combined_val_data = InMemoryDataset(sequences=combined_val_sequences)
        combined_test_data = InMemoryDataset(sequences=combined_test_sequences)

        dict = {}
        dict['combined_train'] = combined_train_data
        dict['combined_val'] = combined_val_data
        dict['combined_test'] = combined_test_data
        for catalog in catalogs:
            name = str(catalog.metadata['name'])
            dict[name +'_test'] = catalog.test
        
        self.combined_info = dict


    @property
    def required_files(self):
        return []

    def generate_catalog(self):
        # This method is not used for the combined catalog
        pass