
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

        # min_sequence_length_train = min(len(sequence.arrival_times) for sequence in combined_train_sequences)
        # min_sequence_length_val = min(len(sequence.arrival_times) for sequence in combined_val_sequences)
        # min_sequence_length_test = min(len(sequence.arrival_times) for sequence in combined_test_sequences)

        combined_train_data = self.even_sequences(combined_train_sequences, 200)
        combined_val_data = self.even_sequences(combined_val_sequences, 200)
        combined_test_data = self.even_sequences(combined_test_sequences, 200)

        dict = {}
        dict['combined_train'] = combined_train_data
        dict['combined_val'] = combined_val_data
        dict['combined_test'] = combined_test_data
        for catalog in catalogs:
            name = str(catalog.metadata['name'])
            dict[name +'_test'] = catalog.test
            dict[name +'_train'] = catalog.train
            dict[name +'_val'] = catalog.val
        
        self.combined_info = dict


    def even_sequences(self, sequences, target_sequence_length):
        new_sequences = []
        for sequence in sequences:
            if len(sequence.arrival_times) > target_sequence_length:
                num_segments = (len(sequence.arrival_times) + target_sequence_length - 1) // target_sequence_length
                for i in range(num_segments):
                    start = i * target_sequence_length
                    end = min((i + 1) * target_sequence_length, len(sequence.arrival_times))
                    new_seq = sequence.extract_events_in_interval(start, end)
                    new_sequences.append(new_seq)

        return InMemoryDataset(sequences=new_sequences)

    @property
    def required_files(self):
        return []

    def generate_catalog(self):
        # This method is not used for the combined catalog
        pass