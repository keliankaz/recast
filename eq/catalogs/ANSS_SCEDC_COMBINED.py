
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
            for train in catalog.train:
                combined_train_sequences.append(train)

        combined_val_sequences = []
        for catalog in catalogs:
            for val in catalog.val:
                combined_val_sequences.append(val)
        
        combined_test_sequences = []
        for catalog in catalogs:
            for test in catalog.test:
                combined_test_sequences.append(test)


        combined_train_data = self.even_sequences(combined_train_sequences, 1000)
        combined_val_data = self.even_sequences(combined_val_sequences, 1000)
        combined_test_data = self.even_sequences(combined_test_sequences, 1000)

        dict = {}
        dict['combined_train'] = InMemoryDataset(sequences=combined_train_data)
        dict['combined_val'] = InMemoryDataset(sequences=combined_val_data)
        dict['combined_test'] = InMemoryDataset(sequences=combined_test_data)
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
                    start = sequence.arrival_times[i * target_sequence_length]
                    end = sequence.arrival_times[min((i + 1) * target_sequence_length, len(sequence.arrival_times) - 1)]
                    new_seq = sequence.get_subsequence(start, end)
                    new_sequences.append(new_seq)

                    # Debugging: Print information to track data changes
                    print(f"Length of new sequence: {len(new_seq.arrival_times)}")
                    print(f"New sequence content: {new_seq.arrival_times}")
                    
                    if len(new_seq.arrival_times) > 0:  # Check for non-empty sequences
                        new_sequences.append(new_seq)
                    else:
                        print("Warning: Empty sequence found.")
                        # Handle empty sequence if necessary

        return InMemoryDataset(sequences=new_sequences)

    @property
    def required_files(self):
        return []

    def generate_catalog(self):
        # This method is not used for the combined catalog
        pass