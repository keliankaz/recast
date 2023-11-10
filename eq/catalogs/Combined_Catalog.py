
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
    def __init__(self, catalogs, sub_sampling_prop_list: list = [.25, .25, .25, .25] , sub_sampling_seq_length: int = 1000):
        
        if sum(sub_sampling_prop_list) != 1:
            raise ValueError("The sum of sub_sampling_prop_list must be equal to 1")

        
        # Initialize with metadata for the combined catalog
        metadata = {
            "catalogs": [catalog.__class__.__name__ for catalog in catalogs],
            "prop_list": sub_sampling_prop_list,
            "sub_length": sub_sampling_seq_length
            # You might want to add more metadata or information about the combined catalogs here
        }

        super().__init__(root_dir="combined", metadata=metadata)
        # proportionalize catalogs
        
        #final dictionary that catalog produces
        dict = {}
        for catalog in catalogs:
            name = str(catalog.metadata['name'])
            dict[name +'_test'] = catalog.test
            dict[name +'_train'] = catalog.train
            dict[name +'_val'] = catalog.val
        
        #sub sample for each catalog
        for catalog in catalogs:
            catalog.train = self.even_sequences(catalog.train.sequences, self.metadata['sub_length'])
            catalog.val = self.even_sequences(catalog.val.sequences, self.metadata['sub_length'])
            catalog.test = self.even_sequences(catalog.test.sequences, self.metadata['sub_length'])

        #proportionalize
        combined_train_data, combined_val_data, combined_test_data = self.proportionalize(catalogs, self.metadata['prop_list'])

        #make into final train, val, test lists
        train = self.sequence_list(combined_train_data)
        val = self.sequence_list(combined_val_data)
        test = self.sequence_list(combined_test_data)


        #Add combined sets to final dict
        dict['combined_train'] = InMemoryDataset(sequences=train)
        dict['combined_val'] = InMemoryDataset(sequences=val)
        dict['combined_test'] = InMemoryDataset(sequences=test)
        
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
            else:
                new_sequences.append(sequence)

        return InMemoryDataset(sequences=new_sequences)
    
    def proportionalize(self, catalogs, prop_list):
        final_train = []
        final_val = []
        final_test = []

        total_train_length = 0
        total_val_length = 0
        total_test_length = 0
        for catalog in catalogs:
            total_train_length += len(catalog.train)
            total_val_length += len(catalog.val)
            total_test_length += len(catalog.test)
        print(total_train_length)

        for i in np.arange(0,len(prop_list)):
            new_catalog_train_length = np.round(prop_list[i] * total_train_length)
            new_catalog_train = catalogs[i].train[0: int(min(len(catalogs[i].train), new_catalog_train_length))]
            final_train.append(new_catalog_train)

            new_catalog_val_length = np.round(prop_list[i] * total_val_length)
            new_catalog_val = catalogs[i].val[0: int(min(len(catalogs[i].val),new_catalog_val_length))]
            final_val.append(new_catalog_val)

            new_catalog_test_length = np.round(prop_list[i] * total_test_length)
            new_catalog_test = catalogs[i].test[0: int(min(len(catalogs[i].test),new_catalog_test_length))]
            final_test.append(new_catalog_test)
        
        return final_train, final_val, final_test

    def sequence_list(self, list):
        final_list = []
        for seqlist in list:
            for seq in seqlist:
                final_list.append(seq)
        return final_list
        

    @property
    def required_files(self):
        return []

    def generate_catalog(self):
        # This method is not used for the combined catalog
        pass