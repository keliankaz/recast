# %%
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import eq


from eq.data import Catalog, InMemoryDataset, Sequence, default_catalogs_dir

def combine_catalogs_sequences(seqlist):
    sequences = []
    for seq in seqlist:
        sequences.append(seq)
    return InMemoryDataset(sequences=sequences)

def build_seqlist(catalogs):
    train_sequences = []
    for catalog in catalogs:
        for seq in range(len(catalog)):
            train_sequences.append(catalog[seq])
    return train_sequences

def subtract_magnitudes(sequences, mag_completeness):
    for seq in sequences:
        seq.mag -= mag_completeness

def get_catalog(scedc_mag_complete, anss_mag_complete, anss_num_seq, include_loc=True, include_depth=True):
    final_dict = {}
    
    scedc = eq.catalogs.SCEDC(mag_completeness=scedc_mag_complete, include_loc=include_loc, include_depth = include_depth)
    anss = eq.catalogs.ANSS_MultiCatalog(    
        num_sequences=anss_num_seq,
        t_end_days=1*365,
        mag_completeness=anss_mag_complete,
        minimum_mainshock_mag=6.0,
        include_loc=include_loc,
        include_depth = include_depth
    )
    #subtract mag_completeness from catalogs
    subtract_magnitudes(scedc.train, 2.0)
    subtract_magnitudes(anss.train, 4.5)
    subtract_magnitudes(scedc.val, 2.0)
    subtract_magnitudes(anss.val, 4.5)
    subtract_magnitudes(scedc.test, 2.0)
    subtract_magnitudes(anss.test, 4.5)

    #make the 2 catalogs into a list for training and val data
    catalogs_train = []
    catalogs_train.append(scedc.train)
    catalogs_train.append(anss.train)

    catalogs_val = []
    catalogs_val.append(scedc.val)
    catalogs_val.append(anss.val)

    catalogs_test = []
    catalogs_test.append(scedc.test)
    catalogs_test.append(anss.test)

    #build list of catalog sequences
    seqlist_train = build_seqlist(catalogs_train)
    seqlist_val = build_seqlist(catalogs_val)
    seqlist_test = build_seqlist(catalogs_test)

    #combined the sequences into one memory i=object to train on
    combined_cat_train = combine_catalogs_sequences(seqlist_train)
    combined_cat_val = combine_catalogs_sequences(seqlist_val)
    combined_cat_test = combine_catalogs_sequences(seqlist_test)

    #individual test sets for testing purposes
    scedc_test = scedc.test
    anss_test = anss.test

    #build final dict
    final_dict['combined train'] = combined_cat_train
    final_dict['combined val'] = combined_cat_val
    final_dict['combined test'] = combined_cat_test
    final_dict['SCEDC test'] = scedc_test
    final_dict['ANSS test'] = anss_test
    final_dict['SCEDC train'] = scedc.train
    final_dict['ANSS train'] = anss.train

    return final_dict


class ANSS_SCEDC_Combined():
    def __init__(
        get_catalog(2.0, 4.5, 99)
    )   