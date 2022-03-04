import tempfile
from dataclasses import dataclass

import numpy as np
import wandb
from tqdm.auto import tqdm

import eq


@dataclass
class DummyCatalog(eq.data.Catalog):
    train: eq.data.InMemoryDataset
    val: eq.data.InMemoryDataset
    test: eq.data.InMemoryDataset
    metadata: dict


def trim_train_and_test(
    catalog: eq.data.Catalog,
    train_frac: float,
    val_frac: float,
) -> eq.data.Catalog:
    """Shorten the train and val sequences.

    Args:
        catalog: Catalog, where the sequences must be shortened.
        train_frac: Fraction of the train sequence used for training.
        val_frac: Fraction of the val sequence used for training.
    """
    if not len(catalog.train) == len(catalog.val) == len(catalog.test) == 1:
        raise ValueError("Expected # of train/val/test sequences to be 1")
    if not 0 < train_frac <= 1:
        raise ValueError(f"train_frac must be in (0, 1] (got {train_frac})")
    if not 0 < val_frac <= 1:
        raise ValueError(f"val_frac must be in (0, 1] (got {val_frac})")
    # Cut the training sequence
    train_seq = catalog.train[0]
    train_duration = train_seq.t_end - train_seq.t_start
    new_train_start = train_seq.t_end - train_duration * train_frac
    new_train_seq = train_seq.get_subsequence(new_train_start, train_seq.t_end)
    new_train_seq.t_nll_start = new_train_start

    # Cut the validation sequence
    val_seq = catalog.val[0]
    val_duration = val_seq.t_end - val_seq.t_nll_start
    new_val_end = val_seq.t_nll_start + val_duration * val_frac
    new_val_seq = val_seq.get_subsequence(new_train_start, new_val_end)
    new_catalog = DummyCatalog(
        train=eq.data.InMemoryDataset([new_train_seq]),
        val=eq.data.InMemoryDataset([new_val_seq]),
        test=catalog.test,
        metadata=catalog.metadata.copy(),
    )
    return new_catalog


def split_sequence(seq, mean_batch_size=50):
    num_nll_events = (seq.arrival_times >= seq.t_nll_start).sum().item()
    num_splits = int(num_nll_events / mean_batch_size)
    linspace = np.linspace(seq.t_nll_start, seq.t_end, num_splits + 1)
    short_sequences = []
    for start, end in tqdm(zip(linspace[:-1], linspace[1:]), total=len(linspace) - 1):
        short_seq = seq.get_subsequence(seq.t_start, end)
        short_seq.t_nll_start = start
        short_sequences.append(short_seq)
    return eq.data.InMemoryDataset(short_sequences)


def split_minibatches(catalog: eq.data.Catalog) -> eq.data.Catalog:
    if not len(catalog.train) == len(catalog.val) == len(catalog.test) == 1:
        raise ValueError("Expected # of train/val/test sequences to be 1")
    d_train = split_sequence(catalog.train[0])
    d_val = split_sequence(catalog.val[0])
    d_test = split_sequence(catalog.test[0])
    return DummyCatalog(
        train=d_train, val=d_val, test=d_test, metadata=catalog.metadata
    )


def load_model(model_name, url):
    """Load W&B model given the run url."""
    api = wandb.Api()
    model_class = getattr(eq.models, model_name)
    run = api.run(url)
    best_model = [file for file in run.files() if file.name == "best_model.ckpt"][0]
    with tempfile.TemporaryDirectory() as tmpdir:
        model = model_class.load_from_checkpoint(
            best_model.download(tmpdir, replace=True).name
        )
    return model
