import pandas as pd

from eq.data import Sequence


def train_val_test_split_sequence(
    seq: Sequence,
    start_ts: pd.Timestamp,
    train_start_ts: pd.Timestamp,
    val_start_ts: pd.Timestamp,
    test_start_ts: pd.Timestamp,
    freq: pd.Timedelta = pd.Timedelta("1 day"),
):
    """Generate train, validation and test subsequences.

    Original sequence with events in [start_ts, end_ts] is split into 3 parts:
    1) train: Includes events in [start_ts, val_start_ts], t_nll_start = start_ts
    2) val: Includes events in [start_ts, test_start_ts], t_nll_start = val_start_ts
    3) test: Includes events in [start_ts, end_ts], t_nll_start = test_start_ts
    """
    # Start of the train / val / test intervals as timestamps
    start_ts = pd.Timestamp(start_ts)
    if train_start_ts is not None:
        train_start_ts = pd.Timestamp(train_start_ts)
        t_train_start = (train_start_ts - start_ts) / freq
    else:
        t_train_start = seq.t_start

    val_start_ts = pd.Timestamp(val_start_ts)
    t_val_start = (val_start_ts - start_ts) / freq

    test_start_ts = pd.Timestamp(test_start_ts)
    t_test_start = (test_start_ts - start_ts) / freq
    freq = pd.Timedelta(freq)

    # Start of the train / val / test intervals as floats
    seq_train = seq.get_subsequence(seq.t_start, t_val_start)
    seq_train.t_nll_start = t_train_start
    seq_val = seq.get_subsequence(seq.t_start, t_test_start)
    seq_val.t_nll_start = t_val_start
    seq_test = seq.get_subsequence(seq.t_start, seq.t_end)
    seq_test.t_nll_start = t_test_start
    return seq_train, seq_val, seq_test
