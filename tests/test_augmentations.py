from eq.data.augmentation import jitter, superimpose, sub_radius
from test_data import TestSequence
import torch
import numpy as np

class TestAugmentations:

    seq = TestSequence().create_sequence()

    def test_jitter(self):
        original_magnitudes = self.seq.mag
        new_magnitudes = jitter(self.seq, std=0, key="mag", enforce_bounds=True).mag
        assert torch.all(
            original_magnitudes == new_magnitudes
        ), "Jitter with std=0 should not change the magnitudes"

        new_magnitudes = jitter(self.seq, std=0.1, key="mag", enforce_bounds=True).mag
        assert torch.all(
            new_magnitudes != original_magnitudes
        ), "Jitter with std=0.1 should change the magnitudes"

        # the jitter should not drive magnitudes below the catalog completeness threshold
        # the dummy catalog's completeness threshold is 0
        assert torch.all(
            new_magnitudes >= 0
        ), "Jitter should not drive magnitudes below 0"

    def test_superimpose(self):
        
        other_seq = TestSequence().create_sequence()
        seq_bank = [other_seq]
        new_seq = superimpose(self.seq, seq_bank)

        assert (
            len(new_seq) == len(self.seq) + len(other_seq)
        ), "Superimpose should add the number of events of the sequences in the bank"
        assert (
            new_seq.t_start == self.seq.t_start
        ), "Superimpose should preserve the start time"
        assert (
            (new_seq.t_end - self.seq.t_end) < 1e-4
        ), "Superimpose should preserve the end time"
        assert (
            new_seq.t_nll_start == self.seq.t_nll_start
        ), "Superimpose should preserve the NLL start time"
        
        seq_bank = [TestSequence().create_sequence() for _ in range(10)]
        new_seq = superimpose(self.seq, seq_bank)
        assert len(new_seq) in [len(self.seq) + len(seq) for seq in seq_bank]
        assert new_seq.t_start is not None
        assert new_seq.t_end is not None
        assert new_seq.t_nll_start is not None
        for key in self.seq.keys():
            assert key in new_seq.keys(), f"Superimpose should preserve the key {key}"
            
    def test_sub_radius(self):
        new_seq = sub_radius(self.seq, fraction_range=[0.5, 1.0], radius_km=TestSequence.max_distance_km)
        assert len(new_seq) <= len(self.seq)
        assert new_seq.t_start == self.seq.t_start
        assert np.abs(new_seq.t_end - self.seq.t_end) < 1e-4
        assert new_seq.t_nll_start == self.seq.t_nll_start
        assert new_seq.distance_km.max() < TestSequence.max_distance_km
        for key in self.seq.keys():
            assert key in new_seq.keys(), f"Sub radius should preserve the key {key}"