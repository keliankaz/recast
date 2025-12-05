# %%
import eq
import numpy as np
import torch


# #%%
class TestSequence:

    num_events = 5
    start_time = 0
    end_time = 100
    mag_mean = 1
    max_distance_km = 1000

    @staticmethod
    def create_sequence(
        num_events: int = 5,
        start_time: float = 0,
        end_time: float = 100,
        mag_mean: float = 1,
        max_distance_km: float = 1000,
    ):
        time = np.sort(np.random.uniform(start_time, end_time, num_events))
        inter_times = np.diff(time, prepend=[start_time], append=[end_time])
        mag = np.random.exponential(mag_mean, num_events)
        
        mag_bounds = [0, 10]
        mag_nll_bounds = [0, 10]
        mag = eq.data.ContinuousMarks(
            values=torch.tensor(mag, dtype=torch.float32),
            bounds=torch.tensor(mag_bounds, dtype=torch.float32),
            nll_bounds=torch.tensor(mag_nll_bounds, dtype=torch.float32),
        )
        distance_km = np.random.uniform(0, max_distance_km, num_events)
        
        seq = eq.data.Sequence(
            inter_times=torch.tensor(inter_times, dtype=torch.float32),
            mag=mag,
            t_start=start_time,
            t_end=end_time,
            distance_km=torch.tensor(distance_km, dtype=torch.float32),
        )

        return seq

    def test_init(self):
        seq = self.create_sequence(
            self.num_events, self.start_time, self.end_time, self.mag_mean
        )
        assert seq.inter_times.shape == (self.num_events + 1,)
        assert seq.mag.shape == (self.num_events,)
        assert seq.t_start == self.start_time, "t_start is not start_time"
        assert seq.t_end == self.end_time, "t_end is not end_time"

    def test_get_subsequence(self):
        seq = self.create_sequence(
            self.num_events, self.start_time, self.end_time, self.mag_mean
        )
        start = 10
        end = 20
        subseq = seq.get_subsequence(start, end)
        assert subseq.t_start == start, "t_start is not start"
        assert subseq.t_end == end, "t_end is not end"

    def test_continuous_marks(self):
        marks = eq.data.ContinuousMarks(
            values=torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32),
            bounds=torch.tensor([0, 10], dtype=torch.float32),
            nll_bounds=torch.tensor([2, 8], dtype=torch.float32),
        )
        seq = eq.data.Sequence(
            inter_times=torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32),
            t_start=0,
            mag=marks,
        )

        assert seq.mag_bounds.shape == (2,)
        assert seq.mag_nll_bounds.shape == (2,)
        assert seq.mag_nll_bounds[0] == 2
        assert seq.mag_nll_bounds[1] == 8
        
    def test_subsequence_by_index(self):
        seq = self.create_sequence(
            self.num_events, self.start_time, self.end_time, self.mag_mean, self.max_distance_km
        )
        
        for fraction in [0.99, 0.5, 0.01]:
            indices = seq.distance_km < self.max_distance_km*fraction
            subseq = seq.subsequence_by_index(indices)
            assert len(subseq) == indices.sum().item(), "subsequence length does not match the number of events in the subset"
            assert subseq.t_start == self.start_time, "t_start is not start"
            assert np.abs(subseq.t_end - self.end_time) < 1e-4, "t_end is not end"
            assert subseq.distance_km.max() if len(subseq) > 0 else 0 < self.max_distance_km*fraction, "distance_km is greater than the max distance"
        
        indices = np.random.randint(0, self.num_events-1)
        subseq = seq.subsequence_by_index(indices)
        assert len(subseq) == 1, "subsequence length is not 1"
        assert subseq.t_start == self.start_time, "t_start is not start"
        assert subseq.t_end == self.end_time, "t_end is not end"
        assert len(subseq.inter_times) == 2, "For single event subsequence, inter_times length should be 2"
        

class TestBatch:

    def test_from_list(self):
        start_time = 0
        end_time = 100
        num_sequences = 3
        max_num_events = 10
        sequences = []
        for _ in range(num_sequences):
            num_events = np.random.randint(1, max_num_events)
            time = np.sort(np.random.uniform(start_time, end_time, num_events))
            inter_times = np.diff(time, prepend=[start_time], append=[end_time])
            mag = np.random.exponential(1, num_events)
            seq = eq.data.Sequence(
                inter_times=torch.tensor(inter_times, dtype=torch.float32),
                mag=torch.tensor(mag, dtype=torch.float32),
                t_start=start_time,
                t_end=end_time,
                t_nll_start=start_time + (end_time - start_time) * np.random.rand(),
            )
            sequences.append(seq)

        batch = eq.data.Batch.from_list(sequences)
        assert batch.batch_size == num_sequences, "batch_size is not num_sequences"
        assert batch.seq_len <= max_num_events, "seq_len is greater than max_num_events"
        assert batch.t_start.shape == (
            num_sequences,
        ), "t_start shape is not (num_sequences,)"
        assert batch.t_end.shape == (
            num_sequences,
        ), "t_end shape is not (num_sequences,)"
        assert batch.t_nll_start.shape == (
            num_sequences,
        ), "t_nll_start shape is not (num_sequences,)"

        assert torch.all(batch.inter_times >= 0), "Inter-event times are negative"
        assert torch.all(
            batch.inter_times[batch.mask.bool()] > 0
        ), "Inter-event times are zero"

    def test_batch_bounds(self):
        start_time = 0
        end_time = 100
        num_sequences = 3
        max_num_events = 10
        sequences = []
        for _ in range(num_sequences):
            num_events = np.random.randint(1, max_num_events)
            time = np.sort(np.random.uniform(start_time, end_time, num_events))
            inter_times = np.diff(time, prepend=[start_time], append=[end_time])
            mag = np.random.exponential(1, num_events)
            marks = eq.data.ContinuousMarks(
                values=torch.tensor(mag, dtype=torch.float32),
                bounds=torch.tensor([np.random.uniform(0, 1), 10], dtype=torch.float32),
            )
            sequences.append(
                eq.data.Sequence(
                    inter_times=torch.tensor(inter_times, dtype=torch.float32),
                    mag=marks,
                    t_start=start_time,
                    t_end=end_time,
                    t_nll_start=start_time + (end_time - start_time) * np.random.rand(),
                )
            )
        batch = eq.data.Batch.from_list(sequences)
        assert batch.mag_bounds.shape == (num_sequences, 2)
        assert batch.mag_nll_bounds.shape == (num_sequences, 2)
        assert torch.all(batch.mag_bounds[:, 0] <= batch.mag_bounds[:, 1])
        assert torch.all(batch.mag_nll_bounds[:, 0] <= batch.mag_nll_bounds[:, 1])
        assert torch.all(batch.mag_bounds[:, 0] <= batch.mag_nll_bounds[:, 0])
        assert torch.all(batch.mag_bounds[:, 1] >= batch.mag_nll_bounds[:, 1])


# %%
if __name__ == "__main__":
    # scratch pad
    start_time = 0
    end_time = 100
    num_sequences = 3
    max_num_events = 10
    sequences = []
    for _ in range(num_sequences):
        num_events = np.random.randint(1, max_num_events)
        time = np.sort(np.random.uniform(start_time, end_time, num_events))
        inter_times = np.diff(time, prepend=[start_time], append=[end_time])
        mag = np.random.exponential(1, num_events)
        seq = eq.data.Sequence(
            inter_times=torch.tensor(inter_times, dtype=torch.float32),
            mag=torch.tensor(mag, dtype=torch.float32),
            t_start=start_time,
            t_end=end_time,
            t_nll_start=start_time + (end_time - start_time) * np.random.rand(),
        )
        sequences.append(seq)
    batch = eq.data.Batch.from_list(sequences)
    print(batch)
# %%
