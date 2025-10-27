#%%
import eq
import numpy as np
import torch
import pytest

# %%

# create two sequences with different inter-event times and magnitudes
times = [1,2,4]
start_time = 0
end_time = 10
mag = [1, 2, 3]
inter_times = np.diff(times, prepend=[start_time], append=[end_time])
seq1 = eq.data.Sequence(
    inter_times=torch.tensor(inter_times, dtype=torch.float32), 
    mag=torch.as_tensor(mag, dtype=torch.float32),
    t_start=start_time, 
    t_end=end_time,
    t_nll_start=start_time,
)

times = [1, 2, 4, 7]
mag = [4, 0, 6, -1]
start_time = 1
end_time = 20   
inter_times = np.diff(times, prepend=[start_time], append=[end_time])
seq2 = eq.data.Sequence(
    inter_times=torch.tensor(inter_times, dtype=torch.float32), 
    mag=torch.as_tensor(mag, dtype=torch.float32),
    t_start=start_time, 
    t_end=end_time,
    t_nll_start=start_time+2.5,
)

batch = eq.data.Batch.from_list([seq1, seq2])

class TestRecurrentTPP:
    
    model = eq.models.RecurrentTPP()
    
    def test_encode_time(self):
        dummy_inter_times = torch.tensor([1, 2, 3], dtype=torch.float32)
        assert not np.isnan(self.model.encode_time(dummy_inter_times)).any(), (
            "Encoded time contains NaN values, possibly due to log(0) or log(-1)"
        )
        assert not np.isinf(self.model.encode_time(dummy_inter_times)).any(), (
            "Encoded time contains Inf values, possibly due to log(0) or log(-1)"
        )
    
    def test_get_context(self):
        context = self.model.get_context(batch)
        
        # inter_time is number of events - 1 + 2 (start and end). 
        # -1 because we calculate intervals, 
        # +2 for intervals after start and before end 
        # -> number of intervals = number of events + 1
        assert context.shape == (
            2, max(len(seq1), len(seq2))+1, self.model.context_size), (  
            f"Context shape is {context.shape} but should be "
            f"(2, {max(len(seq1), len(seq2))}, {self.model.context_size})"
        )
        assert not torch.isnan(context).any(), "Context contains NaN values"
        assert not torch.isinf(context).any(), "Context contains Inf values"
        
    @pytest.mark.parametrize(
        "context", 
        [
            torch.rand(2,3,model.context_size), 
            model.get_context(batch)
        ]
    )
    def test_get_inter_time_dist(self, context):
        inter_time_dist = self.model.get_inter_time_dist(context)
        assert (inter_time_dist.mean > 0).all(), "Mean of inter-time distribution is not positive"
        assert (inter_time_dist.sample() > 0).all(), "Sample of inter-time distribution is not positive"
        tc = 5
        assert (inter_time_dist.sample_conditional(torch.zeros(*context.shape[:-1],1) + tc) > tc).all(), "Conditional sample of inter-time distribution is not positive"
        # note that all 0 or negative times get clamped such that 

    def test_nll_loss(self):
        loss = self.model.nll_loss(batch)
        assert loss.shape == (batch.batch_size,), "Loss shape is not (batch_size,)"
        assert not torch.isnan(loss).any(), "Loss contains NaN values"
        assert not torch.isinf(loss).any(), "Loss contains Inf values"
    
    @pytest.mark.parametrize(
        "batch_size, duration, past_seq",
        [
            (2, 10, None),
            (1, 10, seq1)
        ]
    )
    def test_sample(self, batch_size, duration, past_seq):
        
        sample = self.model.sample(batch_size=batch_size,duration=duration,past_seq=past_seq)
        assert sample.inter_times.shape[0] == batch_size
        assert (sample.inter_times[~sample.mask.bool()] != 0).all(), "Inter-event times are zero"
        assert (sample.inter_times[sample.mask.bool()] >= 0).all(), "Inter-event times are negative"
        
    @pytest.mark.parametrize("sequence", [seq1, seq2])
    def test_evaluate_intensity(self, sequence):
        grid, intensity = self.model.evaluate_intensity(sequence, num_grid_points=5)
        assert grid.shape == (5*sequence.inter_times.shape[0],)
        assert intensity.shape == (5*sequence.inter_times.shape[0],)
        assert not torch.isnan(intensity).any(), "Intensity contains NaN values"
        assert not torch.isinf(intensity).any(), "Intensity contains Inf values"
        assert intensity.min() >= 0, "Intensity is negative"
    
    @pytest.mark.parametrize("sequence", [seq1, seq2])
    def test_evaluate_compensator(self, sequence):
        grid, compensator = self.model.evaluate_compensator(sequence, num_grid_points=5)
        assert grid.shape == (5*sequence.inter_times.shape[0],)
        assert compensator.shape == (5*sequence.inter_times.shape[0],)
        assert not torch.isnan(compensator).any(), "Compensator contains NaN values"
        assert not torch.isinf(compensator).any(), "Compensator contains Inf values"
        assert compensator.min() >= 0, "Compensator is negative"
        assert compensator.diff().min() >= 0, "Compensator is not increasing"
        
        
        
#%%

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    model = eq.models.RecurrentTPP()
    dummy_context = torch.rand(2,3,model.context_size)
    t = np.linspace(0.001,50,20000)
    dt = t[1] - t[0]

    ll = []
    ls = []

    for it in t:
        dist = model.get_inter_time_dist(dummy_context)
        ll.append(dist.log_prob(torch.zeros(2,3)+it))
        ls.append(dist.log_survival(torch.zeros(2,3)+it))

    f = np.exp([il[0,0].item() for il in ll])
    s = np.exp([ls[0,0].item() for ls in ls])

    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(t,f)

    ax[1].plot(t, np.cumsum(f) * dt)
    ax[1].plot(t, 1-s)
    ax[1].axhline(1,color='r',ls='--')

    dist.sample()

    #%%
    sample = model.sample(2,10)
    inter_times = sample.inter_times[0]
    sample.mask

