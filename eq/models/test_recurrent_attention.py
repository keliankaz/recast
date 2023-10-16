#%%
from recurrent_attention import RecurrentTPP_Attention
from recurrent import RecurrentTPP
import eq
import numpy as np

t_start = 10.0
arrival_times = np.array([11.1, 11.5, 12.1, 14.2])
# If you don't know t_end, just set t_end = arrival_times[-1]
t_end = 20.0
mag = np.array([4.0, 2.1, 2.5, 2.9])
inter_times = np.diff(arrival_times, prepend=[t_start], append=[t_end])
seq = eq.data.Sequence(inter_times, t_start=0.0, mag=mag)
#%%
catalog = eq.catalogs.White(mag_completeness=4)
batch_list = []
for _ in range(3):
    seq = catalog.train[0] # eq.data.Batch.from_list([seq])
    batch_list.append(seq)
batch_multiple = eq.data.Batch.from_list(batch_list)

model = RecurrentTPP_Attention()
context = model.get_context(batch_multiple)
distr = model.get_inter_time_dist(context)
log_pdf = distr.log_prob(batch_multiple.inter_times.clamp_min(1e-10))
print("NO ERROR")
print("finish")
# %%
