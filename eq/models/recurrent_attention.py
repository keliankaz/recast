from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import eq
import eq.distributions as dist

from eq.models.recurrent import RecurrentTPP


class RecurrentTPP_Attention(RecurrentTPP):
    """Neural TPP model with an recurrent encoder and a multihead
    attention. STILL UNDER DEVELOPMENT

    Args:
        input_magnitude: Should magnitude be used as model input?
        predict_magnitude: Should the model predict the magnitude?
        num_extra_features: Number of extra features to use as input.
        context_size: Size of the RNN hidden state.
        num_components: Number of mixture components in the output distribution.
        rnn_type: Type of the RNN. Possible choices {'GRU', 'RNN'}
        dropout_proba: Dropout probability.
        tau_mean: Mean inter-event times in the dataset.
        mag_mean: Mean earthquake magnitude in the dataset.
        richter_b: Fixed b value of the Gutenberg-Richter distribution for magnitudes.
        mag_completeness: Magnitude of completeness of the catalog.
        learning_rate: Learning rate used in optimization.
    """

    def __init__(
        self,
        input_magnitude: bool = True,
        predict_magnitude: bool = True,
        num_extra_features: Optional[int] = None,
        context_size: int = 32,
        num_components: int = 32,
        rnn_type: str = "GRU",
        dropout_proba: float = 0.5,
        tau_mean: float = 1.0,
        richter_b: float = 1.0,
        learning_rate: float = 5e-2,
    ):
        super().__init__(
            input_magnitude,
            predict_magnitude,
            num_extra_features,
            context_size,
            num_components,
            rnn_type,
            dropout_proba,
            tau_mean,
            richter_b,
            learning_rate
        )

        # num_heads = 4
        assert context_size % 4 == 0, f"context size={context_size} should be divisible by num_heads={4}"
        self.multihead_attn = nn.MultiheadAttention(embed_dim=context_size, num_heads=4, batch_first=True) # unsure about embed_dim
        self.layer_norm = nn.LayerNorm()

    def get_context(self, batch):
        """Get context embedding for each event in the batch of padded sequences.

        Returns:
            context: Context vectors, shape (batch_size, seq_len, context_size)
        """
        feat_list = [self.encode_time(batch.inter_times)]
        if self.input_magnitude:
            feat_list.append(self.encode_magnitude(batch.mag, batch.mag_bounds[:,0]))
        if self.num_extra_features is not None:
            feat_list.append(self.encode_extra_features(batch.extra_feat))
        features = torch.cat(feat_list, dim=-1)

        # rnn_output = self.rnn(features)[0][:, :-1, :]
        rnn_output = self.rnn(features)[0] # check size
        print(rnn_output.shape)
        print(self.context_size, batch.shape)
        raise Exception
        attention_output = self.multihead_attn(rnn_output, rnn_output, rnn_output)[0] #TODO should be (B, L, C), testing/reading about is_causal
        # see https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms#:~:text=The%20short%20answer%20is%20that,K%2C%20V%20are%20first%20introduced.
        # for a discussion of where to get Q, K, V values. It is possible that we should fetch V differently!
        assert len(attention_output.shape) == 3, f"attention_output's shape is {attention_output.shape}"
        truncated_attention_output = attention_output[:, :-1, :]
        output = F.pad(truncated_attention_output, (0, 0, 1, 0))  # (B, L, C)
        return self.dropout(output)  # (B, L, C)