from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn


def add_batch_dim(x):
    return x[None, ...] if x.ndim == 2 else x


class QNetwork(nn.Module):
    rnn_dim: int = 16
    dense_dim: int = 32
    num_layers: int = 4
    num_actions: int = 2
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, s):
        dense = partial(nn.Dense, dtype=self.dtype)
        scan_lstm = nn.scan(nn.OptimizedLSTMCell, variable_broadcast='params', split_rngs={'params': False}, in_axes=1, out_axes=1)
        scan_lstm = partial(scan_lstm, self.rnn_dim, dtype=self.dtype)
        ln = partial(nn.LayerNorm, dtype=self.dtype)

        s = add_batch_dim(s)
        h = scan_lstm().initialize_carry(jax.random.key(0), s[:, 0].shape)

        for _ in range(self.num_layers):
            h, s = scan_lstm()(h, s)
            s = dense(self.dense_dim)(s)
            s = nn.gelu(s)

        s = ln()(s)
        s = s.reshape((s.shape[0], -1))
        s = dense(self.num_actions)(s)

        return s
