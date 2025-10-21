import jax
from flax import linen as nn


def add_batch_dim(x):
    return x[None, ...] if x.ndim == 2 else x


class QNetwork(nn.Module):
    rnn_dim: int = 32
    dense_dim: int = 64
    num_layers: int = 4
    num_actions: int = 3

    @nn.compact
    def __call__(self, s):
        scan_lstm = nn.scan(nn.OptimizedLSTMCell, variable_broadcast='params', split_rngs={'params': False}, in_axes=1, out_axes=1)

        s = add_batch_dim(s)
        h = scan_lstm(self.rnn_dim).initialize_carry(jax.random.key(0), s[:, 0].shape)

        for _ in range(self.num_layers):
            h, s = scan_lstm(self.rnn_dim)(h, s)
            s = nn.Dense(self.dense_dim)(s)
            s = nn.gelu(s)

        s = nn.LayerNorm()(s)
        s = s.reshape((s.shape[0], -1))
        s = nn.Dense(self.num_actions)(s)

        return s


class QNetworkDropout(nn.Module):
    rnn_dim: int = 32
    dense_dim: int = 64
    num_layers: int = 4
    num_actions: int = 3
    rate: float = 0.2

    @nn.compact
    def __call__(self, s):
        scan_lstm = nn.scan(nn.OptimizedLSTMCell, variable_broadcast='params', split_rngs={'params': False}, in_axes=1, out_axes=1)

        s = add_batch_dim(s)
        h = scan_lstm(self.rnn_dim).initialize_carry(jax.random.key(0), s[:, 0].shape)

        for _ in range(self.num_layers):
            h, s = scan_lstm(self.rnn_dim)(h, s)
            s = nn.Dense(self.dense_dim)(s)
            s = nn.gelu(s)

        s = nn.LayerNorm()(s)
        s = s.reshape((s.shape[0], -1))
        s = nn.Dropout(rate=self.rate, deterministic=False)(s)
        s = nn.Dense(self.num_actions)(s)

        return s
