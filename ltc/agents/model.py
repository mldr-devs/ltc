import jax
import jax.numpy as jnp
from flax import linen as nn


def add_batch_dim(x):
    return x[None, ...] if x.ndim == 3 else x


class QNetwork(nn.Module):
    num_actions: int
    rnn_dim = 32
    fc_dim = 32

    @nn.compact
    def __call__(self, s):
        scan_gru = nn.scan(nn.GRUCell, variable_broadcast='params', split_rngs={'params': False}, in_axes=1, out_axes=1)
        h = scan_gru(self.rnn_dim).initialize_carry(jax.random.PRNGKey(0), s[:, 0].shape)

        _, s = scan_gru(self.rnn_dim)(h, s)
        s = nn.Dense(self.fc_dim)(s[:, -1])
        s = nn.relu(s)
        s = nn.Dense(self.num_actions)(s)

        return s


class MixingNetwork(nn.Module):
    num_actions: int
    fc_dim: int = 32

    @nn.compact
    def __call__(self, qs, g):
        b, n, _ = qs.shape
        _, g_feat = g.shape

        W1 = self.param('W1', nn.initializers.lecun_normal(), (self.fc_dim * n * self.num_actions, g_feat))
        b1 = self.param('b1', nn.initializers.lecun_normal(), (self.fc_dim, g_feat))
        W2 = self.param('W2', nn.initializers.lecun_normal(), ((n + 1) * self.fc_dim, g_feat))
        b2a = self.param('b2a', nn.initializers.lecun_normal(), (self.fc_dim, g_feat))
        b2b = self.param('b2b', nn.initializers.lecun_normal(), ((n + 1), self.fc_dim))

        W1s = jnp.abs(g @ W1.T).reshape(b, self.fc_dim, n * self.num_actions)
        b1s = g @ b1.T
        W2s = jnp.abs(g @ W2.T).reshape(b, n + 1, self.fc_dim)
        b2s = g @ b2a.T
        b2s = nn.relu(b2s)
        b2s = b2s @ b2b.T

        def fwd(q, W1, b1, W2, b2):
            s = q.flatten()
            s = jnp.dot(s, W1.T) + b1.T
            s = nn.elu(s)
            s = jnp.dot(s, W2.T) + b2.T
            return s

        s = jax.vmap(fwd)(qs, W1s, b1s, W2s, b2s)
        return s


class QLBTNetwork(nn.Module):
    num_actions: int
    rnn_dim: int = 32
    fc_dim: int = 32

    @nn.compact
    def __call__(self, s):
        BatchQNetwork = nn.vmap(
            QNetwork,
            in_axes=1, out_axes=1,
            variable_axes={'params': 0},
            split_rngs={'params': True}
        )

        s = add_batch_dim(s)
        ss = s[..., :-4]  # remove auxiliary features (raw d2lt, ret_c, buffer_state, and reward)
        actions, d2lt = s[..., -1, 0], s[..., -1, 5]
        d2lt = d2lt / jnp.maximum(d2lt.sum(axis=-1, keepdims=True), 1)
        g = jnp.concatenate([actions, d2lt], axis=-1)

        q_loc = BatchQNetwork(self.num_actions)(ss)
        q_mix = MixingNetwork(self.num_actions, self.fc_dim)(q_loc, g)
        q_loc = q_loc.reshape(s.shape[0], -1)
        return jnp.concatenate([q_mix, q_loc], axis=-1)
