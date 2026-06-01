import jax.numpy as jnp
from flax import linen as nn


class QNetwork(nn.Module):
    num_actions: int
    dim: int
    num_layers: int

    @nn.compact
    def __call__(self, s, training=True):
        x = s[None, ...] if s.ndim == 2 else s             # (b, t, f)
        x = x.reshape(x.shape[0], -1).astype(jnp.float32)  # (b, t * f)

        for _ in range(self.num_layers):
            x = nn.Dense(self.dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)

        return nn.Dense(self.num_actions)(x)
