import jax.numpy as jnp
from flax import linen as nn


class FeedForwardBlock(nn.Module):
    ff_dim: int
    dropout_rate: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, training=True):
        *_, out_dim = x.shape
        x = nn.Dense(self.ff_dim, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(out_dim, dtype=self.dtype)(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        return x


class TransformerBlock(nn.Module):
    num_heads: int
    ff_dim: int
    dropout_rate: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, mask, training=True):
        residual = x
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = nn.MultiHeadDotProductAttention(self.num_heads, qkv_features=x.shape[-1], dtype=self.dtype)(x, mask=mask)
        x = x + residual

        residual = x
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = FeedForwardBlock(self.ff_dim, self.dropout_rate, self.dtype)(x, training=training)
        x = x + residual

        return x


class Transformer(nn.Module):
    num_layers: int
    num_heads: int
    ff_dim: int
    dropout_rate: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, mask=None, training=True):
        for _ in range(self.num_layers):
            x = TransformerBlock(self.num_heads, self.ff_dim, self.dropout_rate, self.dtype)(x, mask, training=training)

        x = nn.LayerNorm(dtype=self.dtype)(x)
        return x


def add_batch_dim(x):
    return x[None, ...] if x.ndim == 2 else x


class QNetwork(nn.Module):
    num_actions: int
    num_layers: int
    dim: int
    num_heads: int
    dropout_rate: float = 0.25
    mc_dropout: bool = True
    dtype: jnp.dtype = 'float32'

    @nn.compact
    def __call__(self, s, training=True):
        x = add_batch_dim(s)
        b, t, _ = x.shape

        pos_embed = self.param('pos_embed', nn.initializers.xavier_uniform(), (1, t + 1, self.dim), self.dtype)
        cls_token = self.param('cls_token', nn.initializers.xavier_uniform(), (1, 1, self.dim), self.dtype)
        cls_tokens = jnp.tile(cls_token, (b, 1, 1))

        x = nn.Dense(self.dim, dtype=self.dtype)(x)
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x = x + pos_embed

        x = Transformer(self.num_layers, self.num_heads, 4 * self.dim, self.dropout_rate, self.dtype)(x, training=training)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training or self.mc_dropout)
        x = nn.Dense(self.num_actions, dtype=self.dtype)(x[:, 0])

        return x
