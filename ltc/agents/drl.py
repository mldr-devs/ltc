import jax
import flax.linen as nn
import jax.numpy as jnp
from jax.scipy import stats

def add_batch_dim(x):
    return x[None, ...] if x.ndim == 2 else x


class QNetwork(nn.Module):
    rnn_dim: int = 32
    dense_dim: int = 64
    num_layers: int = 4
    num_actions: int = 2

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
    num_actions: int = 2
    rate:float = 0.2

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


class Uncertainty(nn.Module):
    prior_scale: float
    clip_min: float = 0.01
    scale_loc_ratio: float = 0.1

    @nn.compact
    def __call__(self, x):
        m = self.scale_loc_ratio * jnp.clip(jnp.abs(x), min=self.clip_min)
        log_scale = self.param('log_scale', lambda _: jnp.log(m))
        eps_key = self.make_rng('rlib')
        scale = jnp.exp(log_scale)
        posterior_sample = x + jax.random.normal(eps_key,
                                                 shape=x.shape) * scale
        kl = jnp.mean(stats.norm.logpdf(posterior_sample, loc=x, scale=scale) -
                      stats.norm.logpdf(posterior_sample, loc=0,
                                        scale=self.prior_scale), axis=0)
        loss_key = self.make_rng('rlib')
        loss = self.variable('loss', 'kl', nn.initializers.zeros, key=loss_key,
                             shape=(), dtype=x.dtype)
        loss.value = kl.sum()
        return posterior_sample

class StochasticVariationalQNetwork(nn.Module):
    rnn_dim: int = 32
    dense_dim: int = 64
    num_layers: int = 4
    num_actions: int = 2

    def setup(self):
        self.model = QNetwork(rnn_dim=self.rnn_dim, dense_dim=self.dense_dim,
                              num_layers=self.num_layers,
                              num_actions=self.num_actions)

    @nn.compact
    def __call__(self, x):
        if not 'params' in self.model.variables:
            _ = self.model(x)
        old_params = self.model.variables['params']

        def add_noise(kp, x):
            name = '/'.join((k.key for k in kp))
            return Uncertainty(name=name, prior_scale=2.0)(x)

        new_params = jax.tree.map_with_path(add_noise, old_params)
        vars = self.model.variables.copy()
        vars.update({'params': new_params})
        hat = self.model.clone(parent=None).apply(vars, x)
        return hat
