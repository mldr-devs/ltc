from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from ltc.sim.constants import Actions


def simplex_code(T: int) -> jax.Array:
    """
    Simplex code matrix for T classes, shape (T-1, T).

    Each column is a unit-norm codeword with pairwise inner product -1/(T-1).

    Algorithm 1 from: Multiclass Learning with Simplex Coding,
    Crammer & Singer (2002), Supplementary Material.

    C[2] = [[1, -1]]
    C[i+1] = | 1      u^T          |   u = (-1/i, ..., -1/i) in R^i
              | 0...0  sqrt(1-1/i²)·C[i] |
    """
    assert T >= 2, "T must be >= 2"
    C = jnp.array([[1.0, -1.0]])  # C[2], shape (1, 2)

    for i in range(2, T):
        scale = jnp.sqrt(1.0 - 1.0 / i**2)
        u = jnp.full((1, i), -1.0 / i)
        v = jnp.zeros((i - 1, 1))
        top = jnp.concatenate([jnp.ones((1, 1)), u], axis=1)
        bottom = jnp.concatenate([v, scale * C], axis=1)
        C = jnp.concatenate([top, bottom], axis=0)

    return C


@jax.tree_util.register_dataclass
@dataclass
class SimplexCode:
    """Simplex codewords plus the scale of the probabilistic decoder.

    ``scale`` is the calibration constant ``a`` of ltc.symbolic.calibration:
    a poorly fitted or heavily regularized ``f`` is shrunk towards 0, which the
    decoder reads as the uniform distribution, so ``a > 1`` re-sharpens it. It
    multiplies the scores only, so it never changes the argmax.
    """

    T: int = field(metadata=dict(static=True), default=len(Actions))
    codes: jax.Array | None = None
    scale: jax.Array | float = 1.0

    def __post_init__(self):
        if self.codes is None:
            self.codes = simplex_code(self.T)
        else:
            assert self.codes.shape == (self.T - 1, self.T), (
                "Codes must have shape (T-1, T)"
            )
            self.codes = self.codes

    @jax.jit
    def encode(self, labels: jax.Array) -> jax.Array:
        """
        Encode integer labels (shape (N,)) to simplex codes (shape (N, T-1)).
        """
        return jnp.take(self.codes, labels, axis=1).T

    @jax.jit
    def probs(self, codes: jax.Array) -> jax.Array:
        """
        Per-class conditional probabilities of simplex codes (shape (N, T-1)) -> shape (N, T).

        Supporting a stochastic choice.
        """
        fT =  jnp.asarray(self.T, dtype=codes.dtype)
        one = jnp.ones_like(fT)
        scale = jnp.asarray(self.scale, dtype=codes.dtype)
        rho = (fT-one)/fT * scale * (codes @ self.codes) + one/fT

        # Lower clip only: rho sums to 1 identically, so something is positive and
        # the renormalization keeps the rest below 1. An upper clip would flatten
        # several classes onto 1 at large scales and move the argmax.
        rho = jnp.clip(rho, a_min=0.0)
        rho = rho / jnp.sum(rho, axis=1, keepdims=True)
        return rho

    @jax.jit
    def decode(self, codes: jax.Array) -> jax.Array:
        """
        Decode simplex codes (shape (N, T-1)) to integer labels (shape (N,)).
        """
        return jnp.argmax(codes @ self.codes, axis=1)


@jax.jit
def history_reshape(observations: jax.Array) -> jax.Array:
    """
    Flatten a history of observations into a PySR-style feature matrix.

    [n_steps, n_agents, window_size, n_features] -> [n_agents * n_steps, window_size * n_features]

    Rows are grouped by agent (agent-major), and within a row the features are
    window-step-major, matching ``build_column_names`` in ``history2csv``.
    """
    n_steps, n_agents = observations.shape[0], observations.shape[1]
    return observations.transpose(1, 0, 2, 3).reshape(n_agents * n_steps, -1)


if __name__ == "__main__":
    T = 5
    sc = SimplexCode(T=T)
    for i in range(T):
        print(jnp.take(sc.codes, i, axis=1))

    labels = jnp.array([0, 1, 2, 3, 4])
    codes = sc.encode(labels)
    print("Codes:\n", codes)
    decoded_labels = sc.decode(codes)
    print("Decoded labels:\n", decoded_labels)
    assert jnp.array_equal(labels, decoded_labels), "Decoded labels do not match original"
