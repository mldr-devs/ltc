import jax
import jax.numpy as jnp

# @jax.jit
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
        scale = jnp.sqrt(1.0 - 1.0 / i ** 2)
        u = jnp.full((1, i), -1.0 / i)
        v = jnp.zeros((i - 1, 1))
        top = jnp.concatenate([jnp.ones((1, 1)), u], axis=1)
        bottom = jnp.concatenate([v, scale * C], axis=1)
        C = jnp.concatenate([top, bottom], axis=0)

    return C

if __name__ == "__main__":
    T=5
    sc = simplex_code(T=T)
    for i in range(T):
        print(jnp.take(sc,i,axis=1))

    ...