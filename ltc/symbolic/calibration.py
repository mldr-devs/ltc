"""Scale calibration for the simplex probabilistic decoder.

The decoder of ext/simplex-probabilistic-decoding-spec.md,

    rho_j(x) = (T-1)/T * <f(x), c_j> + 1/T,

is exact only when ``f`` is the least-squares minimizer. A distilled expression is
neither exact nor unbiased in norm: PySR trades accuracy for complexity, and the
class-balanced weights of ``fit_sr`` pull the fit towards the weighted mean of the
codewords, which is ~0. Since ``f -> 0`` decodes to the uniform distribution, the
estimate comes out systematically too flat -- on the saturated run the distilled
expression has ||f|| ~ 0.22 against codewords of norm 1, and the replayed agent
transmits in ~48% of the steps where the teacher transmits in 8.6%.

``ScaleCalibrator`` is the multiclass counterpart of Platt scaling for this decoder:
a single ``a > 0`` fitted on held-out data by minimizing the NLL of

    rho_j(x) = (T-1)/T * a * <f(x), c_j> + 1/T,

projected back onto the simplex. Because ``a > 0`` is a monotone rescaling of every
score by the same factor, the argmax -- i.e. the deterministic policy -- is
unchanged; only the sampling distribution moves.

Numpy/scipy only, so the module can be used without JAX, and the fitted scalar is
carried into the agent through ``SimplexCode.scale``.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

# Below this the log in the NLL blows up; the same floor the sampler uses.
_EPS = 1e-12


def simplex_code_np(T: int) -> np.ndarray:
    """``ltc.symbolic.util.simplex_code`` without the JAX dependency, shape (T-1, T)."""
    assert T >= 2, "T must be >= 2"
    C = np.array([[1.0, -1.0]])
    for i in range(2, T):
        scale = np.sqrt(1.0 - 1.0 / i**2)
        top = np.concatenate([np.ones((1, 1)), np.full((1, i), -1.0 / i)], axis=1)
        bottom = np.concatenate([np.zeros((i - 1, 1)), scale * C], axis=1)
        C = np.concatenate([top, bottom], axis=0)
    return C


def decode_probs(f: np.ndarray, C: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Class probabilities of raw outputs ``f`` (n, T-1), clipped onto the simplex.

    The affine part sums to 1 identically (sum_y c_y = 0); the clip and the
    renormalization only matter for rows where ``f`` fell outside conv(C).
    """
    T = C.shape[1]
    rho = (T - 1) / T * scale * (np.asarray(f, dtype=np.float64) @ C) + 1 / T
    # Only the lower clip: rho sums to 1 identically, so at least one entry is
    # positive and the renormalization brings the rest below 1 anyway. Clipping
    # from above would flatten several classes onto 1 at large scales and hand the
    # argmax to whichever index comes first.
    rho = np.maximum(rho, 0.0)
    return rho / rho.sum(axis=1, keepdims=True)


@dataclass
class ScaleCalibrator:
    """One scalar ``a > 0`` sharpening the decoded probabilities. Sklearn-style.

    Parameters
    ----------
    T : int
        Number of classes.
    max_log_scale : float
        Search bound on ``|log a|``. The default admits scales in [1/e^6, e^6],
        far past the point where every row is already clipped to a vertex.

    Attributes
    ----------
    scale : float
        The fitted ``a``; 1.0 before ``fit``.
    clipped_fraction_ : float
        Share of validation rows whose decoded probabilities left the simplex at
        the fitted scale. Large values mean ``f`` is badly scaled, not just flat.
    """

    T: int
    max_log_scale: float = 6.0
    scale: float = 1.0
    clipped_fraction_: float = float("nan")

    def __post_init__(self) -> None:
        self.codes_ = simplex_code_np(self.T)

    def fit(self, f: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "ScaleCalibrator":
        """Minimize the weighted NLL over ``log a`` on validation outputs ``f``."""
        f = np.asarray(f, dtype=np.float64).reshape(len(y), self.T - 1)
        y = np.asarray(y, dtype=int)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        w = w / w.sum()
        rows = np.arange(len(y))

        def nll(log_a: float) -> float:
            p = decode_probs(f, self.codes_, float(np.exp(log_a)))
            return float(-np.sum(w * np.log(np.clip(p[rows, y], _EPS, None))))

        result = minimize_scalar(nll, bounds=(-self.max_log_scale, self.max_log_scale), method="bounded")
        self.scale = float(np.exp(result.x))

        rho = (self.T - 1) / self.T * self.scale * (f @ self.codes_) + 1 / self.T
        self.clipped_fraction_ = float(np.mean((rho < 0).any(axis=1) | (rho > 1).any(axis=1)))
        return self

    def predict_proba(self, f: np.ndarray) -> np.ndarray:
        return decode_probs(f, self.codes_, self.scale)

    def predict(self, f: np.ndarray) -> np.ndarray:
        """Argmax class. Independent of ``scale`` by construction."""
        return np.asarray(np.asarray(f).reshape(-1, self.T - 1) @ self.codes_).argmax(axis=1)
