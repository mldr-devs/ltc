"""
jax_random_forest.py

JIT-able JAX inference for a RandomForest already trained with scikit-learn.
No training happens here — this only converts a fitted
RandomForestClassifier / RandomForestRegressor into flat, padded JAX
arrays and re-implements the tree-traversal forward pass so it can be
jitted, vmapped, and run on GPU/TPU.

Usage
-----
    from sklearn.ensemble import RandomForestClassifier
    from jax_random_forest import JaxRandomForest

    rf = RandomForestClassifier(n_estimators=200, max_depth=12).fit(X_train, y_train)

    jrf = JaxRandomForest.from_sklearn(rf)
    probs = jrf.predict_proba(X_test)        # (n_samples, n_classes), jitted
    preds = jrf.predict(X_test)              # (n_samples,)

For a RandomForestRegressor, use jrf.predict(X_test) directly (predict_proba
is not defined for regressors).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


# --------------------------------------------------------------------------
# Conversion: sklearn fitted forest -> padded JAX arrays
# --------------------------------------------------------------------------

@dataclass
class JaxRandomForest:
    feature: jax.Array     # (n_trees, max_nodes)      int32, -2 marks a leaf
    threshold: jax.Array   # (n_trees, max_nodes)      float32
    left: jax.Array        # (n_trees, max_nodes)      int32 child index
    right: jax.Array       # (n_trees, max_nodes)      int32 child index
    value: jax.Array       # (n_trees, max_nodes, n_out) float32
    max_depth: int           # static: longest root-to-leaf path over all trees
    is_classifier: bool

    # ---- construction ----------------------------------------------------

    @classmethod
    def from_sklearn(cls, rf) -> "JaxRandomForest":
        trees = rf.estimators_
        n_trees = len(trees)
        max_nodes = max(t.tree_.node_count for t in trees)
        max_depth = max(t.tree_.max_depth for t in trees)
        is_classifier = hasattr(rf, "classes_")
        n_out = int(rf.n_classes_) if is_classifier else 1

        feature = np.full((n_trees, max_nodes), -2, dtype=np.int32)
        threshold = np.zeros((n_trees, max_nodes), dtype=np.float32)
        left = np.zeros((n_trees, max_nodes), dtype=np.int32)
        right = np.zeros((n_trees, max_nodes), dtype=np.int32)
        value = np.zeros((n_trees, max_nodes, n_out), dtype=np.float32)

        for i, est in enumerate(trees):
            t = est.tree_
            n = t.node_count
            feature[i, :n] = t.feature
            threshold[i, :n] = t.threshold
            left[i, :n] = t.children_left
            right[i, :n] = t.children_right

            if is_classifier:
                v = t.value[:, 0, :]                              # (n, n_classes) counts
                row_sums = v.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1, row_sums)   # guard padding rows
                value[i, :n] = v / row_sums
            else:
                value[i, :n, 0] = t.value[:, 0, 0]

        return cls(
            feature=jnp.asarray(feature),
            threshold=jnp.asarray(threshold),
            left=jnp.asarray(left),
            right=jnp.asarray(right),
            value=jnp.asarray(value),
            max_depth=int(max_depth),
            is_classifier=is_classifier,
        )

    # ---- inference ---------------------------------------------------------

    def predict_proba(self, X) -> jax.Array:
        if not self.is_classifier:
            raise ValueError("predict_proba is only defined for classifiers.")
        X = jnp.asarray(X, dtype=jnp.float32)
        return _forest_forward(
            self.feature, self.threshold, self.left, self.right, self.value,
            X, self.max_depth,
        )

    def predict(self, X) -> jax.Array:
        X = jnp.asarray(X, dtype=jnp.float32)
        out = _forest_forward(
            self.feature, self.threshold, self.left, self.right, self.value,
            X, self.max_depth,
        )
        if self.is_classifier:
            return jnp.argmax(out, axis=-1)
        return out[:, 0]


# --------------------------------------------------------------------------
# Core traversal, jit + vmap
# --------------------------------------------------------------------------

def _traverse_tree(feature, threshold, left, right, value, x, max_depth):
    """Walk one tree for one sample using a fixed number of loop iterations.

    Once a leaf is reached the loop keeps re-selecting the same node
    (jnp.where freezes it), so the total iteration count can stay static
    (= max_depth) for jit/fori_loop, regardless of the sample's actual path
    length.
    """

    def step(_, node):
        feat = feature[node]
        is_leaf = feat == -2
        safe_feat = jnp.where(is_leaf, 0, feat)      # avoid OOB gather on x
        go_left = x[safe_feat] <= threshold[node]
        next_child = jnp.where(go_left, left[node], right[node])
        return jnp.where(is_leaf, node, next_child)

    final_node = jax.lax.fori_loop(0, max_depth, step, 0)
    return value[final_node]  # (n_out,)


@partial(jax.jit, static_argnames=("max_depth",))
def _forest_forward(feature, threshold, left, right, value, X, max_depth):
    """
    feature, threshold, left, right : (n_trees, max_nodes)
    value                           : (n_trees, max_nodes, n_out)
    X                               : (n_samples, n_features)
    returns                         : (n_samples, n_out), averaged over trees
    """

    def per_tree_per_sample(f, th, l, r, v, x):
        return _traverse_tree(f, th, l, r, v, x, max_depth)

    # vmap over trees (axis 0 of tree arrays), fixed sample
    per_sample = jax.vmap(per_tree_per_sample, in_axes=(0, 0, 0, 0, 0, None))

    # vmap over samples (axis 0 of X), fixed trees
    all_preds = jax.vmap(per_sample, in_axes=(None, None, None, None, None, 0))(
        feature, threshold, left, right, value, X
    )  # (n_samples, n_trees, n_out)

    return all_preds.mean(axis=1)


# --------------------------------------------------------------------------
# Smoke test / example
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier  # noqa: I001
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    rf.fit(X_train, y_train)

    jrf = JaxRandomForest.from_sklearn(rf)

    sk_probs = rf.predict_proba(X_test)
    jax_probs = np.asarray(jrf.predict_proba(X_test))

    max_abs_diff = np.max(np.abs(sk_probs - jax_probs))
    print(f"max |sklearn - jax| prob difference: {max_abs_diff:.2e}")

    sk_preds = rf.predict(X_test)
    jax_preds = np.asarray(jrf.predict(X_test))
    agreement = (sk_preds == jax_preds).mean()
    print(f"prediction agreement: {agreement * 100:.2f}%")
