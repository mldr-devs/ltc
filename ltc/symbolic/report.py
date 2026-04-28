import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import pickle
    import glob

    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sympy

    os.environ.setdefault("PYTHON_JULIAPKG_EXE", "/opt/homebrew/bin/julia")
    return glob, joblib, np, os, pd, pickle, plt, sns, sympy


@app.cell
def _(glob, mo, os):
    _csv_files = sorted(glob.glob("out/*.csv"))
    _basenames = [os.path.splitext(os.path.basename(f))[0] for f in _csv_files]
    experiment = mo.ui.dropdown(
        options=_basenames,
        value=_basenames[1] if _basenames else None,
        label="Experiment",
    )
    experiment
    return (experiment,)


@app.cell
def _(experiment, joblib, os, pd, pickle):
    base = experiment.value
    df = pd.read_csv(f"out/{base}.csv")
    forest = joblib.load(f"out/{base}.forest.pkl")
    agents = sorted(df["agent"].unique().tolist())
    sr_models = {}
    for _ag in agents:
        _path = f"out/{base}_agent_{_ag}.sr.pkl"
        if os.path.exists(_path):
            with open(_path, "rb") as _fh:
                sr_models[_ag] = pickle.load(_fh)
    return agents, base, df, forest, sr_models


@app.cell
def _(mo):
    mo.md("""
    ## Feature Importance
    """)
    return


@app.cell
def _(df, forest, plt):
    _feat_cols = [c for c in df.columns if c not in {"agent", "action"}]
    _imp = dict(zip(_feat_cols, forest.feature_importances_))
    _imp_series = sorted(_imp.items(), key=lambda x: x[1], reverse=True)[:15]
    _names, _vals = zip(*_imp_series)

    fig_fi, ax_fi = plt.subplots(figsize=(12, 4))
    ax_fi.bar(_names, _vals)
    ax_fi.set_xticklabels(_names, rotation=45, ha="right")
    ax_fi.set_ylabel("Importance")
    ax_fi.set_title("Random Forest Feature Importance (top 15)")
    plt.tight_layout()
    fig_fi
    return


@app.cell
def _(mo):
    mo.md("""
    ## Symbolic Equations
    """)
    return


@app.cell
def _(agents, mo):
    agent_sel = mo.ui.slider(
        min(agents), max(agents), value=agents[0], label="Agent"
    )
    agent_sel
    return (agent_sel,)


@app.cell
def _(agent_sel, mo, sr_models):
    _ag = agent_sel.value
    _model = sr_models[_ag]
    _eqs = _model.equations_[["equation", "loss", "complexity",]].copy()

    # from PySR
    threshold = 1.5 * _model.equations_["loss"].min()
    filtered_equations = _model.equations_.query(f"loss <= {threshold}")
    _best_idx = filtered_equations["score"].idxmax()



    _styled = _eqs.style.apply(
        lambda row: [
            "background-color: #90EE90; font-weight: bold" if row.name == _best_idx else ""
            for _ in row
        ],
        axis=1,
    ).format({"loss": "{:.6f}"})

    mo.as_html(_styled)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Cross-Agent Accuracy (corner plot)
    """)
    return


@app.cell
def _(agents, base, df, np, plt, sns, sr_models, sympy):
    _feat_cols = [c for c in df.columns if c not in {"agent", "action"}]

    def _predict(model, X):
        expr = model.sympy()
        free_syms = sorted(expr.free_symbols, key=lambda s: s.name)
        if not free_syms:
            return np.full(len(X), float(expr))
        f = sympy.lambdify(free_syms, expr, "numpy")
        return f(*[X[s.name].values for s in free_syms])

    n = len(agents)
    _acc = np.zeros((n, n))
    for _i, _ag_i in enumerate(agents):
        for _j, _ag_j in enumerate(agents):
            _df_j = df[df["agent"] == _ag_j]
            _X_j = _df_j[_feat_cols].astype(np.float32)
            _y_j = _df_j["action"].values
            _y_pred = (_predict(sr_models[_ag_i], _X_j) > 0).astype(int)
            _acc[_i, _j] = (_y_pred == _y_j).mean()

    fig_cp, ax_cp = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        _acc,
        annot=True,
        fmt=".2f",
        ax=ax_cp,
        xticklabels=agents,
        yticklabels=agents,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    ax_cp.set_xlabel("Test agent j  (data source)")
    ax_cp.set_ylabel("Train agent i  (model source)")
    ax_cp.set_title(f"Accuracy: equation of agent i evaluated on data of agent j \n{base}")
    plt.tight_layout()
    fig_cp
    return


if __name__ == "__main__":
    app.run()
