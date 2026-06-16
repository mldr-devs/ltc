import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import json
    import glob
    import pickle

    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.environ.setdefault("PYTHON_JULIAPKG_EXE", "/opt/homebrew/bin/julia")

    from ltc.symbolic.util import SimplexCode

    return SimplexCode, glob, joblib, json, np, os, pd, pickle, plt, sns


@app.cell
def _(glob, mo, os):
    _csv_files = sorted(glob.glob("out/*.csv"))
    _basenames = [os.path.splitext(os.path.basename(f))[0] for f in _csv_files]
    experiment = mo.ui.dropdown(
        options=_basenames,
        value=_basenames[0] if _basenames else None,
        label="Experiment",
    )
    experiment
    return (experiment,)


@app.cell
def _(SimplexCode, experiment, joblib, json, pd, pickle):
    base = experiment.value
    df = pd.read_csv(f"out/{base}.csv")

    with open(f"out/{base}.split.json") as _fh:
        _split = json.load(_fh)
    train_agents = _split["train_agents"]
    test_agents = _split["test_agents"]

    with open(f"out/{base}.split_sr.pkl", "rb") as _fh:
        sr_model = pickle.load(_fh)
    forest = joblib.load(f"out/{base}.split_forest.pkl")

    simplex_code = SimplexCode(T=df["action"].nunique())
    feat_cols = [c for c in df.columns if c not in {"agent", "action"}]
    return (
        base,
        df,
        feat_cols,
        forest,
        simplex_code,
        sr_model,
        test_agents,
        train_agents,
    )


@app.cell
def _(sr_model):
    sr_model
    return


@app.cell
def _(mo, sr_model):

    _best_idx = sr_model.equations_["score"].idxmax()

    _eqs = sr_model.equations_[["equation", "loss", "complexity","score"]].copy()

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
def _(mo, test_agents, train_agents):
    mo.md(f"""
    ## Split

    A single symbolic-regression model and a random-forest baseline are
    trained on the **train half** of the agents and evaluated on the
    **held-out half**.

    - **Train agents:** {train_agents}
    - **Held-out test agents:** {test_agents}
    """)
    return


@app.cell
def _(np, simplex_code):
    def predict_labels_sr(model, X):
        """Predict integer action labels by decoding the SR simplex-code output."""
        _eqs = model.equations_
        if isinstance(_eqs, list):
            _idx = [_e["score"].idxmax() for _e in _eqs]
        else:
            _idx = _eqs["score"].idxmax()
        _codes = np.asarray(model.predict(X.values, _idx))
        _codes = _codes.reshape(len(X), simplex_code.T - 1)
        return np.asarray(simplex_code.decode(_codes))

    return (predict_labels_sr,)


@app.cell
def _(df, feat_cols, forest, np, pd, predict_labels_sr, sr_model, test_agents):
    _rows = []
    for _ag in sorted(df["agent"].unique()):
        _df_ag = df[df["agent"] == _ag]
        _X = _df_ag[feat_cols].astype(np.float32)
        _y = _df_ag["action"].values
        _sr_pred = predict_labels_sr(sr_model, _X)
        _rf_pred = forest.predict(_X)
        _rows.append(
            {
                "agent": _ag,
                "split": "held-out" if _ag in test_agents else "train",
                "n": len(_y),
                "sr_acc": (_sr_pred == _y).mean(),
                "rf_acc": (_rf_pred == _y).mean(),
            }
        )
    acc_df = pd.DataFrame(_rows)
    return (acc_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Accuracy by agent (SR vs Random Forest)
    """)
    return


@app.cell
def _(acc_df, base, np, plt):
    _held = acc_df[acc_df["split"] == "held-out"]
    _x = np.arange(len(_held))
    _w = 0.38

    fig_acc, ax_acc = plt.subplots(figsize=(10, 4))
    ax_acc.bar(_x - _w / 2, _held["sr_acc"], _w, label="Symbolic regression")
    ax_acc.bar(_x + _w / 2, _held["rf_acc"], _w, label="Random forest")
    ax_acc.set_xticks(_x)
    ax_acc.set_xticklabels(_held["agent"])
    ax_acc.set_xlabel("Held-out agent (data source)")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_title(f"Held-out accuracy grouped by agent\n{base}")
    ax_acc.legend()
    ax_acc.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_acc
    return


@app.cell
def _(acc_df):
    acc_df.style.format({"sr_acc": "{:.4f}", "rf_acc": "{:.4f}"})
    return


@app.cell
def _(mo):
    mo.md("""
    ## Overall accuracy
    """)
    return


@app.cell
def _(acc_df, mo, np):
    def _overall(_sub):
        _w = _sub["n"].to_numpy()
        _sr = np.average(_sub["sr_acc"], weights=_w)
        _rf = np.average(_sub["rf_acc"], weights=_w)
        return _sr, _rf

    _held = acc_df[acc_df["split"] == "held-out"]
    _train = acc_df[acc_df["split"] == "train"]

    _h_sr, _h_rf = _overall(_held)
    _t_sr, _t_rf = _overall(_train)
    _a_sr, _a_rf = _overall(acc_df)

    mo.md(
        f"""
        Sample-weighted accuracy aggregated over agent groups:

        | Group | Symbolic regression | Random forest |
        |---|---|---|
        | Held-out agents | {_h_sr:.4f} | {_h_rf:.4f} |
        | Train agents (in-sample) | {_t_sr:.4f} | {_t_rf:.4f} |
        | **All agents** | **{_a_sr:.4f}** | **{_a_rf:.4f}** |
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Head-to-head: SR vs Random Forest (all agents)
    """)
    return


@app.cell
def _(acc_df, base, np, plt):
    _x = np.arange(len(acc_df))
    _w = 0.38
    _colors_split = ["#1a7a1a" if _s == "held-out" else "#999999" for _s in acc_df["split"]]

    fig_h2h, ax_h2h = plt.subplots(figsize=(12, 4))
    _b1 = ax_h2h.bar(_x - _w / 2, acc_df["sr_acc"], _w, label="Symbolic regression")
    _b2 = ax_h2h.bar(_x + _w / 2, acc_df["rf_acc"], _w, label="Random forest")
    ax_h2h.set_xticks(_x)
    ax_h2h.set_xticklabels(
        [f"{_a}\n({_s})" for _a, _s in zip(acc_df["agent"], acc_df["split"])]
    )
    ax_h2h.set_xlabel("Agent")
    ax_h2h.set_ylabel("Accuracy")
    ax_h2h.set_ylim(0, 1)
    ax_h2h.set_title(f"SR vs RF accuracy per agent\n{base}")
    ax_h2h.legend()
    ax_h2h.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_h2h
    return


@app.cell
def _(acc_df, base, np, plt, sns):
    _delta = (acc_df["sr_acc"] - acc_df["rf_acc"]).to_numpy().reshape(1, -1)
    _max_abs = max(float(np.abs(_delta).max()), 0.01)

    fig_delta, ax_delta = plt.subplots(figsize=(12, 1.8))
    sns.heatmap(
        _delta,
        annot=True,
        fmt=".3f",
        ax=ax_delta,
        xticklabels=[
            f"{_a} ({_s})" for _a, _s in zip(acc_df["agent"], acc_df["split"])
        ],
        yticklabels=["SR − RF"],
        cmap="RdBu",
        center=0,
        vmin=-_max_abs,
        vmax=_max_abs,
    )
    ax_delta.set_title(f"Accuracy advantage of SR over RF (positive = SR wins)\n{base}")
    plt.tight_layout()
    fig_delta
    return


@app.cell
def _(mo):
    mo.md("""
    ## Confusion matrices — held-out agents

    Row-normalised (recall per class). Top row: symbolic regression,
    bottom row: random forest.
    """)
    return


@app.cell
def _(
    base,
    df,
    feat_cols,
    forest,
    np,
    plt,
    predict_labels_sr,
    sns,
    sr_model,
    test_agents,
):
    from sklearn.metrics import confusion_matrix

    _n = len(test_agents)
    _cell = 2.8
    fig_cm, axes_cm = plt.subplots(
        2, _n, figsize=(_n * _cell, 2 * _cell), squeeze=False
    )

    for _col, _ag in enumerate(test_agents):
        _df_ag = df[df["agent"] == _ag]
        _X = _df_ag[feat_cols].astype(np.float32)
        _y = _df_ag["action"].values

        _sr_pred = predict_labels_sr(sr_model, _X)
        _rf_pred = forest.predict(_X)

        for _row, (_name, _pred) in enumerate(
            [("SR", _sr_pred), ("RF", _rf_pred)]
        ):
            _ax = axes_cm[_row, _col]
            _cm = confusion_matrix(_y, _pred, labels=[0, 1], normalize="true")
            sns.heatmap(
                _cm,
                annot=True,
                fmt=".2f",
                ax=_ax,
                cmap="Blues",
                cbar=False,
                xticklabels=["0", "1"],
                yticklabels=["0", "1"],
                vmin=0,
                vmax=1,
                linewidths=0.4,
                linecolor="gray",
            )
            _ax.set_title(f"{_name}  agent {_ag}", fontsize=8, fontweight="bold")
            _ax.tick_params(labelsize=7)
            if _col == 0:
                _ax.set_ylabel("true", fontsize=7)
            _ax.set_xlabel("pred", fontsize=7)

    fig_cm.suptitle(
        f"Confusion matrices on held-out agents   |   {base}",
        fontsize=10,
        fontweight="bold",
    )
    plt.tight_layout()
    fig_cm
    return (confusion_matrix,)


@app.cell
def _(mo):
    mo.md("""
    ## Aggregate confusion matrix — all held-out data
    """)
    return


@app.cell
def _(
    base,
    confusion_matrix,
    df,
    feat_cols,
    forest,
    np,
    plt,
    predict_labels_sr,
    sns,
    sr_model,
    test_agents,
):
    _df_held = df[df["agent"].isin(test_agents)]
    _X = _df_held[feat_cols].astype(np.float32)
    _y = _df_held["action"].values

    _sr_pred = predict_labels_sr(sr_model, _X)
    _rf_pred = forest.predict(_X)

    fig_cm_agg, axes_cm_agg = plt.subplots(1, 2, figsize=(5, 3.2))
    for _ax, _name, _pred in [
        (axes_cm_agg[0], "Symbolic regression", _sr_pred),
        (axes_cm_agg[1], "Random forest", _rf_pred),
    ]:
        _cm = confusion_matrix(_y, _pred, labels=[0, 1], normalize="true")
        sns.heatmap(
            _cm,
            annot=True,
            fmt=".3f",
            ax=_ax,
            cmap="Blues",
            cbar=False,
            xticklabels=["0", "1"],
            yticklabels=["0", "1"],
            vmin=0,
            vmax=1,
            linewidths=0.4,
            linecolor="gray",
        )
        _acc = (_pred == _y).mean()
        _ax.set_title(f"{_name}\nacc={_acc:.4f}", fontsize=9, fontweight="bold")
        _ax.set_xlabel("pred", fontsize=8)
        _ax.set_ylabel("true", fontsize=8)

    fig_cm_agg.suptitle(
        f"Aggregate over all held-out agents   |   {base}",
        fontsize=10,
        fontweight="bold",
    )
    plt.tight_layout()
    fig_cm_agg
    return


if __name__ == "__main__":
    app.run()
