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
    return glob, joblib, np, os, pd, pickle, plt, sns


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
    _eqs = _model.equations_[["equation", "loss", "complexity","score"]].copy()

    # from PySR
    # threshold = 1.5 * _model.equations_["loss"].min()
    # filtered_equations = _model.equations_.query(f"loss <= {threshold}")
    # _best_idx = filtered_equations["score"].idxmax()

    _best_idx = _model.equations_["score"].idxmax()



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
def _(agents, base, df, np, plt, sns, sr_models):
    _feat_cols = [c for c in df.columns if c not in {"agent", "action"}]

    def _predict(model, X):
        best_idx = model.equations_["score"].idxmax()
        return model.predict(X.values, best_idx)

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


@app.cell
def _(mo):
    mo.md("""
    ## Statystyki klasyfikacji vs. baseline (najczęstsza klasa)
    """)
    return


@app.cell
def _(agents, df, np, sr_models):
    from sklearn.metrics import balanced_accuracy_score, f1_score

    _feat_cols = [c for c in df.columns if c not in {"agent", "action"}]

    def _predict_best(model, X):
        best_idx = model.equations_["score"].idxmax()
        return model.predict(X.values, best_idx)

    _n = len(agents)
    _acc2 = np.zeros((_n, _n))
    baseline_acc = np.zeros(_n)
    lift = np.zeros((_n, _n))
    bal_acc = np.zeros((_n, _n))
    f1 = np.zeros((_n, _n))

    for _j, _ag_j in enumerate(agents):
        _df_j = df[df["agent"] == _ag_j]
        _X_j = _df_j[_feat_cols].astype(np.float32)
        _y_j = _df_j["action"].values
        _counts = np.bincount(_y_j.astype(int))
        baseline_acc[_j] = _counts.max() / len(_y_j)

        for _i, _ag_i in enumerate(agents):
            _y_pred = (_predict_best(sr_models[_ag_i], _X_j) > 0).astype(int)
            _acc2[_i, _j] = (_y_pred == _y_j).mean()
            lift[_i, _j] = _acc2[_i, _j] - baseline_acc[_j]
            bal_acc[_i, _j] = balanced_accuracy_score(_y_j, _y_pred)
            f1[_i, _j] = f1_score(_y_j, _y_pred, zero_division=0)
    return bal_acc, baseline_acc, f1, lift


@app.cell
def _(agents, baseline_acc, plt, sns):
    _n = len(agents)
    fig_bl, ax_bl = plt.subplots(figsize=(9, 2))
    sns.heatmap(
        baseline_acc.reshape(1, _n),
        annot=True,
        fmt=".2f",
        ax=ax_bl,
        xticklabels=agents,
        yticklabels=["baseline"],
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    ax_bl.set_xlabel("Agent (źródło danych)")
    ax_bl.set_title("Baseline: accuracy najczęstszej klasy (per agent)")
    plt.tight_layout()
    fig_bl
    return


@app.cell
def _(agents, base, lift, plt, sns):
    _max_abs = max(float(abs(lift).max()), 0.01)
    fig_lift, ax_lift = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        lift,
        annot=True,
        fmt=".2f",
        ax=ax_lift,
        xticklabels=agents,
        yticklabels=agents,
        cmap="RdBu",
        center=0,
        vmin=-_max_abs,
        vmax=_max_abs,
    )
    ax_lift.set_xlabel("Test agent j  (źródło danych)")
    ax_lift.set_ylabel("Train agent i  (źródło modelu)")
    ax_lift.set_title(f"Lift nad baseline — SR (accuracy − baseline)\n{base}")
    plt.tight_layout()
    fig_lift
    return


@app.cell
def _(agents, base, baseline_acc, df, forest, np, plt, sns):
    _rf_feat_cols = [c for c in df.columns if c != "action"]
    _n = len(agents)
    _rf_lift = np.zeros(_n)
    for _j, _ag_j in enumerate(agents):
        _df_j = df[df["agent"] == _ag_j]
        _X_j = _df_j[_rf_feat_cols]
        _y_j = _df_j["action"].values
        _y_pred_rf = forest.predict(_X_j)
        _rf_lift[_j] = (_y_pred_rf == _y_j).mean() - baseline_acc[_j]

    _max_abs_rf = max(float(abs(_rf_lift).max()), 0.01)
    fig_rf_lift, ax_rf_lift = plt.subplots(figsize=(9, 2))
    sns.heatmap(
        _rf_lift.reshape(1, _n),
        annot=True,
        fmt=".2f",
        ax=ax_rf_lift,
        xticklabels=agents,
        yticklabels=["RF"],
        cmap="RdBu",
        center=0,
        vmin=-_max_abs_rf,
        vmax=_max_abs_rf,
    )
    ax_rf_lift.set_xlabel("Agent j (źródło danych)")
    ax_rf_lift.set_title(f"Lift nad baseline — Random Forest (accuracy − baseline)\n{base}")
    plt.tight_layout()
    fig_rf_lift
    return


@app.cell
def _(agents, bal_acc, base, plt, sns):
    fig_bal, ax_bal = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        bal_acc,
        annot=True,
        fmt=".2f",
        ax=ax_bal,
        xticklabels=agents,
        yticklabels=agents,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    ax_bal.set_xlabel("Test agent j  (źródło danych)")
    ax_bal.set_ylabel("Train agent i  (źródło modelu)")
    ax_bal.set_title(f"Balanced Accuracy\n{base}")
    plt.tight_layout()
    fig_bal
    return


@app.cell
def _(agents, base, f1, plt, sns):
    fig_f1, ax_f1 = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        f1,
        annot=True,
        fmt=".2f",
        ax=ax_f1,
        xticklabels=agents,
        yticklabels=agents,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    ax_f1.set_xlabel("Test agent j  (źródło danych)")
    ax_f1.set_ylabel("Train agent i  (źródło modelu)")
    ax_f1.set_title(f"F1 Score (klasa mniejszościowa)\n{base}")
    plt.tight_layout()
    fig_f1
    return


@app.cell
def _(mo):
    mo.md("""
    ## Macierze pomyłek — siatka (model i → dane j)
    """)
    return


@app.cell
def _(agents, base, df, np, plt, sns, sr_models):
    from sklearn.metrics import confusion_matrix as _confusion_matrix

    _feat_cols = [c for c in df.columns if c not in {"agent", "action"}]

    def _predict_best(model, X):
        best_idx = model.equations_["score"].idxmax()
        return model.predict(X.values, best_idx)

    _n = len(agents)
    _cell = 2.6
    fig_grid, axes_grid = plt.subplots(
        _n, _n, figsize=(_n * _cell, _n * _cell), squeeze=False
    )

    for _i, _ag_i in enumerate(agents):
        for _j, _ag_j in enumerate(agents):
            _ax = axes_grid[_i, _j]
            _df_j = df[df["agent"] == _ag_j]
            _X_j = _df_j[_feat_cols].astype(np.float32)
            _y_j = _df_j["action"].values
            _y_pred = (_predict_best(sr_models[_ag_i], _X_j) > 0).astype(int)
            _cm = _confusion_matrix(_y_j, _y_pred, normalize="true")
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
            _ax.tick_params(labelsize=6)
            _is_diag = _i == _j
            _ax.set_title(
                f"i={_ag_i} → j={_ag_j}",
                fontsize=7,
                fontweight="bold" if _is_diag else "normal",
                color="#1a7a1a" if _is_diag else "black",
            )
            if _i < _n - 1:
                _ax.set_xticklabels([])
            if _j > 0:
                _ax.set_yticklabels([])
            if _i == _n - 1:
                _ax.set_xlabel("pred", fontsize=6)
            if _j == 0:
                _ax.set_ylabel(f"true\n(i={_ag_i})", fontsize=6)

    for _j, _ag_j in enumerate(agents):
        axes_grid[0, _j].set_title(
            f"j={_ag_j}\ni={agents[0]}→j={_ag_j}" if _n == 1 else f"j={_ag_j}",
            fontsize=8,
            fontweight="bold",
            color="steelblue",
        )

    fig_grid.suptitle(
        f"Macierze pomyłek (znormalizowane wierszowo — recall per klasa)\n"
        f"wiersze=model agenta i, kolumny=dane agenta j   |   {base}",
        fontsize=10,
        fontweight="bold",
    )
    plt.tight_layout()
    fig_grid
    return


if __name__ == "__main__":
    app.run()
