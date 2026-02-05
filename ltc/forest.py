import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestClassifier


def _debug(actions):
    unique_elements = [0, 1]

    # Use a list comprehension to count each and concat them into a new dimension
    counts_per_val = xr.concat(
        [(actions == v).sum(dim='sample') for v in unique_elements],
        dim='value')
    counts_per_val.coords['value'] = unique_elements
    counts_per_val.plot()
    plt.show()

    for i in range(40):
        u, c = np.unique(actions.sel(step=i, agent=3), return_counts=True)
        c = c / c.sum()
        print(np.dot(u, c))
        plt.stem(u, c)
        # print(qvals.sel(step=i,agent=0, sample=1))
        plt.show()


def main():
    ds = xr.load_dataset(
        '../explain/qnet_history_5_5_42.pkl.lz4sFalserTrueN8.nc')
    obs = ds['observations']
    qvals = ds['qvalues']
    actions = qvals.argmax('action')
    actions.mean(dim='step')
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    m = actions.mean(dim='step').to_pandas()
    table = ax.table(cellText=m.values, colLabels=m.columns, loc='center')
    plt.savefig('mean_action_along_step.pdf')
    plt.show()

    y = actions.sel(agent=3, sample=1)
    X = obs.sel(agent=3)
    X = X.stack(X=('window', 'feature'))
    flattened_coords = [f"{w}_{f}" for w, f in X.X.values]
    X = X.drop_vars(['X', 'window', 'feature']).assign_coords(
        X=flattened_coords)
    Xdf = X.to_pandas()

    forest = RandomForestClassifier(n_estimators=500, oob_score=True)
    forest.fit(Xdf, y)
    print(forest.oob_score_)
    importances = pd.DataFrame(forest.feature_importances_, index=Xdf.columns)
    importances.plot(kind='bar')
    plt.title(f'Feature importance, OOB score: {forest.oob_score_}')
    plt.tight_layout()
    plt.savefig("forest_importance_cherrypick.png")
    plt.show()

    y_all = actions.stack(flat=('agent', 'step', 'sample'))
    X_all = obs.expand_dims(sample=actions.sample).stack(
        flat=('agent', 'step', 'sample')).transpose('flat', 'window',
                                                    'feature').stack(
        y=('window', 'feature'))

    flat_coords = X_all.flat
    agent_ids = xr.DataArray([c[0] for c in flat_coords.values], dims='flat')
    sample_ids = xr.DataArray([c[2] for c in flat_coords.values], dims='flat')

    X_all_df = X_all.to_pandas()
    X_all_df['agent_id'] = agent_ids.values
    X_all_df['sample_id'] = sample_ids.values

    forest = RandomForestClassifier(n_estimators=500, oob_score=True)
    forest.fit(X_all_df, y_all)
    print(forest.oob_score_)
    importances = pd.DataFrame(forest.feature_importances_, index=X_all_df.columns)
    importances.plot(kind='bar')
    plt.title(f'Feature importance, OOB score: {forest.oob_score_}')
    plt.tight_layout()
    plt.savefig("forest_importance_agent_seed.png")
    plt.show()
    ...


if __name__ == '__main__':
    main()
