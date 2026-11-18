import numpy as np

DEFAULT_LAST_PERCENT = 0.2
TX_SLOTS = 20


def _flat(x):
    x = np.asarray(x)
    return x.reshape(-1, *x.shape[2:]) if x.ndim >= 3 else x


def per_agent_success(history, tx_slots: int = TX_SLOTS):
    """Delivered packets per mini-slot and station, scaled to packets per frame-time, shape ``(T, n)``."""
    return _flat(history.successful_packets).astype(float) * tx_slots


def steady_state_metrics(history, last_percent: float = DEFAULT_LAST_PERCENT, tx_slots: int = TX_SLOTS):
    """Aggregate throughput (packets per frame-time) and Jain's index of delivered packets over the tail."""
    success = per_agent_success(history, tx_slots)
    tail = success[int(len(success) * (1 - last_percent)):]
    per = tail.sum(axis=0)
    denom = len(per) * (per ** 2).sum()
    fairness = 0.0 if denom == 0 else per.sum() ** 2 / denom
    return float(tail.sum(axis=1).mean()), float(fairness)
