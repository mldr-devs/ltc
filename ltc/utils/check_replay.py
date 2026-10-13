"""Fail a replay whose stations ended up colliding with each other forever.

A distilled policy is shared by every station, so one that transmits too eagerly
locks the network: all transmit, all collide, buffers never drain, throughput is
zero. It has happened twice here -- a class-balanced forest and a saturating
symbolic expression -- and both times the pipeline shipped the dead policy without
a word, the plots and the summary table being the first sign.

The collision share is the signature to test, not throughput: throughput is 0.07
on bursty traffic and 0.36 on saturated, so any threshold on it needs tuning per
experiment, while collisions separate cleanly either way. Measured over the tail
of the runs to hand: 0.056, 0.060 and 0.358 when healthy, 0.96 to 1.00 when
deadlocked.
"""

from argparse import ArgumentParser

import cloudpickle
import lz4.frame
import numpy as np

from ltc.utils.history import unpack_history
from ltc.utils.metrics import DEFAULT_LAST_PERCENT

COLLISION = -1


def collision_share(history, last_percent: float = DEFAULT_LAST_PERCENT) -> float:
    """Share of slots lost to collisions over the final ``last_percent``.

    The tail, not the whole run: a replay opens congested while the policy settles
    -- 0.65 over the first 500 steps of a healthy bursty forest -- which would mask
    the difference this is looking for.
    """
    channel = np.asarray(history.channel_state).reshape(-1)
    tail = channel[int(len(channel) * (1 - last_percent)):]
    return float((tail == COLLISION).mean()) if len(tail) else 0.0


if __name__ == '__main__':
    parser = ArgumentParser(description='Reject a replay that collided its way to a standstill.')
    parser.add_argument('--file', type=str, required=True, help='Replay history .pkl.lz4.')
    parser.add_argument('--max_collisions', type=float, default=0.9,
                        help='Fail above this share of collided slots in the measured tail. '
                             'The default sits between the 0.36 a busy healthy run reaches '
                             'and the 0.96 a deadlocked one starts at.')
    args = parser.parse_args()

    with lz4.frame.open(args.file, 'rb') as f:
        _, history, _ = unpack_history(cloudpickle.load(f))

    share = collision_share(history)
    if share > args.max_collisions:
        raise SystemExit(
            f'{args.file}: {share:.1%} of slots collided -- the distilled policy has '
            f'deadlocked the network. The history is kept for inspection.'
        )
    print(f'{args.file}: {share:.1%} of slots collided.')
