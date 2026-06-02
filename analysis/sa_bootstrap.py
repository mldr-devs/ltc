"""Subprocess entry point for sensitivity analysis runs.

Patches ltc.sim.process_output module globals before running training,
since process_output uses `from ltc.sim.constants import *` which binds
constants directly into its namespace at import time. JAX traces against
that namespace, so patching ltc.sim.constants afterward has no effect.
"""
import os
import sys
import runpy

import ltc.sim.process_output as _po

_po.TX_REWARD = float(os.environ['SA_TX_REWARD'])
_po.EMPTY_BUFFER_REWARD = float(os.environ['SA_EMPTY_BUFFER_REWARD'])
_po.NO_TX_REWARD = float(os.environ['SA_NO_TX_REWARD'])
_po.NO_TX_PENALTY = float(os.environ['SA_NO_TX_PENALTY'])
_po.EMPTY_TX_PENALTY = float(os.environ['SA_EMPTY_TX_PENALTY'])
_po.COLLISION_PENALTY = float(os.environ['SA_COLLISION_PENALTY'])
_po.MAX_RETRANSMISSION_PENALTY = float(os.environ['SA_MAX_RETRANSMISSION_PENALTY'])
_po.SAFE_IDLE_PERIOD = max(1, round(float(os.environ['SA_SAFE_IDLE_PERIOD'])))
_po.PENALIZED_IDLE_PERIOD = max(1, round(float(os.environ['SA_PENALIZED_IDLE_PERIOD'])))

sys.argv = [
    'run.py',
    '--n_epochs', os.environ.get('SA_N_EPOCHS', '30'),
    '--n_steps', os.environ.get('SA_N_STEPS', '2000'),
    '--skip_git_check',
    '--seed', os.environ.get('SA_SEED', '42'),
]

run_py = os.path.join(os.path.dirname(__file__), '..', 'ltc', 'run.py')
runpy.run_path(run_py, run_name='__main__')
