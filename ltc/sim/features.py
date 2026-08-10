import jax.numpy as jnp

from ltc.sim.constants import Actions, Features


def select_features(obs, features):
    return obs[..., tuple(int(f) for f in features)]


def raw_action(obs):
    action_tx = obs[..., int(Features.ACTION_TX)]
    action_cs = obs[..., int(Features.ACTION_CS)]
    return jnp.where(
        action_tx > 0, Actions.TX.value,
        jnp.where(action_cs > 0, Actions.CS.value, Actions.IDLE.value)
    )
