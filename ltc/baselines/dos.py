import math

import jax
import jax.numpy as jnp
from chex import dataclass
from reinforced_lib.agents import BaseAgent, AgentState

from ltc.sim.constants import Actions, Features
from ltc.sim.features import select_features


@dataclass
class DOSState(AgentState):
    p: float            # channel access probability p_i
    e_hat: float        # filtered error signal Ê_p (output of the low-pass filter)
    idle_counter: int   # number of empty mini slots seen since the last transmission (O_p)


class DOS(BaseAgent):
    FEATURES = (Features.BUFFER, Features.CHANNEL)

    """
    Access-probability part of the ADOS mechanism from Garcia-Saavedra et al.,
    "Adaptive Mechanism for Distributed Opportunistic Scheduling" (arXiv:1412.4535).

    Only the p_i adaptation algorithm (§III-B, tuned per §IV-A) is implemented.
    We assume a transmission lasts a single mini slot, so the whole R_i / threshold
    machinery (channel probing, the R̄_i controller) is dropped and every station
    that wins a slot simply transmits. What remains is plain slotted-ALOHA access
    probability control: a proportional controller drives the channel-empty
    probability p_e towards the optimum 1/e.

    Control loop (all quantities in mini-slot units, τ = 1):
        interval n ends on each transmission (busy slot, success OR collision)
        O_p(n)     = number of empty mini slots since the previous transmission
        E_p(n+1)   = R_p - O_p(n)                              (eq. 7)
        Ê_p(n)     = α_p E_p(n) + (1 - α_p) Ê_p(n-1)           (low-pass filter F_p)
        t_i(n)     = K_{p,i} Ê_p(n)                            (proportional controller C_{p,i})
        p_i(n)     = 1 / t_i(n)
    """

    # --- fixed model parameters ---
    TAU = 1.0           # mini slot duration (work in slot units)
    T = 1.0             # data transmission lasts a single mini slot
    T_I = TAU           # time a station holds the channel on a success (one mini slot)

    # --- controller parameters ---
    ALPHA_P = 1e-4      # low-pass filter smoothing factor α_p
    G_P = 100.0         # target signal-to-noise gain G_p

    R_P = 1.0 / (math.e - 1.0)

    _DEN = T + math.e * TAU
    K_STABILITY = (2.0 - ALPHA_P) / (2.0 * ALPHA_P * _DEN)
    K_NOISE = (1.0 - ALPHA_P / 2.0) / (G_P * ALPHA_P * _DEN)
    K_P = min(K_NOISE, K_STABILITY)

    # Per-station gain K_{p,i} = K_p (T_i + (e-1)τ) so that the p_i's satisfy eq. (4).
    K_P_I = K_P * (T_I + (math.e - 1.0) * TAU)

    # --- access-probability clamping ---
    P_INIT = 1.0 / math.e
    P_MIN = 1e-4
    P_MAX = 1.0
    E_HAT_MIN = 1e-9    # floor on Ê_p so that t_i = K_{p,i} Ê_p stays strictly positive

    def __init__(self):
        self.init = jax.jit(self.init)
        self.update = jax.jit(self.update)
        self.sample = jax.jit(self.sample)

    @staticmethod
    def init(key):
        del key
        p = DOS.P_INIT
        # t_i = 1/p and t_i = K_{p,i} Ê_p  ->  Ê_p = 1 / (K_{p,i} p)
        e_hat = 1.0 / (DOS.K_P_I * p)
        return DOSState(p=p, e_hat=e_hat, idle_counter=0)

    @staticmethod
    def update(state, key, env_state, action, reward, terminal):
        del key, action, reward, terminal
        _, channel = select_features(env_state, DOS.FEATURES)[-1]

        def on_transmission():
            # A busy slot (success or collision) closes the current interval.
            o_p = state.idle_counter.astype(jnp.float32)      # O_p(n)
            e_p = DOS.R_P - o_p                               # E_p(n+1) = R_p - O_p(n)
            e_hat = DOS.ALPHA_P * e_p + (1.0 - DOS.ALPHA_P) * state.e_hat
            e_hat = jnp.maximum(e_hat, DOS.E_HAT_MIN)
            t_i = DOS.K_P_I * e_hat                           # proportional controller
            p = jnp.clip(1.0 / t_i, DOS.P_MIN, DOS.P_MAX)     # p_i = 1/t_i
            return DOSState(p=p, e_hat=e_hat, idle_counter=0)

        def on_idle():
            # Empty mini slot: just accumulate it into O_p.
            return DOSState(p=state.p, e_hat=state.e_hat,
                            idle_counter=state.idle_counter + 1)

        return jax.lax.cond(channel != 0, on_transmission, on_idle)

    @staticmethod
    def sample(state, key, env_state):
        buffer, _ = select_features(env_state, DOS.FEATURES)[-1]

        # Slotted-ALOHA access: contend for the slot with probability p_i.
        transmit = jax.random.uniform(key) < state.p
        return jnp.where(
            buffer == 0,
            Actions.IDLE.value,
            jnp.where(transmit, Actions.TX.value, Actions.CS.value),
        )
