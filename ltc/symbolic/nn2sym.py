import cloudpickle
import jax
import jax.numpy as jnp
import lz4.frame

from ltc.agents import QNetwork, StochasticVariationalNetwork


# from ltc.agents.svi import


class Target:

    def __init__(self,file_path: str):

        with lz4.frame.open(file_path, "rb") as f:
            self.ddqn_state, self.history = cloudpickle.load(f)

        self.model = StochasticVariationalNetwork(
            QNetwork(num_actions=3, num_layers=4, dim=64, num_heads=4)
        )

        def pred(params, state, obs, key):
            q_vals, _ = self.model.apply({'params': params, **state}, obs[jnp.newaxis,...], rngs=key, mutable=['loss'])
            acts = jnp.argmax(q_vals, axis=-1).astype(int)
            return q_vals[0]
        _pred = jax.vmap(pred, in_axes=(None, None, 0, 0))
        self.pred = jax.jit(_pred)



    def __call__(self):
        # n_epochs, n_steps, n_agents, window_size, n_features = history.observations.shape
        # features: buffer_state (0 or 1), channel_state (0 - idle, 1 - busy, -1 - unknown), retransmission_counter (N+), num_of_slots_with_no_tx (N+), power_left (0 - INITIAL_CAPACITY)

        params, state = self.ddqn_state.params, self.ddqn_state.net_state
        params_0, state_0 = jax.tree.map(
            lambda x: x[0], (params, state)
        )  # take the parameters and state of the first agent
        observations = self.history.observations[-1, :, 0]  # last epoch, all steps, first agent

        key = jax.random.key(42)

        keys = jax.random.split(key, num=observations.shape[0])

        # pred_fn = jax.vmap(self.pred, in_axes=(None, None, 0, 0))
        qvals  = self.pred(params_0, state_0, observations, keys)
        return observations,qvals
        # q_vals, _ = self.model.apply(
        #     {"params": params_0, **state_0},
        #     observations,
        #     rngs=jax.random.key(42),
        #     mutable=["loss"],
        # )
        # return observations, q_vals