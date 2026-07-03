import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'

import argparse
from functools import partial

import cloudpickle
import jax
import jax.numpy as jnp
import lz4.frame
import optax
from tqdm import trange

from ltc.agents import BayesianDDQN, DCF, QNetwork, StochasticVariationalNetwork, QALOHA, EBALOHA, FWALOHA, TDMA
from ltc.sim import InitialStateConf, cox_traffic, process_output, simulate
from ltc.sim.constants import INITIAL_CAPACITY, Actions
from ltc.utils.history import agents_slug, build_history_filename, ensure_clean_git_worktree, get_short_commit_hash
from ltc.utils.scan_states import Carry, Output
from ltc.utils.plots import plot_all, plot_first


def parse_agent_groups(specs: list[str]) -> list[tuple[str, int, float | None]]:
    """
    Parse agent specs of the form 'type:count' or 'type:count:param'.
    DRL must be first if present. Returns list of (type, count, param) tuples.
    """
    groups = []
    for spec in specs:
        parts = spec.split(':')
        if len(parts) < 2:
            raise ValueError(f'Invalid agent spec: {spec!r}. Expected type:count or type:count:param')
        atype = parts[0]
        count = int(parts[1])
        param = float(parts[2]) if len(parts) > 2 else None
        groups.append((atype, count, param))

    drl_indices = [i for i, (t, _, _) in enumerate(groups) if t == 'drl']
    if len(drl_indices) > 1:
        raise ValueError('At most one DRL group is allowed in --agents')
    if drl_indices and drl_indices[0] != 0:
        raise ValueError('The DRL group must be first in --agents')

    return groups


def make_legacy_agent(atype: str, param: float | None):
    if atype == 'q-aloha':
        return QALOHA(q=param if param is not None else 0.1)
    elif atype == 'eb-aloha':
        return EBALOHA(window_size=4, max_backoff=2)
    elif atype == 'fw-aloha':
        return FWALOHA(window_size=4)
    elif atype == 'tdma':
        return TDMA(state_size=10, assigned_slots=5)
    else:
        raise ValueError(f'Unknown legacy agent type: {atype!r}')


def init_agents(agent, key, n):
    keys = jax.random.split(key, n)
    states = jax.vmap(agent.init)(keys)
    step_fn = jax.vmap(partial(agent_step, agent))
    return states, step_fn


def agent_step(agent, state, key, obs, action, reward, terminal, active):
    update_key, sample_key = jax.random.split(key)

    def power_on(state, update_key, sample_key, obs, action, reward, terminal):
        state = agent.update(state, update_key, obs, action, reward, terminal)
        action = agent.sample(state, sample_key, obs)
        return state, action

    def power_off(state, update_key, sample_key, obs, action, reward, terminal):
        return state, Actions.IDLE.value

    return jax.lax.cond(
        jnp.logical_or(~active, terminal), power_off, power_on,
        state, update_key, sample_key, obs, action, reward, terminal
    )


def init_traffic(traffic, key, n):
    keys = jax.random.split(key, n)
    states = jax.vmap(traffic.init)(keys)
    step_fn = jax.vmap(traffic.sample)
    return states, step_fn


def schedule_active_stations(
    active,
    step,
    *,
    one_shot_step=None,
    one_shot_target=None,
    station_change_interval=None,
    station_change_delta=0,
    station_change_start_step=0,
    station_change_stop_step=None,
    station_change_target=None,
):
    n = active.shape[0]
    next_active = active

    if one_shot_step is not None and one_shot_target is not None:
        target = jnp.asarray(one_shot_target, dtype=jnp.int32)
        one_shot_active = jnp.arange(n) < target
        next_active = jnp.where(step == one_shot_step, one_shot_active, next_active)

    if station_change_interval is not None and station_change_delta != 0:
        should_change = step >= station_change_start_step
        if station_change_stop_step is not None:
            should_change = jnp.logical_and(should_change, step <= station_change_stop_step)
        should_change = jnp.logical_and(
            should_change, (step - station_change_start_step) % station_change_interval == 0
        )

        current_count = jnp.sum(next_active.astype(jnp.int32))
        updated_count = jnp.clip(current_count + station_change_delta, 0, n)
        if station_change_target is not None:
            target = jnp.asarray(station_change_target, dtype=jnp.int32)
            updated_count = jnp.where(
                station_change_delta > 0,
                jnp.minimum(updated_count, target),
                jnp.maximum(updated_count, target),
            )

        changed_active = jnp.arange(n) < updated_count
        next_active = jnp.where(should_change, changed_active, next_active)

    return next_active


def rl_step(
    drl_step, legacy_steps, traffic_step, n, n_drl, phy_error_prob, n_bins=50,
    one_shot_step=None, one_shot_target=None, station_change_interval=None, station_change_delta=0,
    station_change_start_step=0, station_change_stop_step=None, station_change_target=None,
):
    """
    legacy_steps: list of (step_fn, count) pairs, one per legacy agent group.
    """
    def rl_step_coroutine(c, step):
        active = schedule_active_stations(
            c.active,
            step,
            one_shot_step=one_shot_step,
            one_shot_target=one_shot_target,
            station_change_interval=station_change_interval,
            station_change_delta=station_change_delta,
            station_change_start_step=station_change_start_step,
            station_change_stop_step=station_change_stop_step,
            station_change_target=station_change_target,
        )

        key, drl_key_raw, legacy_key_raw, traffic_key, reward_key, sim_key = jax.random.split(c.key, 6)
        n_legacy = n - n_drl
        traffic_keys = jax.random.split(traffic_key, n)

        # DRL step (skipped if n_drl == 0)
        if n_drl > 0:
            drl_keys = jax.random.split(drl_key_raw, n_drl)
            drl_intermediate = drl_step(
                c.drl_states, drl_keys,
                c.obs[:n_drl], c.actions[:n_drl], c.rewards[:n_drl], c.terminals[:n_drl], active[:n_drl]
            )
        else:
            drl_intermediate = (c.drl_states, jnp.zeros(0, dtype=int))
        drl_states, drl_actions = yield drl_intermediate

        # Legacy steps — one call per group
        all_legacy_keys = jax.random.split(legacy_key_raw, n_legacy) if n_legacy > 0 else jnp.zeros((0, 2), dtype=jnp.uint32)
        new_legacy_states = []
        all_legacy_actions = []
        lkey_offset = 0
        sta_offset = n_drl
        for i, (lstep, lcount) in enumerate(legacy_steps):
            lkeys = all_legacy_keys[lkey_offset:lkey_offset + lcount]
            lst, lact = lstep(
                c.legacy_states[i], lkeys,
                c.obs[sta_offset:sta_offset + lcount],
                c.actions[sta_offset:sta_offset + lcount],
                c.rewards[sta_offset:sta_offset + lcount],
                c.terminals[sta_offset:sta_offset + lcount],
                active[sta_offset:sta_offset + lcount],
            )
            new_legacy_states.append(lst)
            all_legacy_actions.append(lact)
            lkey_offset += lcount
            sta_offset += lcount

        legacy_states = tuple(new_legacy_states)
        actions = jnp.concatenate([drl_actions] + all_legacy_actions)

        traffic_states, new_frames = traffic_step(c.traffic_states, traffic_keys)
        buffer_states, channel_state, phy_error = simulate(c.buffer_states, new_frames, actions, sim_key, error_probability=phy_error_prob)
        obs, rewards, powers = process_output(
            c.buffer_states, buffer_states, c.power_states, channel_state, c.obs, actions, c.terminals, reward_key
        )
        terminals = jnp.logical_or(c.terminals, powers < 0)

        if n_drl > 0:
            params = c.drl_states.params['model'] if 'model' in c.drl_states.params else c.drl_states.params
            flat_params, _ = jax.tree.flatten(params)
            flat_params = jax.tree.map(lambda x: x.reshape(n_drl, -1), flat_params)
            flat_params = jnp.hstack(flat_params)
            hist, bin_edges = jax.vmap(jnp.histogram, in_axes=(0, None))(flat_params, n_bins)
        else:
            hist, bin_edges = None, None

        c = Carry(
            drl_states, legacy_states, traffic_states, buffer_states, powers,
            channel_state, key, obs, actions, rewards, terminals, active
        )
        o = Output(
            legacy_states, obs, actions, rewards, terminals, buffer_states, powers,
            (new_frames > 0).astype(int), channel_state, active, hist, bin_edges, phy_error
        )
        yield c, o

    def rl_step_fn(*args, **kwargs):
        gen = rl_step_coroutine(*args, **kwargs)
        intercepted = next(gen)
        return gen.send(intercepted)

    def pre_rl_fn(*args, **kwargs):
        gen = rl_step_coroutine(*args, **kwargs)
        intercepted = next(gen)
        return intercepted

    def post_rl_fn(intermediate, *args, **kwargs):
        gen = rl_step_coroutine(*args, **kwargs)
        _ = next(gen)
        return gen.send(intermediate)

    return rl_step_fn, pre_rl_fn, post_rl_fn


def setup_args():
    parser = argparse.ArgumentParser(description="Run the RL network simulation with configurable parameters.")
    parser.add_argument(
        '--agents', type=str, nargs='+', required=True,
        metavar='TYPE:COUNT[:PARAM]',
        help=(
            'Agent groups in order. DRL must be first if present. '
            'Format: type:count or type:count:param. '
            'Types: drl, q-aloha (param=q), eb-aloha, fw-aloha, tdma. '
            'Examples: --agents drl:5 q-aloha:1:0.333 | --agents eb-aloha:5 q-aloha:1:0.5'
        ),
    )
    parser.add_argument('--n_init', type=int, help='Number of active stations at the start (default: all). Use with --n_final to turn stations on/off at mid-run.')
    parser.add_argument('--n_final', type=int, help='Number of active stations at the end (one-shot change at mid-run).')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of training epochs to run.')
    parser.add_argument('--n_steps', type=int, default=2000, help='Number of steps per epoch.')
    parser.add_argument('--window_size', type=int, default=1, help='Size of the observation window for each agent.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--save_plots', action='store_true', default=False, help='Whether to save the generated plots.')
    parser.add_argument('--loc', type=float, default=5.0, help='loc traffic generator parameter.')
    parser.add_argument('--scale', type=float, default=0.0, help='scale traffic generator parameter')
    parser.add_argument('--f3dB', type=float, default=1.0, help='f3dB traffic generator parameter')
    parser.add_argument('--station_change_interval', type=int, help='Apply periodic station-count changes every N global steps.')
    parser.add_argument('--station_change_delta', type=int, default=0, help='Stations to add/remove per interval tick. Negative values remove stations.')
    parser.add_argument('--station_change_start_step', type=int, help='Global step when periodic station changes start. Defaults to interval.')
    parser.add_argument('--station_change_stop_step', type=int, help='Global step when periodic station changes stop.')
    parser.add_argument('--traffic_type', type=str, default='saturated', choices=['constant', 'saturated', 'bursty', 'custom'], help="Traffic model to use.")
    parser.add_argument('--phy_error_prob', type=float, default=0.1, help='Probability of error in phy channel')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_args()
    ensure_clean_git_worktree()
    commit_hash = get_short_commit_hash()

    agent_groups = parse_agent_groups(args.agents)
    n_drl = next((count for atype, count, _ in agent_groups if atype == 'drl'), 0)
    n = sum(count for _, count, _ in agent_groups)
    legacy_group_specs = [(atype, count, param) for atype, count, param in agent_groups if atype != 'drl']

    n_init = args.n_init if args.n_init is not None else n
    n_final = args.n_final if args.n_final is not None else n_init
    if not (0 < n_init <= n):
        raise ValueError(f'--n_init ({n_init}) must be between 1 and total agent count ({n})')
    if not (0 < n_final <= n):
        raise ValueError(f'--n_final ({n_final}) must be between 1 and total agent count ({n})')

    n_epochs = args.n_epochs
    n_steps = args.n_steps
    total_steps = n_epochs * n_steps
    window_size = args.window_size
    seed = args.seed
    traffic_type = args.traffic_type

    loc = args.loc
    scale = args.scale
    f3dB = args.f3dB
    station_change_interval = args.station_change_interval
    station_change_delta = args.station_change_delta
    station_change_start_step = args.station_change_start_step
    station_change_stop_step = args.station_change_stop_step
    phy_error_prob = args.phy_error_prob

    if station_change_interval is not None and station_change_interval <= 0:
        raise ValueError('--station_change_interval must be positive.')
    if station_change_interval is None:
        if station_change_start_step is not None or station_change_stop_step is not None:
            raise ValueError('--station_change_start_step/--station_change_stop_step require --station_change_interval.')
    else:
        if station_change_delta == 0:
            raise ValueError('--station_change_delta must be non-zero when --station_change_interval is used.')
        if station_change_start_step is None:
            station_change_start_step = station_change_interval

    one_shot_step = None
    one_shot_target = None
    station_change_target = n_final if args.n_final is not None else None
    if station_change_interval is None and n_final != n_init:
        one_shot_step = total_steps // 2
        one_shot_target = n_final

    key = jax.random.key(seed)
    num_actions = 2
    actions = jnp.zeros(n, dtype=int)
    buffer_states = jnp.zeros(n, dtype=int)
    power_states = jnp.full(n, INITIAL_CAPACITY, dtype=int)
    channel_state = 0
    obs = jnp.zeros((n, window_size, 5), dtype=int)
    rewards = jnp.zeros(n)
    terminals = jnp.full(n, False, dtype=bool)
    active = jnp.ones(n, dtype=bool).at[n_init:].set(False)

    lr_schedule = optax.cosine_decay_schedule(init_value=1e-4, decay_steps=60000, alpha=0.01)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr_schedule, b1=0.95, b2=0.95)
    )

    drl = BayesianDDQN(
        q_network=StochasticVariationalNetwork(QNetwork(num_actions, num_layers=1, dim=64, num_heads=4)),
        obs_space_shape=obs.shape[1:],
        act_space_size=num_actions,
        optimizer=optimizer,
        experience_replay_buffer_size=30000,
        experience_replay_batch_size=128,
        experience_replay_steps=5,
        discount=0.95,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.0,
        tau=0.05
    )
    key, init_key = jax.random.split(key)
    if n_drl > 0:
        drl_states, drl_step_fn = init_agents(drl, init_key, n_drl)
        drl_states = drl_states.replace(prev_env_state=drl_states.prev_env_state.astype(int))
    else:
        # Dummy DRL state (never used in step, just keeps Carry structure consistent)
        drl_states, drl_step_fn = init_agents(drl, init_key, 1)
        drl_states = drl_states.replace(prev_env_state=drl_states.prev_env_state.astype(int))

    # Initialize each legacy group independently
    all_legacy_states = []
    legacy_agent_steps = []
    for atype, lcount, param in legacy_group_specs:
        legacy_agent = make_legacy_agent(atype, param)
        key, init_key = jax.random.split(key)
        lstates, lstep = init_agents(legacy_agent, init_key, lcount)
        all_legacy_states.append(lstates)
        legacy_agent_steps.append((lstep, lcount))
    legacy_states = tuple(all_legacy_states)

    key, init_key = jax.random.split(key)

    if traffic_type == 'constant':
        traffic = cox_traffic(f3dB=1.0, loc=-1.0, scale=0.0, initial_state=InitialStateConf.ZERO)
    elif traffic_type == 'saturated':
        traffic = cox_traffic(f3dB=1.0, loc=5.0, scale=0.0, initial_state=InitialStateConf.ZERO)
    elif traffic_type == 'bursty':
        traffic = cox_traffic(f3dB=0.1, loc=-5.0, scale=5.0, initial_state=InitialStateConf.ZERO)
    elif traffic_type == 'custom':
        traffic = cox_traffic(f3dB=f3dB, loc=loc, scale=scale, initial_state=InitialStateConf.ZERO)
    else:
        raise ValueError(f'Unknown traffic type: {traffic_type}')

    traffic_states, traffic_step = init_traffic(traffic, init_key, n)

    rl_step_fn, _, _ = rl_step(
        drl_step_fn,
        legacy_agent_steps,
        traffic_step,
        n,
        n_drl,
        phy_error_prob,
        one_shot_step=one_shot_step,
        one_shot_target=one_shot_target,
        station_change_interval=station_change_interval,
        station_change_delta=station_change_delta,
        station_change_start_step=station_change_start_step,
        station_change_stop_step=station_change_stop_step,
        station_change_target=station_change_target,
    )
    rl_step_fn = jax.jit(rl_step_fn)
    carry = Carry(
        drl_states, legacy_states, traffic_states, buffer_states, power_states,
        channel_state, key, obs, actions, rewards, terminals, active
    )
    all_outputs = []

    for epoch in trange(n_epochs):
        global_steps = jnp.arange(epoch * n_steps, (epoch + 1) * n_steps, dtype=jnp.int32)
        carry, output = jax.lax.scan(rl_step_fn, carry, xs=global_steps)
        all_outputs.append(output)

    all_outputs = jax.tree.map(lambda *x: jnp.stack(x), *all_outputs)

    metadata = vars(args)
    metadata['n'] = n
    metadata['n_drl'] = n_drl
    metadata['n_init'] = n_init
    metadata['n_final'] = n_final

    filename = build_history_filename(n, n_drl, seed, commit_hash, slug=agents_slug(agent_groups))

    with lz4.frame.open(filename, 'wb') as f:
        cloudpickle.dump((carry.drl_states, all_outputs, metadata), f)

    if args.save_plots:
        plot_all(filename)
