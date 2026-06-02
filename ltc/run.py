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
from reinforced_lib.agents.deep.ddqn import DDQN

from ltc.agents import DCF, QNetwork
from ltc.sim import InitialStateConf, cox_traffic, normalize_obs, process_output, simulate
from ltc.sim.constants import INITIAL_CAPACITY, Actions
from ltc.utils.history import build_history_filename, ensure_clean_git_worktree, get_short_commit_hash
from ltc.utils.scan_states import Carry, Output
from ltc.utils.plots import plot_all, plot_first



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
    drl_step, legacy_step, traffic_step, n, n_drl, phy_error_prob, n_bins=50,
    one_shot_step=None, one_shot_target=None, station_change_interval=None, station_change_delta=0,
    station_change_start_step=0, station_change_stop_step=None, station_change_target=None, noise_dims=0, zero_obs=False,
):
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

        key, drl_keys, legacy_keys, traffic_key, reward_key, sim_key, noise_key = jax.random.split(c.key, 7)
        drl_keys = jax.random.split(drl_keys, n_drl)
        legacy_keys = jax.random.split(legacy_keys, n - n_drl)
        traffic_keys = jax.random.split(traffic_key, n)

        obs_norm = normalize_obs(c.obs)
        if noise_dims > 0:
            noise = jax.random.normal(noise_key, (n_drl, obs_norm.shape[1], noise_dims))
            obs_drl = noise if zero_obs else jnp.concatenate([obs_norm[:n_drl], noise], axis=-1)
        else:
            obs_drl = obs_norm[:n_drl]
        drl_states, drl_actions = yield drl_step(
            c.drl_states, drl_keys, obs_drl, c.actions[:n_drl], c.rewards[:n_drl], c.terminals[:n_drl], active[:n_drl]
        )
        legacy_states, legacy_actions = legacy_step(
            c.legacy_states, legacy_keys, obs_norm[n_drl:], c.actions[n_drl:], c.rewards[n_drl:], c.terminals[n_drl:], active[n_drl:]
        )
        actions = jnp.concatenate([drl_actions, legacy_actions])

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
        # Unused computations will be DCE-ed
        _ = next(gen)
        return gen.send(intermediate)

    return rl_step_fn, pre_rl_fn, post_rl_fn


def setup_args():
    parser = argparse.ArgumentParser(description="Run the RL network simulation with configurable parameters.")
    parser.add_argument('--n', type=int, default=10, help='Initial number of agents in the simulation.')
    parser.add_argument('--n_final', type=int, help='Final number of agents in the simulation.')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of training epochs to run.')
    parser.add_argument('--n_steps', type=int, default=2000, help='Number of steps per epoch.')
    parser.add_argument('--window_size', type=int, default=5, help='Size of the observation window for each agent.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--save_plots', action='store_true', default=False, help='Whether to save the generated plots.')
    parser.add_argument('--loc', type=float, default=5.0, help='loc traffic generator parameter.')
    parser.add_argument('--scale', type=float, default=0.0, help='scale traffic generator parameter')
    parser.add_argument('--f3dB', type=float, default=1.0, help='f3dB traffic generator parameter')
    parser.add_argument('--station_change_interval', type=int, help='Apply periodic station-count changes every N global steps.')
    parser.add_argument('--station_change_delta', type=int, default=0, help='Stations to add/remove per interval tick. Negative values remove stations.')
    parser.add_argument('--station_change_start_step', type=int, help='Global step when periodic station changes start. Defaults to interval.')
    parser.add_argument('--station_change_stop_step', type=int, help='Global step when periodic station changes stop.')
    parser.add_argument('--traffic_type', type=str, default='saturated', choices=['constant', 'saturated', 'bursty', 'custom'],help="Traffic model to use: 'constant', 'saturated', 'bursty', or 'custom'.")
    parser.add_argument('--legacy_type', type=str, default='tdma', choices=['q-aloha', 'eb-aloha', 'fw-aloha', 'tdma'], help="Legacy agent type to use: 'q-aloha', 'eb-aloha', 'fw-aloha', or 'tdma'.")
    parser.add_argument('--skip_git_check', action='store_true', default=False, help='Skip clean git worktree check.')
    parser.add_argument('--phy_error_prob', type=float, default=0.05, help='Probability of error in phy channel')
    parser.add_argument('--noise_dims', type=int, default=0, help='Gaussian noise dimensions appended to the observation.')
    parser.add_argument('--zero_obs', action='store_true', default=False, help='Replace real observations with pure noise (discard real obs).')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_args()
    if not args.skip_git_check:
        ensure_clean_git_worktree()
    commit_hash = get_short_commit_hash()

    n_init = args.n
    has_n_final = args.n_final is not None
    n_final = args.n_final if has_n_final else n_init
    n = max(n_init, n_final)
    n_epochs = args.n_epochs
    n_steps = args.n_steps
    total_steps = n_epochs * n_steps
    window_size = args.window_size
    noise_dims = args.noise_dims
    zero_obs = args.zero_obs
    seed = args.seed
    traffic_type = args.traffic_type
    phy_error_prob = args.phy_error_prob

    loc = args.loc
    scale = args.scale
    f3dB = args.f3dB
    station_change_interval = args.station_change_interval
    station_change_delta = args.station_change_delta
    station_change_start_step = args.station_change_start_step
    station_change_stop_step = args.station_change_stop_step

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
    station_change_target = n_final if has_n_final else None
    if station_change_interval is None and n_final != n_init:
        one_shot_step = total_steps // 2
        one_shot_target = n_final

    key = jax.random.key(seed)
    num_actions = 2
    actions = jnp.zeros(n, dtype=int)
    buffer_states = jnp.zeros(n, dtype=int)
    power_states = jnp.full(n, INITIAL_CAPACITY, dtype=int)
    channel_state = 0
    obs = jnp.zeros((n, window_size, 6), dtype=int)
    rewards = jnp.zeros(n)
    terminals = jnp.full(n, False, dtype=bool)
    active = jnp.ones(n, dtype=bool).at[n_init:].set(False)

    warmup_steps = int(0.02 * total_steps)
    decay_end_steps = int(0.75 * total_steps)
    peak_lr = 5e-5
    lr_schedule = optax.join_schedules([
        optax.linear_schedule(0.0, peak_lr, warmup_steps),
        optax.cosine_decay_schedule(peak_lr, decay_steps=decay_end_steps - warmup_steps, alpha=0.01),
        optax.constant_schedule(peak_lr * 0.01),
    ], boundaries=[warmup_steps, decay_end_steps])
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr_schedule),
    )

    if zero_obs and noise_dims > 0:
        obs_space_shape = (window_size, noise_dims)
    elif noise_dims > 0:
        obs_space_shape = (window_size, 6 + noise_dims)
    else:
        obs_space_shape = (window_size, 6)
    drl = DDQN(
        q_network=QNetwork(num_actions, dim=64, num_layers=2),
        obs_space_shape=obs_space_shape,
        act_space_size=num_actions,
        optimizer=optimizer,
        experience_replay_buffer_size=30000,
        experience_replay_batch_size=128,
        experience_replay_steps=5,
        discount=0.95,
        epsilon=1.0,
        epsilon_decay=0.9997,
        epsilon_min=0.0,
        tau=0.01
    )
    key, init_key = jax.random.split(key)
    drl_states, drl_step = init_agents(drl, init_key, n)

    dcf = DCF()
    key, init_key = jax.random.split(key)
    legacy_states, legacy_step = init_agents(dcf, init_key, 0)

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
        drl_step,
        legacy_step,
        traffic_step,
        n,
        n,
        phy_error_prob,
        one_shot_step=one_shot_step,
        one_shot_target=one_shot_target,
        station_change_interval=station_change_interval,
        station_change_delta=station_change_delta,
        station_change_start_step=station_change_start_step,
        station_change_stop_step=station_change_stop_step,
        station_change_target=station_change_target,
        noise_dims=noise_dims,
        zero_obs=zero_obs,
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
    filename = build_history_filename(n_init, n_final, seed, commit_hash)

    with lz4.frame.open(filename, 'wb') as f:
        cloudpickle.dump((carry.drl_states, all_outputs, vars(args)), f)

    if args.save_plots:
        plot_all(filename)
