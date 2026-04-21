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

from ltc.agents import BayesianDDQN, DCF, QNetwork, StochasticVariationalNetwork
from ltc.sim import InitialStateConf, cox_traffic, process_output, simulate
from ltc.sim.constants import *
from ltc.utils.history import build_history_filename, ensure_clean_git_worktree, get_short_commit_hash
from ltc.utils.structs import MacroActionState, ObsTrackerState, ActionDecodingResults, Carry, Output
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


def action_space_size(action_space_variant, max_duration):
    if action_space_variant in {
        'txcs_duration_set',
        'txcs_to_discrete_duration',
        'txcs_to_continuous_duration',
        'txcs_to_geometric_duration',
    }:
        return 2 * max_duration
    if action_space_variant == 'tx_stage_commit':
        return 3
    raise ValueError(f'Unknown action-space variant: {action_space_variant}')


def decode_drl_actions(raw_actions, staged_tx, keys, action_space_variant, max_duration):
    raw_actions = raw_actions.astype(jnp.int32)

    if action_space_variant in {
        'txcs_duration_set',
        'txcs_to_discrete_duration',
        'txcs_to_continuous_duration',
        'txcs_to_geometric_duration',
    }:
        is_tx = raw_actions < max_duration
        duration_idx = jnp.where(is_tx, raw_actions, raw_actions - max_duration)
        action_type = jnp.where(is_tx, Actions.TX.value, Actions.CS.value)

        if action_space_variant == 'txcs_to_geometric_duration':
            p = (duration_idx.astype(jnp.float32) + 1.0) / (max_duration + 1.0)
            duration = jax.vmap(lambda k, pp: jax.random.geometric(k, pp))(keys, p).astype(jnp.int32)
            duration = jnp.clip(duration, 1, max_duration)
        else:
            duration = duration_idx + 1

        return action_type.astype(jnp.int32), duration.astype(jnp.int32), staged_tx

    if action_space_variant == 'tx_stage_commit':
        is_stage = raw_actions == 1
        is_commit = raw_actions == 2
        staged_next = jnp.where(is_stage, staged_tx + 1, staged_tx)
        commit_duration = jnp.clip(staged_tx, 1, max_duration)
        has_stage = staged_tx > 0
        action_type = jnp.where(is_commit & has_stage, Actions.TX.value, Actions.CS.value).astype(jnp.int32)
        duration = jnp.where(is_commit & has_stage, commit_duration, 1).astype(jnp.int32)
        staged_next = jnp.where(is_commit, 0, staged_next).astype(jnp.int32)
        return action_type, duration, staged_next

    raise ValueError(f'Unknown action-space variant: {action_space_variant}')


def decode_legacy_actions(raw_actions, legacy_tx_duration):
    raw_actions = raw_actions.astype(jnp.int32)
    action_type = jnp.where(
        raw_actions == Actions.TX.value,
        Actions.TX.value,
        jnp.where(raw_actions == Actions.CS.value, Actions.CS.value, Actions.IDLE.value),
    ).astype(jnp.int32)
    duration = jnp.where(
        raw_actions == Actions.TX.value,
        legacy_tx_duration,
        1,
    ).astype(jnp.int32)
    return action_type, duration


def _update_macro_state_from_ready_agents(
    macro_state, obs_tracker, ready_drl, ready_legacy,
    decoded_actions, n_drl
):
    macro_action_types = macro_state.action_types
    macro_remaining = macro_state.remaining
    staged_tx = obs_tracker.staged_tx

    macro_action_types = macro_action_types.at[:n_drl].set(
        jnp.where(ready_drl, decoded_actions.drl_action_types, macro_state.action_types[:n_drl])
    )
    macro_remaining = macro_remaining.at[:n_drl].set(
        jnp.where(ready_drl, decoded_actions.drl_durations, macro_state.remaining[:n_drl])
    )
    staged_tx = staged_tx.at[:n_drl].set(
        jnp.where(ready_drl, decoded_actions.drl_staged_next, obs_tracker.staged_tx[:n_drl])
    )

    macro_action_types = macro_action_types.at[n_drl:].set(
        jnp.where(ready_legacy, decoded_actions.legacy_action_types, macro_state.action_types[n_drl:])
    )
    macro_remaining = macro_remaining.at[n_drl:].set(
        jnp.where(ready_legacy, decoded_actions.legacy_durations, macro_state.remaining[n_drl:])
    )

    return macro_action_types, macro_remaining, staged_tx


def _finalize_macro_accumulators(macro_state, done_now, macro_tx_success_accum, macro_tx_collision_accum, macro_reward_accum):
    macro_tx_success_accum = jnp.where(done_now, 0.0, macro_tx_success_accum)
    macro_tx_collision_accum = jnp.where(done_now, 0.0, macro_tx_collision_accum)
    macro_reward_accum = jnp.where(done_now, 0.0, macro_reward_accum)
    return macro_tx_success_accum, macro_tx_collision_accum, macro_reward_accum


def upgraded_tx_slot_reward(slot_actions, channel_state, successful_packets, prev_ret_c, tx_collision_other_now):
    tx_executing = slot_actions == Actions.TX.value
    tx_success = jnp.logical_and(tx_executing, successful_packets > 0)
    tx_collision = jnp.logical_and(tx_executing, channel_state == -1)
    tx_collision_max = jnp.logical_and(tx_collision, prev_ret_c >= MAX_RETRANSMISSION)
    tx_collision_normal = jnp.logical_and(tx_collision, prev_ret_c < MAX_RETRANSMISSION)
    tx_collision_coex = jnp.logical_and(tx_collision_normal, tx_collision_other_now > 0.5)
    tx_collision_ltc = jnp.logical_and(tx_collision_normal, tx_collision_other_now <= 0.5)
    tx_empty = jnp.logical_and(
        tx_executing,
        jnp.logical_and(channel_state == 1, successful_packets == 0),
    )

    k_tx_slot = successful_packets.astype(jnp.float32)
    k_coll_slot = tx_collision.astype(jnp.float32)

    tx_success_reward = TX_REWARD * (k_tx_slot * TX_REWARD_LENGTH - 2.0) / TX_REWARD_LENGTH
    tx_collision_reward = (
        tx_collision_ltc.astype(jnp.float32) * (TX_LTC_COLLISION_PENALTY * k_coll_slot)
        + tx_collision_coex.astype(jnp.float32) * (TX_COEX_COLLISION_PENALTY * k_coll_slot)
        + tx_collision_max.astype(jnp.float32) * TX_MAX_RETRANSMISSION_PENALTY
    )
    tx_slot_reward = jnp.where(tx_success, tx_success_reward, 0.0) + tx_collision_reward + jnp.where(tx_empty, TX_EMPTY_BUFFER_PENALTY, 0.0)
    return tx_slot_reward, k_tx_slot, k_coll_slot, tx_executing


def rl_step(
    drl_step, legacy_step, traffic_step, n, n_drl, n_bins=50,
    one_shot_step=None, one_shot_target=None, station_change_interval=None, station_change_delta=0, 
    station_change_start_step=0, station_change_stop_step=None, station_change_target=None,
    obs_enable_cs_tx_same_type=False, obs_enable_tx_collision_other=False,
    action_space_variant='txcs_duration_set', max_duration=5, legacy_tx_duration=5,
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

        key, drl_keys, legacy_keys, traffic_key, reward_key = jax.random.split(c.key, 5)
        drl_keys = jax.random.split(drl_keys, n_drl)
        legacy_keys = jax.random.split(legacy_keys, n - n_drl)
        traffic_keys = jax.random.split(traffic_key, n)

        ready = c.macro.remaining == 0
        drl_ready = jnp.logical_and(active[:n_drl], ready[:n_drl])
        legacy_ready = jnp.logical_and(active[n_drl:], ready[n_drl:])

        drl_states, drl_actions_raw = yield drl_step(
            c.drl_states, drl_keys, c.obs[:n_drl], c.actions[:n_drl], c.rewards[:n_drl], c.terminals[:n_drl], drl_ready
        )
        legacy_states, legacy_actions_raw = legacy_step(
            c.legacy_states, legacy_keys, c.obs[n_drl:], c.actions[n_drl:], c.rewards[n_drl:], c.terminals[n_drl:], legacy_ready
        )

        raw_actions = c.actions
        raw_actions = raw_actions.at[:n_drl].set(jnp.where(drl_ready, drl_actions_raw, c.actions[:n_drl]))
        raw_actions = raw_actions.at[n_drl:].set(jnp.where(legacy_ready, legacy_actions_raw, c.actions[n_drl:]))

        drl_action_types, drl_durations, drl_staged_next = decode_drl_actions(
            raw_actions[:n_drl], c.obs_tracker.staged_tx[:n_drl], drl_keys, action_space_variant, max_duration
        )
        legacy_action_types, legacy_durations = decode_legacy_actions(raw_actions[n_drl:], legacy_tx_duration)

        decoded_actions = ActionDecodingResults(
            drl_action_types=drl_action_types,
            drl_durations=drl_durations,
            drl_staged_next=drl_staged_next,
            legacy_action_types=legacy_action_types,
            legacy_durations=legacy_durations,
        )

        macro_action_types, macro_remaining, staged_tx = _update_macro_state_from_ready_agents(
            c.macro, c.obs_tracker, drl_ready, legacy_ready,
            decoded_actions, n_drl
        )

        slot_executing = macro_remaining > 0
        slot_actions = jnp.where(slot_executing, macro_action_types, Actions.IDLE.value)

        traffic_states, new_frames = traffic_step(c.traffic_states, traffic_keys)
        (
            buffer_states,
            buffer_birth_steps,
            channel_state,
            packet_seqs,
            planned_packets,
            successful_packets,
        ) = simulate(c.buffer_states, c.obs_tracker.buffer_birth_steps, new_frames, slot_actions, c.packet_seqs, step)

        is_ltc = jnp.arange(n) < n_drl
        tx_mask = slot_actions == Actions.TX.value
        tx_idx = jnp.argmax(tx_mask.astype(jnp.int32))
        tx_is_ltc = is_ltc[tx_idx]
        cs_tx_same_type_now = (tx_is_ltc == is_ltc).astype(jnp.float32)
        diff_type = is_ltc[:, None] != is_ltc[None, :]
        tx_collision_other_now = jnp.any(diff_type & tx_mask[None, :], axis=1).astype(jnp.float32)

        arrival_hist = jnp.roll(c.obs_tracker.arrival_hist, -1, axis=1).at[:, -1].set(new_frames.astype(jnp.float32))
        planned_tx_hist = jnp.roll(c.obs_tracker.planned_tx_hist, -1, axis=1).at[:, -1].set(planned_packets.astype(jnp.float32))
        success_tx_hist = jnp.roll(c.obs_tracker.success_tx_hist, -1, axis=1).at[:, -1].set(successful_packets.astype(jnp.float32))
        channel_busy_hist = jnp.roll(c.obs_tracker.channel_busy_hist, -1).at[-1].set((channel_state != 0).astype(jnp.float32))
        collision_hist = jnp.roll(c.obs_tracker.collision_hist, -1).at[-1].set((channel_state == -1).astype(jnp.float32))
        tx_hist = jnp.roll(c.tx_hist, -1, axis=0).at[-1].set(tx_mask.astype(jnp.int32))

        arrival_valid = arrival_hist >= 0.0
        arrival_count = jnp.sum(arrival_valid, axis=1)
        traffic_mean_arrival_rate = jnp.where(
            arrival_count > 0,
            jnp.sum(jnp.where(arrival_valid, arrival_hist, 0.0), axis=1) / arrival_count,
            -1.0,
        )

        busy_valid = channel_busy_hist >= 0.0
        busy_count = jnp.sum(busy_valid)
        channel_occupancy_pct_window = jnp.where(
            busy_count > 0,
            jnp.sum(jnp.where(busy_valid, channel_busy_hist, 0.0)) / busy_count,
            -1.0,
        )

        coll_valid = collision_hist >= 0.0
        coll_count = jnp.sum(coll_valid)
        channel_collisions_pct_window = jnp.where(
            coll_count > 0,
            jnp.sum(jnp.where(coll_valid, collision_hist, 0.0)) / coll_count,
            -1.0,
        )

        planned_valid = planned_tx_hist >= 0.0
        planned_sum = jnp.sum(jnp.where(planned_valid, planned_tx_hist, 0.0), axis=1)
        success_sum = jnp.sum(jnp.where(planned_valid, success_tx_hist, 0.0), axis=1)
        back_pct = jnp.where(planned_sum > 0, success_sum / planned_sum, -1.0)

        tx_valid_rows = tx_hist[:, 0] >= 0
        unique_ltc_tx_window = jnp.where(
            jnp.any(tx_valid_rows),
            jnp.sum(jnp.any(jnp.where(tx_valid_rows[:, None], tx_hist[:, :n_drl], 0) > 0, axis=0).astype(jnp.float32)),
            -1.0,
        )

        done_now = jnp.logical_and(slot_executing, macro_remaining == 1)
        prev_ret_c = c.obs[:, -1, ObsIdx.STATUS_RETRY_COUNTER].astype(jnp.int32)
        obs, slot_rewards, powers = process_output(
            c.buffer_states, buffer_states, buffer_birth_steps, c.power_states, channel_state, c.obs, slot_actions,
            c.terminals, reward_key, step, traffic_mean_arrival_rate, channel_occupancy_pct_window,
            channel_collisions_pct_window, back_pct, unique_ltc_tx_window, cs_tx_same_type_now,
            tx_collision_other_now, obs_enable_cs_tx_same_type, obs_enable_tx_collision_other, roll_obs=done_now
        )

        tx_slot_reward, k_tx_slot, k_coll_slot, tx_executing = upgraded_tx_slot_reward(
            slot_actions, channel_state, successful_packets, prev_ret_c, tx_collision_other_now
        )

        slot_reward_upgraded = jnp.where(tx_executing, tx_slot_reward, slot_rewards)

        macro_tx_success_accum = c.macro.tx_success_accum + jnp.where(slot_executing, k_tx_slot, 0.0)
        macro_tx_collision_accum = c.macro.tx_collision_accum + jnp.where(slot_executing, k_coll_slot, 0.0)
        macro_reward_accum = c.macro.reward_accum + jnp.where(slot_executing, slot_reward_upgraded, 0.0)

        size_penalty = TX_SIZE_PENALTY * jnp.minimum(
            1.0, (macro_tx_success_accum - TX_SIZE_THRESHOLD + 1.0) / TX_SIZE_PENALTY_WINDOW
        )
        size_penalty = jnp.where(macro_tx_success_accum >= TX_SIZE_THRESHOLD, size_penalty, 0.0)
        rewards = jnp.where(done_now, macro_reward_accum + size_penalty, c.rewards)

        macro_tx_success_accum, macro_tx_collision_accum, macro_reward_accum = _finalize_macro_accumulators(
            c.macro, done_now, macro_tx_success_accum, macro_tx_collision_accum, macro_reward_accum
        )
        macro_remaining = jnp.where(slot_executing, macro_remaining - 1, macro_remaining)
        terminals = jnp.logical_or(c.terminals, powers < 0)

        if n_drl > 0:
            params = c.drl_states.params['model'] if 'model' in c.drl_states.params else c.drl_states.params
            flat_params, _ = jax.tree.flatten(params)
            flat_params = jax.tree.map(lambda x: x.reshape(n_drl, -1), flat_params)
            flat_params = jnp.hstack(flat_params)
            hist, bin_edges = jax.vmap(jnp.histogram, in_axes=(0, None))(flat_params, n_bins)
        else:
            hist, bin_edges = None, None

        macro_state = MacroActionState(
            remaining=macro_remaining,
            action_types=macro_action_types,
            reward_accum=macro_reward_accum,
            tx_success_accum=macro_tx_success_accum,
            tx_collision_accum=macro_tx_collision_accum,
        )
        obs_tracker_state = ObsTrackerState(
            buffer_birth_steps=c.obs_tracker.buffer_birth_steps,
            arrival_hist=arrival_hist,
            planned_tx_hist=planned_tx_hist,
            success_tx_hist=success_tx_hist,
            channel_busy_hist=channel_busy_hist,
            collision_hist=collision_hist,
            staged_tx=staged_tx,
        )
        c = Carry(
            drl_states, legacy_states, traffic_states, packet_seqs, buffer_states,
            tx_hist, powers, channel_state, key, obs, raw_actions, rewards, terminals, active,
            macro_state, obs_tracker_state
        )
        o = Output(
            legacy_states, obs, raw_actions, rewards, terminals, buffer_states, powers,
            new_frames, channel_state, active, hist, bin_edges
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
    parser.add_argument('--n', type=int, default=5, help='Initial number of agents in the simulation.')
    parser.add_argument('--n_final', type=int, help='Final number of agents in the simulation.')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of training epochs to run.')
    parser.add_argument('--n_steps', type=int, default=2000, help='Number of steps per epoch.')
    parser.add_argument('--window_size', type=int, default=1, help='Size of the observation window for each agent.')
    parser.add_argument('--queue_size', type=int, default=16, help='Fixed packet-queue length for each station.')
    parser.add_argument('--obs_stats_window', type=int, default=1000, help='Window length for rolling observation statistics.')
    parser.add_argument('--obs_enable_cs_tx_same_type', action='store_true', default=False, help='Enable CS me-vs-other transmitter feature.')
    parser.add_argument('--obs_enable_tx_collision_other', action='store_true', default=False, help='Enable TX collision-involves-other feature.')
    parser.add_argument(
        '--action_space_variant',
        type=str,
        default='txcs_duration_set',
        choices=[
            'txcs_duration_set',
            'txcs_to_discrete_duration',
            'txcs_to_continuous_duration',
            'txcs_to_geometric_duration',
            'tx_stage_commit',
        ],
        help='Macro action-space variant.',
    )
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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_args()
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
    queue_size = args.queue_size
    obs_stats_window = args.obs_stats_window
    seed = args.seed
    traffic_type = args.traffic_type

    loc = args.loc
    scale = args.scale
    f3dB = args.f3dB
    station_change_interval = args.station_change_interval
    station_change_delta = args.station_change_delta
    station_change_start_step = args.station_change_start_step
    station_change_stop_step = args.station_change_stop_step
    obs_enable_cs_tx_same_type = args.obs_enable_cs_tx_same_type
    obs_enable_tx_collision_other = args.obs_enable_tx_collision_other
    action_space_variant = args.action_space_variant

    if queue_size <= 0:
        raise ValueError('--queue_size must be positive.')
    if obs_stats_window <= 0:
        raise ValueError('--obs_stats_window must be positive.')

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
    num_actions = action_space_size(action_space_variant, MAX_MACRO_DURATION)
    actions = jnp.zeros(n, dtype=int)
    packet_seqs = jnp.zeros(n, dtype=jnp.int32)
    buffer_states = jnp.full((n, queue_size), EMPTY_PACKET_ID, dtype=jnp.int32)
    buffer_birth_steps = jnp.full((n, queue_size), -1, dtype=jnp.int32)
    arrival_hist = jnp.full((n, obs_stats_window), -1.0, dtype=jnp.float32)
    planned_tx_hist = jnp.full((n, obs_stats_window), -1.0, dtype=jnp.float32)
    success_tx_hist = jnp.full((n, obs_stats_window), -1.0, dtype=jnp.float32)
    channel_busy_hist = jnp.full(obs_stats_window, -1.0, dtype=jnp.float32)
    collision_hist = jnp.full(obs_stats_window, -1.0, dtype=jnp.float32)
    tx_hist = jnp.full((obs_stats_window, n), -1, dtype=jnp.int32)
    staged_tx = jnp.zeros(n, dtype=jnp.int32)
    macro_remaining = jnp.zeros(n, dtype=jnp.int32)
    macro_action_types = jnp.full(n, Actions.IDLE.value, dtype=jnp.int32)
    macro_reward_accum = jnp.zeros(n, dtype=jnp.float32)
    macro_tx_success_accum = jnp.zeros(n, dtype=jnp.float32)
    macro_tx_collision_accum = jnp.zeros(n, dtype=jnp.float32)
    power_states = jnp.full(n, INITIAL_CAPACITY, dtype=int)
    channel_state = 0
    obs = jnp.zeros((n, window_size, OBS_SIZE), dtype=jnp.float32)
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
    drl_states, drl_step = init_agents(drl, init_key, n)
    drl_states = drl_states.replace(prev_env_state=drl_states.prev_env_state.astype(jnp.float32))

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
        one_shot_step=one_shot_step,
        one_shot_target=one_shot_target,
        station_change_interval=station_change_interval,
        station_change_delta=station_change_delta,
        station_change_start_step=station_change_start_step,
        station_change_stop_step=station_change_stop_step,
        station_change_target=station_change_target,
        obs_enable_cs_tx_same_type=obs_enable_cs_tx_same_type,
        obs_enable_tx_collision_other=obs_enable_tx_collision_other,
        action_space_variant=action_space_variant,
        max_duration=MAX_MACRO_DURATION,
        legacy_tx_duration=LEGACY_TX_DURATION,
    )
    rl_step_fn = jax.jit(rl_step_fn)
    macro_state = MacroActionState(
        remaining=macro_remaining,
        action_types=macro_action_types,
        reward_accum=macro_reward_accum,
        tx_success_accum=macro_tx_success_accum,
        tx_collision_accum=macro_tx_collision_accum,
    )
    obs_tracker_state = ObsTrackerState(
        buffer_birth_steps=buffer_birth_steps,
        arrival_hist=arrival_hist,
        planned_tx_hist=planned_tx_hist,
        success_tx_hist=success_tx_hist,
        channel_busy_hist=channel_busy_hist,
        collision_hist=collision_hist,
        staged_tx=staged_tx,
    )
    carry = Carry(
        drl_states, legacy_states, traffic_states, packet_seqs, buffer_states,
        tx_hist, power_states, channel_state, key, obs, actions, rewards, terminals, active,
        macro_state, obs_tracker_state
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
