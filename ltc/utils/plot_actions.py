import argparse

import cloudpickle
import lz4.frame
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ltc.sim.constants import Actions, MAX_MACRO_DURATION
from ltc.utils.history import parse_history_filename, unpack_history

TX_COLOR    = np.array([0.2,  0.75, 0.2 ])   # green
CS_COLOR    = np.array([1.0,  0.85, 0.0 ])   # yellow
IDLE_COLOR  = np.array([0.92, 0.92, 0.92])   # light gray
CH_SUCCESS  = np.array([0.2,  0.75, 0.2 ])   # green
CH_IDLE     = np.array([1.0,  0.85, 0.0 ])   # yellow
CH_COLLISION= np.array([0.9,  0.15, 0.15])   # red


def _x_ticks(ax, n_steps):
    ticks = np.arange(0, n_steps, max(1, n_steps // 20))
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=7, rotation=45)


def _vlines(ax, n_steps, every=10):
    for x in range(every, n_steps, every):
        ax.axvline(x - 0.5, color='black', linewidth=0.4, alpha=0.4)


def decode_action_type(raw_action, action_space_variant, max_duration):
    if action_space_variant in {
        'txcs_duration_set',
        'txcs_to_discrete_duration',
        'txcs_to_continuous_duration',
        'txcs_to_geometric_duration',
    }:
        return np.where(raw_action < max_duration, Actions.TX.value, Actions.CS.value)
    if action_space_variant == 'tx_stage_commit':
        return np.where(raw_action == 0, Actions.CS.value, Actions.TX.value)
    return raw_action


# ── 1. Actions ────────────────────────────────────────────────────────────────

def plot_action_slots(history, metadata, n, n_drl, seed, epoch=0, max_steps=300):
    action_space_variant = (metadata or {}).get('action_space_variant', 'txcs_duration_set')
    max_duration = (metadata or {}).get('max_duration', MAX_MACRO_DURATION)

    raw = np.array(history.actions[epoch, :max_steps])   # (steps, n_agents)
    n_steps, n_agents = raw.shape
    has_slot = hasattr(history, 'slot_actions') and history.slot_actions is not None
    if has_slot:
        types = np.array(history.slot_actions[epoch, :max_steps]).T
    else:
        types = decode_action_type(raw.T, action_space_variant, max_duration)

    grid = np.zeros((n_agents, n_steps, 3))
    grid[types == Actions.TX.value]   = TX_COLOR
    grid[types == Actions.CS.value]   = CS_COLOR
    grid[types == Actions.IDLE.value] = IDLE_COLOR

    fig, ax = plt.subplots(figsize=(max(14, n_steps / 30), max(2, n_agents * 0.7 + 1.2)))
    ax.imshow(grid, aspect='auto', interpolation='nearest', origin='upper')

    ax.set_yticks(range(n_agents))
    ax.set_yticklabels(
        [f'DRL {i}' if i < n_drl else f'Legacy {i - n_drl}' for i in range(n_agents)],
        fontsize=9,
    )
    _x_ticks(ax, n_steps)
    ax.set_xlabel('Slot', fontsize=10)
    ax.set_ylabel('Agent', fontsize=10)
    ax.set_title(f'Actions — epoch {epoch}, {n_steps} slots  [{action_space_variant}]', fontsize=11)
    ax.legend(handles=[
        mpatches.Patch(color=TX_COLOR,   label='TX'),
        mpatches.Patch(color=CS_COLOR,   label='CS'),
        mpatches.Patch(color=IDLE_COLOR, label='IDLE'),
    ], loc='upper right', fontsize=9)

    plt.tight_layout()
    out = f'action_slots_{n}_{n_drl}_{seed}_e{epoch}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.show()


# ── 2. Channel state ──────────────────────────────────────────────────────────

def plot_channel_slots(history, n, n_drl, seed, epoch=0, max_steps=300):
    ch = np.array(history.channel_state[epoch, :max_steps])   # (steps,)  values: -1, 0, 1
    n_steps = len(ch)

    grid = np.zeros((1, n_steps, 3))
    grid[0, ch == 1]  = CH_SUCCESS
    grid[0, ch == 0]  = CH_IDLE
    grid[0, ch == -1] = CH_COLLISION

    fig, ax = plt.subplots(figsize=(max(14, n_steps / 30), 1.4))
    ax.imshow(grid, aspect='auto', interpolation='nearest', origin='upper')

    ax.set_yticks([0])
    ax.set_yticklabels(['Channel'], fontsize=9)
    _x_ticks(ax, n_steps)
    ax.set_xlabel('Slot', fontsize=10)
    ax.set_title(f'Channel state — epoch {epoch}, {n_steps} slots', fontsize=11)
    ax.legend(handles=[
        mpatches.Patch(color=CH_SUCCESS,   label='TX success'),
        mpatches.Patch(color=CH_IDLE,      label='Idle'),
        mpatches.Patch(color=CH_COLLISION, label='Collision'),
    ], loc='upper right', fontsize=9)

    plt.tight_layout()
    out = f'channel_slots_{n}_{n_drl}_{seed}_e{epoch}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.show()


# ── 3. Rewards ────────────────────────────────────────────────────────────────

def plot_reward_slots(history, n, n_drl, seed, epoch=0, max_steps=300):
    rewards = np.array(history.rewards[epoch, :max_steps])   # (steps, n_agents)
    n_steps, n_agents = rewards.shape

    # diverging colormap: red=negative, white=0, green=positive
    vmax = max(np.abs(rewards).max(), 1e-6)

    fig, axes = plt.subplots(
        n_agents, 1,
        figsize=(max(14, n_steps / 30), max(3, n_agents * 1.1 + 0.8)),
        sharex=True,
    )
    if n_agents == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        r = rewards[:, i].reshape(1, -1)
        im = ax.imshow(r, aspect='auto', interpolation='nearest',
                       cmap='RdYlGn', vmin=-vmax, vmax=vmax, origin='upper')
        label = f'DRL {i}' if i < n_drl else f'Legacy {i - n_drl}'
        ax.set_yticks([0])
        ax.set_yticklabels([label], fontsize=9)
        if i == n_agents - 1:
            _x_ticks(ax, n_steps)
            ax.set_xlabel('Slot', fontsize=10)

    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.015, pad=0.01, label='Reward')
    fig.suptitle(f'Rewards per agent — epoch {epoch}, {n_steps} slots', fontsize=11, y=1.01)

    plt.tight_layout()
    out = f'reward_slots_{n}_{n_drl}_{seed}_e{epoch}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.show()


# ── 4. Combined ───────────────────────────────────────────────────────────────

def plot_combined(history, metadata, n, n_drl, seed, epoch=0, max_steps=300, grid_every=10):
    action_space_variant = (metadata or {}).get('action_space_variant', 'txcs_duration_set')
    max_duration = (metadata or {}).get('max_duration', MAX_MACRO_DURATION)

    raw     = np.array(history.actions[epoch, :max_steps])        # (steps, n_agents)
    has_slot = hasattr(history, 'slot_actions') and history.slot_actions is not None
    slot_act = np.array(history.slot_actions[epoch, :max_steps]) if has_slot else None
    ch      = np.array(history.channel_state[epoch, :max_steps])  # (steps,)
    rewards = np.array(history.rewards[epoch, :max_steps])        # (steps, n_agents)
    has_packets = hasattr(history, 'successful_packets') and history.successful_packets is not None
    pkts    = np.array(history.successful_packets[epoch, :max_steps]) if has_packets else None

    n_steps, n_agents = raw.shape

    # row layout: actions×n_agents | channel×1 | rewards×n_agents | packets×n_agents
    n_rows  = n_agents + 1 + n_agents + (n_agents if has_packets else 0)
    heights = [1] * n_agents + [1] + [1] * n_agents + ([1] * n_agents if has_packets else [])

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(max(16, n_steps / 25), n_rows * 0.55 + 1.0),
        gridspec_kw={'height_ratios': heights, 'hspace': 0.08},
        sharex=True,
    )

    # ── actions ── (slot_actions if available, else decoded raw macro-actions)
    if has_slot:
        types = slot_act.T  # (n_agents, n_steps) — already TX/CS/IDLE values
    else:
        types = decode_action_type(raw.T, action_space_variant, max_duration)
    action_grid = np.zeros((n_agents, n_steps, 3))
    action_grid[types == Actions.TX.value]   = TX_COLOR
    action_grid[types == Actions.CS.value]   = CS_COLOR
    action_grid[types == Actions.IDLE.value] = IDLE_COLOR

    for i in range(n_agents):
        ax = axes[i]
        ax.imshow(action_grid[i:i+1], aspect='auto', interpolation='nearest', origin='upper')
        _vlines(ax, n_steps, grid_every)
        label = f'DRL {i}' if i < n_drl else f'Legacy {i - n_drl}'
        ax.set_yticks([0])
        ax.set_yticklabels([label], fontsize=8)
        ax.tick_params(left=False)
        if i == 0:
            ax.set_title(
                f'Simulation overview — epoch {epoch}, {n_steps} slots  [{action_space_variant}]',
                fontsize=11, pad=6,
            )

    # divider label
    axes[n_agents - 1].spines['bottom'].set_linewidth(1.5)

    # ── channel ──
    ch_grid = np.zeros((1, n_steps, 3))
    ch_grid[0, ch == 1]  = CH_SUCCESS
    ch_grid[0, ch == 0]  = CH_IDLE
    ch_grid[0, ch == -1] = CH_COLLISION

    ax_ch = axes[n_agents]
    ax_ch.imshow(ch_grid, aspect='auto', interpolation='nearest', origin='upper')
    _vlines(ax_ch, n_steps, grid_every)
    ax_ch.set_yticks([0])
    ax_ch.set_yticklabels(['Channel'], fontsize=8)
    ax_ch.tick_params(left=False)
    ax_ch.spines['bottom'].set_linewidth(1.5)

    # ── rewards ──
    vmax = max(np.abs(rewards).max(), 1e-6)
    reward_ims = []
    for i in range(n_agents):
        ax = axes[n_agents + 1 + i]
        r = rewards[:, i].reshape(1, -1)
        im = ax.imshow(r, aspect='auto', interpolation='nearest',
                       cmap='RdYlGn', vmin=-vmax, vmax=vmax, origin='upper')
        _vlines(ax, n_steps, grid_every)
        reward_ims.append(im)
        label = f'R DRL {i}' if i < n_drl else f'R Leg {i - n_drl}'
        ax.set_yticks([0])
        ax.set_yticklabels([label], fontsize=8)
        ax.tick_params(left=False)

    fig.colorbar(reward_ims[-1], ax=axes[n_agents + 1:n_agents + 1 + n_agents], orientation='vertical',
                 fraction=0.012, pad=0.01, label='Reward')

    # ── packets ──
    if has_packets:
        PKT_COLOR = np.array([0.2, 0.75, 0.2])
        NO_PKT_COLOR = np.array([0.93, 0.93, 0.93])
        for i in range(n_agents):
            ax = axes[n_agents + 1 + n_agents + i]
            pkt_grid = np.where(pkts[:, i:i+1].T == 1, 1.0, 0.0)
            rgb = np.where(pkts[:, i:i+1].T[..., None] == 1, PKT_COLOR, NO_PKT_COLOR)
            ax.imshow(rgb.reshape(1, n_steps, 3), aspect='auto', interpolation='nearest', origin='upper')
            _vlines(ax, n_steps, grid_every)
            label = f'Pkt DRL {i}' if i < n_drl else f'Pkt Leg {i - n_drl}'
            ax.set_yticks([0])
            ax.set_yticklabels([label], fontsize=8)
            ax.tick_params(left=False)

    # ── x ticks only on last row ──
    _x_ticks(axes[-1], n_steps)
    axes[-1].set_xlabel('Slot', fontsize=10)

    # ── shared legend (actions + channel) ──
    legend_handles = [
        mpatches.Patch(color=TX_COLOR,     label='TX'),
        mpatches.Patch(color=CS_COLOR,     label='CS'),
        mpatches.Patch(color=IDLE_COLOR,   label='IDLE'),
        mpatches.Patch(color=CH_SUCCESS,   label='Ch: success'),
        mpatches.Patch(color=CH_IDLE,      label='Ch: idle'),
        mpatches.Patch(color=CH_COLLISION, label='Ch: collision'),
    ]
    if has_packets:
        legend_handles.append(mpatches.Patch(color=np.array([0.2, 0.75, 0.2]), label='Packet sent'))
        legend_handles.append(mpatches.Patch(color=np.array([0.93, 0.93, 0.93]), label='No packet'))
    fig.legend(handles=legend_handles, loc='upper right', fontsize=8, ncol=2, bbox_to_anchor=(0.88, 1.0))

    out = f'overview_{n}_{n_drl}_{seed}_e{epoch}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.show()


# ── 5. Raw data inspector ─────────────────────────────────────────────────────

def inspect(history, metadata, n, n_drl, seed, epoch=0, max_steps=20):
    from ltc.sim.constants import ObsIdx

    n_epochs = history.actions.shape[0]
    n_steps_total = history.actions.shape[1]
    n_agents = history.actions.shape[2]

    print("=" * 70)
    print(f"FILE:    n={n}  n_drl={n_drl}  seed={seed}")
    print(f"SHAPE:   {n_epochs} epochs × {n_steps_total} steps × {n_agents} agents")
    print(f"ARGS:    {metadata}")
    print("=" * 70)

    raw     = np.array(history.actions[epoch, :max_steps])         # (steps, agents)
    ch      = np.array(history.channel_state[epoch, :max_steps])   # (steps,)
    rewards = np.array(history.rewards[epoch, :max_steps])         # (steps, agents)
    obs     = np.array(history.observations[epoch, :max_steps])    # (steps, agents, win, obs_size)
    buf     = np.array(history.buffer_states[epoch, :max_steps])   # (steps, agents, queue)
    has_pkts = hasattr(history, 'successful_packets') and history.successful_packets is not None
    pkts_raw = np.array(history.successful_packets[epoch, :max_steps]) if has_pkts else None
    has_slot = hasattr(history, 'slot_actions') and history.slot_actions is not None
    slot_raw = np.array(history.slot_actions[epoch, :max_steps]) if has_slot else None

    action_space_variant = (metadata or {}).get('action_space_variant', 'txcs_duration_set')
    max_duration = (metadata or {}).get('max_duration', MAX_MACRO_DURATION)
    macro_types = decode_action_type(raw.T, action_space_variant, max_duration).T  # (steps, agents)

    action_names = {Actions.TX.value: 'TX', Actions.CS.value: 'CS', Actions.IDLE.value: 'IDLE'}
    ch_names = {1: 'SUCCESS', 0: 'IDLE', -1: 'COLLISION'}

    print(f"\n── Epoch {epoch}, first {max_steps} slots ──\n")
    col_hdr = "raw/macro/slot/rew/pkt" if has_pkts else "raw/macro/slot/rew"
    header = f"{'Slot':>5}  {'Channel':>10}  " + "  ".join(f"A{i}({col_hdr})" for i in range(n_agents))
    print(header)
    print("-" * len(header))

    for t in range(min(max_steps, n_steps_total)):
        ch_str = ch_names.get(int(ch[t]), str(ch[t]))
        agent_cols = []
        for i in range(n_agents):
            macro_str = action_names.get(int(macro_types[t, i]), str(macro_types[t, i]))
            slot_str = action_names.get(int(slot_raw[t, i]), '?') if has_slot else '?'
            pkt_part = f"/{int(pkts_raw[t,i])}" if has_pkts else ""
            agent_cols.append(f"{int(raw[t,i]):>3}/{macro_str:<4}/{slot_str:<4}/{rewards[t,i]:>+6.2f}{pkt_part}")
        print(f"{t:>5}  {ch_str:>10}  " + "  ".join(agent_cols))

    print(f"\n── Observations (last obs window, epoch {epoch}, first {max_steps} steps) ──\n")
    obs_names = [name for name, _ in sorted(vars(ObsIdx).items(), key=lambda x: x[1])
                 if not name.startswith('_')]
    for t in range(min(max_steps, n_steps_total)):
        print(f"  Step {t}:")
        for i in range(n_agents):
            label = f'DRL {i}' if i < n_drl else f'Legacy {i-n_drl}'
            vals = obs[t, i, -1]  # last window entry
            obs_str = "  ".join(f"{obs_names[j]}={vals[j]:.2f}" for j in range(len(obs_names)))
            print(f"    [{label}] {obs_str}")

    print(f"\n── Buffer states (epoch {epoch}, first {max_steps} steps) ──\n")
    for t in range(min(max_steps, n_steps_total)):
        bufs = "  ".join(f"A{i}:{list(buf[t,i])}" for i in range(n_agents))
        print(f"  Step {t}: {bufs}")

    print("\n── Summary statistics ──\n")
    has_pkts_hist = hasattr(history, 'successful_packets') and history.successful_packets is not None
    for i in range(n_agents):
        label = f'DRL {i}' if i < n_drl else f'Legacy {i-n_drl}'
        all_types = decode_action_type(
            np.array(history.actions[:, :, i]).flatten(),
            action_space_variant, max_duration,
        )
        tx_pct  = 100 * np.mean(all_types == Actions.TX.value)
        cs_pct  = 100 * np.mean(all_types == Actions.CS.value)
        total_r = float(np.array(history.rewards[:, :, i]).sum())
        mean_r  = float(np.array(history.rewards[:, :, i]).mean())
        pkt_str = ''
        if has_pkts_hist:
            total_pkts = int(np.array(history.successful_packets[:, :, i]).sum())
            pkt_str = f'  packets={total_pkts}'
        print(f"  {label:12s}  TX={tx_pct:.1f}%  CS={cs_pct:.1f}%  "
              f"total_reward={total_r:.1f}  mean_reward={mean_r:.4f}{pkt_str}")

    ch_all = np.array(history.channel_state).flatten()
    print(f"\n  Channel:  success={100*np.mean(ch_all==1):.1f}%  "
          f"idle={100*np.mean(ch_all==0):.1f}%  "
          f"collision={100*np.mean(ch_all==-1):.1f}%")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze simulation history file')
    parser.add_argument('filename', help='Path to .pkl.lz4 history file')
    parser.add_argument('--epoch', type=int, default=0, help='Epoch index to visualize')
    parser.add_argument('--steps', type=int, default=300, help='Max slots to show')
    parser.add_argument('--grid', type=int, default=10, help='Vertical grid every N slots (0=off)')
    parser.add_argument('--inspect', action='store_true', help='Print raw data instead of plotting')
    parser.add_argument('--inspect_steps', type=int, default=20, help='Slots to print in inspect mode')
    args = parser.parse_args()

    n, n_drl, seed, _ = parse_history_filename(args.filename)

    with lz4.frame.open(args.filename, 'rb') as f:
        _, history, metadata = unpack_history(cloudpickle.load(f))

    if args.inspect:
        inspect(history, metadata, n, n_drl, seed, epoch=args.epoch, max_steps=args.inspect_steps)
    else:
        plot_combined(history, metadata, n, n_drl, seed, epoch=args.epoch, max_steps=args.steps,
                      grid_every=args.grid if args.grid > 0 else 10**9)
