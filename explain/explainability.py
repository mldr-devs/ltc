#!/usr/bin/env python
# coding: utf-8

import functools as ft

import cloudpickle
import jax
import jax.numpy as jnp
import lz4.frame
import matplotlib.pyplot as plt
import seaborn as sns

from ltc.agents.svi import StochasticVariationalNetwork
from ltc.agents import QNetwork

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")

    # Load data
    with lz4.frame.open('history_5_5_42.pkl.lz4', 'rb') as f:
        carry, history = cloudpickle.load(f)

    # Extract parameters for agent i
    i = 2
    t = 3

    params_i = {'params': jax.tree.map(lambda x: x[i], carry.params)}
    net_state_i = jax.tree.map(lambda x: x[i], carry.net_state)
    variables_i = params_i | net_state_i

    # Get observation for agent i at time t
    env_state = history.observations[t][i]

    # Initialize network
    qnetwork = StochasticVariationalNetwork(
        QNetwork(num_actions=3, num_layers=4, dim=64, num_heads=4))
    _ = qnetwork.apply(variables_i, env_state, rngs=jax.random.key(42), mutable=['loss'])[0]

    print("observations shape:", history.observations.shape)
    print("params shapes:", jax.tree.map(lambda x: x.shape, carry.params))


    @jax.jit
    @ft.partial(jax.vmap, in_axes=(0, 2))      # vmap po agentach: params[i], observations[:,:,i,...]
    @ft.partial(jax.vmap, in_axes=(None, 0))   # vmap po dim 0
    @ft.partial(jax.vmap, in_axes=(None, 0))   # vmap po dim 1 (time)
    def apply(params, state):
        return qnetwork.apply(params, state, rngs=jax.random.key(42), mutable=['loss'])[0]


    variables = {'params': carry.params} | carry.net_state
    q = apply(variables, history.observations)

    print("q.shape:", q.shape)


    def plot_q(q, filename='plot_q.pdf'):
        # q shape: (agents, dim0, dim1, 1, num_actions)
        # flatten first two time dimensions
        q_flat = q.reshape(q.shape[0], -1, *q.shape[3:])  # (agents, time, 1, num_actions)
        fig, axs = plt.subplots(nrows=q_flat.shape[0], sharex=True, sharey=True, figsize=(4, 40))
        for i, ax in enumerate(axs):
            ax.scatter(q_flat[i, :, 0, 0], q_flat[i, :, 0, 1], alpha=0.1)
            ax.set_aspect('equal')
            ax.set_title(f'agent{i}')
            # ax.set_xlim(-5, 0)
            # ax.set_ylim(-5, 0)
        plt.savefig(filename)
        plt.show()


    plot_q(q)


    # Average parameters
    def average_params(x):
        a = jnp.mean(x, axis=0, keepdims=True)
        y = jnp.repeat(a, x.shape[0], axis=0)
        return y


    aparams = jax.tree.map(average_params, variables)
    aq = apply(aparams, history.observations)
    plot_q(aq, 'plot_q_avg.pdf')


    def take_n(x, n):
        a = x[n]
        a = jnp.expand_dims(a, 0)
        y = jnp.repeat(a, x.shape[0], axis=0)
        return y


    p0 = jax.tree.map(ft.partial(take_n, n=0), variables)
    q0 = apply(p0, history.observations)

    # Plot all agent/network combinations
    n_agents = q.shape[0]
    fig, axs = plt.subplots(nrows=n_agents, ncols=n_agents, sharex=True, sharey=True, figsize=(40, 40))

    for i in range(n_agents):
        for j in range(n_agents):
            ax = axs[i, j]
            p_j = jax.tree.map(ft.partial(take_n, n=j), variables)
            q_j = apply(p_j, history.observations)
            q_j_flat = q_j.reshape(q_j.shape[0], -1, *q_j.shape[3:])
            ax.scatter(q_j_flat[i, :, 0, 0], q_j_flat[i, :, 0, 1], alpha=0.1)
            ax.set_aspect('equal')
            ax.set_title(f'agent{i} net {j}')
            # ax.set_xlim(-5, 0)
            # ax.set_ylim(-5, 0)

    plt.savefig('all.pdf')
    plt.show()
