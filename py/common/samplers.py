"""
Nicholas M. Boffi
8/19/24

Simple Euler-based samplers for probability flow equations and SDE-based diffusions.
"""

import jax
import jax.numpy as np
from flax import linen as nn
import functools
from typing import Dict, Tuple


def rollout_rev_sde(
    params: Dict,
    x0: np.ndarray,  # [d]
    ts: np.ndarray,  # [nsteps+1]
    noises: np.ndarray,  # [nsteps, d]
    score: nn.Module,
) -> Tuple[jax.Array, jax.Array]:
    """Simple Euler discretization of the reverse SDE for a variance-preserving difufsion model.
    Assumes uniform spacing.
    """
    dt = ts[1] - ts[0]
    T = ts[-1]

    def scan_fn(xt: np.ndarray, t_noise: Tuple):
        t, noise = t_noise
        xnext = (
            xt + dt * (xt + 2 * score.apply(params, xt, t)) + np.sqrt(2 * dt) * noise
        )
        return xnext, xnext

    final, traj = jax.lax.scan(
        scan_fn,
        x0,
        (ts[:-1], noises),
    )
    return final, traj


@functools.partial(jax.vmap, in_axes=(None, 0, None, 0, None))
def batch_rollout_rev_sde(
    params: Dict,
    batch_x0s: np.ndarray,  # [bs, d]
    ts: np.ndarray,  # [nsteps+1]
    batch_noises: np.ndarray,  # [bs, nsteps, d]
    score: nn.Module,
) -> Tuple[jax.Array, jax.Array]:
    """Batched version of the Euler discretization of the reverse SDE."""
    return rollout_rev_sde(params, batch_x0s, ts, batch_noises, score)


def rollout_pflow(
    params: Dict,
    x0: np.ndarray,  # [d]
    ts: np.ndarray,  # [nsteps]
    vel: nn.Module,
) -> Tuple[jax.Array, jax.Array]:
    """Simple Euler discretization of the probability flow.
    Assumes uniform spacing.
    """
    dt = ts[1] - ts[0]

    def scan_fn(xt: np.ndarray, t: np.ndarray):
        xnext = xt + dt * vel.apply(params, xt, t)
        return xnext, xnext

    # we always evaluate at t, so do not pass the final time (which is assumed to be 1)
    final, traj = jax.lax.scan(scan_fn, x0, ts[:-1])
    return final, traj


@functools.partial(jax.vmap, in_axes=(None, 0, None, None))
def batch_rollout_pflow(
    params: Dict,
    x0s: np.ndarray,  # [bs, d]
    ts: np.ndarray,  # [nsteps]
    vel: nn.Module,
) -> Tuple[jax.Array, jax.Array]:
    """Batched version of the Euler discretization of the probability flow."""
    return rollout_pflow(params, x0s, ts, vel)
