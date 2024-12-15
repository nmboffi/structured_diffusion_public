"""
Nicholas M. Boffi
8/19/24

Loss functions for learning.
"""

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as np
from typing import Dict
import functools
from . import interpolant as interpolant
import flax.linen as nn

Parameters = Dict[str, Dict]


@jax.jit
def compute_grad_norm(grads: Dict) -> float:
    """Computes the norm of the gradient, where the gradient is input
    as an hk.Params object (treated as a PyTree)."""
    flat_params = ravel_pytree(grads)[0]
    return np.linalg.norm(flat_params) / np.sqrt(flat_params.size)


def mean_reduce(func):
    """A decorator that computes the mean of the output of the decorated function.
    Designed to be used on functions that are already batch-processed (e.g., with jax.vmap).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        batched_outputs = func(*args, **kwargs)
        return np.mean(batched_outputs)

    return wrapper


def score_loss(
    params: Parameters,
    t: float,
    x0: np.ndarray,  # [d]
    x1: np.ndarray,  # [d]
    *,
    net: nn.Module,
    interp: interpolant.Interpolant,
) -> np.ndarray:
    """Compute the loss for the score on a single sample."""
    xt = interp.calc_It(t, x0, x1)
    net_eval = net.apply(params, xt, t)
    return np.sum(net_eval**2) - 2 * np.sum(
        net_eval * interp.calc_score_target(t, x0, x1)
    )


def vel_loss(
    params: Parameters,
    t: float,
    x0: np.ndarray,  # [d]
    x1: np.ndarray,  # [d]
    *,
    net: nn.Module,
    interp: interpolant.Interpolant,
) -> np.ndarray:
    """Compute the loss for the velocity on a single sample."""
    xt = interp.calc_It(t, x0, x1)
    net_eval = net.apply(params, xt, t)
    return np.sum(net_eval**2) - 2 * np.sum(net_eval * interp.calc_It_dot(t, x0, x1))
