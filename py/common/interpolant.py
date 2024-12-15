"""
Nicholas M. Boffi
8/19/24

Simple implementation of a stochastic interpolant / a diffusion model.
"""

import jax
import jax.numpy as np
import dataclasses
import functools
from typing import Callable
from ml_collections import config_dict


@dataclasses.dataclass
class Interpolant:
    """Basic class for a stochastic interpolant."""

    alpha: Callable[[float], float]
    beta: Callable[[float], float]
    alpha_dot: Callable[[float], float]
    beta_dot: Callable[[float], float]

    def calc_It(self, t: float, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return self.alpha(t) * x0 + self.beta(t) * x1

    def calc_It_dot(self, t: float, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return self.alpha_dot(t) * x0 + self.beta_dot(t) * x1

    def calc_score_target(self, t: float, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        """Note: assumes that x0 is a Gaussian!"""
        return -x0 / self.alpha(t)

    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def batch_calc_It(
        self, t: np.ndarray, x0: np.ndarray, x1: np.ndarray
    ) -> np.ndarray:
        return self.calc_It(t, x0, x1)

    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def batch_calc_It_dot(
        self, t: np.ndarray, x0: np.ndarray, x1: np.ndarray
    ) -> np.ndarray:
        return self.calc_It_dot(t, x0, x1)

    def __hash__(self):
        return hash((self.alpha, self.beta))

    def __eq__(self, other):
        return self.alpha == other.alpha and self.beta == other.beta


def setup_interpolant(
    config: config_dict.ConfigDict,
) -> Interpolant:
    """Set up the interpolant for the system.

    Args:
        config: Configuration dictionary.
    """
    if config.interpolant_type == "linear":
        return Interpolant(
            alpha=lambda t: 1 - t,
            beta=lambda t: t,
            alpha_dot=lambda t: -1,
            beta_dot=lambda t: 1,
        )

    elif config.interpolant_type == "vp_diffusion":
        return Interpolant(
            alpha=lambda t: np.sqrt(1 - np.exp(2 * (t - config.tmax))),
            beta=lambda t: np.exp(t - config.tmax),
            alpha_dot=lambda t: -np.exp(2 * (t - config.tmax))
            / np.sqrt(1 - np.exp(2 * (t - config.tmax))),
            beta_dot=lambda t: np.exp(t - config.tmax),
        )

    elif config.interpolant_type == "vp_diffusion_logscale":
        return Interpolant(
            alpha=lambda t: np.sqrt(1 - t**2),
            beta=lambda t: t,
            alpha_dot=lambda t: -t / np.sqrt(1 - t**2),
            beta_dot=lambda t: 1,
        )

    elif config.interpolant_type == "ve_diffusion":
        return Interpolant(
            alpha=lambda t: config.tf - t,
            beta=lambda t: 1,
            alpha_dot=lambda t: -1,
            beta_dot=lambda t: 0,
        )

    else:
        raise ValueError(f"Interpolant type {config.interpolant_type} not recognized.")
