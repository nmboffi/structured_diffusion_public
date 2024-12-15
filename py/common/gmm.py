"""
Nicholas M. Boffi
8/19/24

Simple code for Gaussian mixture models.
"""

import jax
import jax.numpy as np
import numpy as onp
from typing import Tuple
import functools

## TODO: set up experiments
## EXPERIMENT 1: compare mengdi architecture to barron to random feature
## compare mengdi architecture to barron to random feature
## (implement mengdi architecture)
## (could try this both correctly specified in terms of knowledge of lower-dimension)
## can do this on flower mapped to higher-d as start

## EXPERIMENT 2: product of independent components
## don't need to compare to mengdi, can compare RF to barron
## product of independent components in high-dimensional space => sum of low-dimensional barron scores
## - could do product of 1d gaussian mixtures

## EXPERIMENT 3: MNIST
## compare a shallow net (barron) to the top k principal components

## EXPERIMENT 4: testing the middle region
## try optimizing over the covariance of the noise in the interpolant and see if it
## makes a difference compared to pure barron


@functools.partial(jax.jit, static_argnums=0)
def sample_gmm(
    num_samples: int,
    keys: np.ndarray,  # [num_samples+1, ...]
    *,
    weights: np.ndarray,  # [num_components]
    means: np.ndarray,  # [num_components, latent_dim]
    covariances: np.ndarray,  # [num_components, latent_dim, latent_dim]
    embedding: np.ndarray = None,  # [latent_dim, high_dim]
) -> np.ndarray:
    """Sample from a Gaussian mixture model."""

    num_components = weights.size
    key1, rest = keys[0], keys[1:]

    component_indices = jax.random.choice(
        key1, a=num_components, p=weights, shape=(num_samples,)
    )  # [num_samples]

    samples = jax.vmap(jax.random.multivariate_normal, in_axes=(0, 0, 0))(
        rest, means[component_indices], covariances[component_indices]
    )  # [num_samples, d]

    if embedding is not None:
        samples = np.dot(samples, embedding)

    return samples


def eval_gmm(
    x: np.ndarray,  # [num_samples, latent_dim]
    *,
    weights: np.ndarray,  # [num_components]
    means: np.ndarray,  # [num_components, latent_dim]
    covariances: np.ndarray,  # [num_components, latent_dim, latent_dim]
) -> np.ndarray:
    """Evaluate a Gaussian mixture model at a point."""

    num_components = weights.size
    num_samples = x.shape[0]

    densities = np.zeros(num_samples)
    for kk in range(num_components):
        densities += weights[kk] * jax.scipy.stats.multivariate_normal.pdf(
            x, means[kk], covariances[kk]
        )

    return densities


def setup_gmm(
    gmm_type: str, latent_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Set up some basic mixture models."""

    if gmm_type == "std_normal_gmm":
        weights = np.array([1.0])
        means = np.zeros((1, latent_dim))
        covariances = np.zeros((1, latent_dim, latent_dim))
        covariances[0] = np.eye(latent_dim)
        covariances = np.array(covariances)

    elif gmm_type == "basic_gmm":
        weights = np.ones(1)
        means = np.ones((4, latent_dim))
        covariances = onp.zeros((1, latent_dim, latent_dim))
        covariances[0] = onp.eye(latent_dim)
        covariances = np.array(covariances)

    elif gmm_type == "flower_gmm":
        assert latent_dim == 2, "Flower GMM only works in 2D."
        num_components = 8
        weights = np.ones(num_components) / num_components
        means = onp.zeros((num_components, latent_dim))
        covariances = onp.zeros((num_components, latent_dim, latent_dim))
        for kk in range(num_components):
            means[kk] = np.array(
                [
                    5 * np.cos(2 * np.pi * kk / num_components),
                    5 * np.sin(2 * np.pi * kk / num_components),
                ]
            )
            covariances[kk] = 0.5 * onp.eye(latent_dim)

        means = np.array(means)
        covariances = np.array(covariances)

    elif gmm_type == "square_gmm":
        assert latent_dim == 2, "Square GMM only works in 2D."
        num_components = 4
        weights = np.ones(num_components) / num_components
        means = onp.zeros((num_components, latent_dim))
        covariances = onp.zeros((num_components, latent_dim, latent_dim))
        for kk in range(num_components):
            means[kk] = np.array(
                [
                    5 * np.cos(2 * np.pi * kk / num_components),
                    5 * np.sin(2 * np.pi * kk / num_components),
                ]
            )
            covariances[kk] = 0.5 * onp.eye(latent_dim)

        means = np.array(means)
        covariances = np.array(covariances)

    elif gmm_type == "line_gmm":
        assert latent_dim == 2, "Line GMM only works in 2D."
        num_components = 2
        weights = np.ones(num_components) / num_components
        means = onp.zeros((num_components, latent_dim))
        covariances = onp.zeros((num_components, latent_dim, latent_dim))
        for kk in range(num_components):
            means[kk] = np.array(
                [
                    5 * np.cos(2 * np.pi * kk / num_components),
                    5 * np.sin(2 * np.pi * kk / num_components),
                ]
            )
            covariances[kk] = 0.5 * onp.eye(latent_dim)

        means = np.array(means)
        covariances = np.array(covariances)

    else:
        raise ValueError(f"Invalid GMM type: {gmm_type}")

    return weights, means, covariances


def setup_ind_components(
    num_ind_components: int,
    prng_key: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Set up a product of independent components.
    Hardcoded to be the random square GMM."""

    latent_dim = 2
    num_gmm_components = 4

    weights = onp.ones((num_ind_components, num_gmm_components)) / num_gmm_components
    means = onp.zeros((num_ind_components, num_gmm_components, latent_dim))
    covariances = onp.zeros(
        (num_ind_components, num_gmm_components, latent_dim, latent_dim)
    )

    for ii in range(num_ind_components):
        rot = jax.random.uniform(prng_key) * 2 * np.pi
        prng_key = jax.random.split(prng_key)[0]
        for kk in range(num_gmm_components):
            means[ii, kk] = np.array(
                [
                    5 * np.cos(2 * np.pi * kk / num_gmm_components + rot),
                    5 * np.sin(2 * np.pi * kk / num_gmm_components + rot),
                ]
            )
            covariances[ii, kk] = 0.5 * np.eye(latent_dim)

    return weights, means, covariances, prng_key


@functools.partial(jax.jit, static_argnums=0)
def sample_ind_components(
    num_samples: int,
    keys: np.ndarray,  # [num_ind_comonents, num_samples+1, ...]
    *,
    weights: np.ndarray,  # [num_ind_components, num_gmm_components]
    means: np.ndarray,  # [num_ind_components, num_gmm_components, latent_dim]
    covariances: np.ndarray,  # [num_ind_components, num_gmm_components, latent_dim, latent_dim]
) -> np.ndarray:
    """Sample from a product of independent components."""

    # [num_ind_components, num_samples, latent_dim]
    samples = jax.vmap(
        lambda key, w, m, c: sample_gmm(
            num_samples, key, weights=w, means=m, covariances=c, embedding=None
        ),
        in_axes=(0, 0, 0, 0),
    )(keys, weights, means, covariances)

    # transpose to [num_samples, num_ind_components, latent_dim]
    samples = np.transpose(samples, (1, 0, 2))

    # flatten to [num_samples, num_ind_components * latent_dim]
    samples = np.reshape(samples, (num_samples, -1))

    return samples
