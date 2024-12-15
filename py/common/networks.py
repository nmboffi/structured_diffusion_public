"""
Nicholas M. Boffi
8/19/24

Basic networks for structured experiments.
"""

import jax
import jax.numpy as np
import flax.linen as nn
from ml_collections import config_dict
from typing import Callable


class Barron(nn.Module):
    n_neurons: int
    output_dim: int
    random_weights: np.ndarray = None
    random_biases: np.ndarray = None

    def setup(self):
        self.random_feature_model = self.random_weights is not None

        if not self.random_feature_model:
            self.hidden = nn.Dense(self.n_neurons)

        self.output = nn.Dense(self.output_dim)

    def __call__(self, x: np.ndarray, t: float):
        x = np.concatenate([x, np.array([t])])
        if self.random_feature_model:
            x = np.dot(x, self.random_weights) + self.random_biases
        else:
            x = self.hidden(x)
        x = jax.nn.swish(x)
        return self.output(x)


class Mengdi(nn.Module):
    """Architecture from Mengdi Wang paper, with learnable encoder/decoder linear layer and an MLP in the latent space.
    Allows option for known encoder/decoder."""

    n_neurons: int
    n_hidden: int
    output_dim: int
    latent_dim: int
    embedding: np.ndarray  # [latent_dim, d]
    alpha: Callable[[float], float]

    @nn.compact
    def __call__(self, x: np.ndarray, t: float):
        if self.embedding is not None:
            embedding = self.embedding
        else:
            embedding = self.param(
                "encoder", nn.initializers.normal(), (self.latent_dim, self.output_dim)
            )  # [latent_dim, d]

        z = np.dot(x, embedding.T)
        z = MLP(self.n_neurons, self.n_hidden, self.latent_dim)(z, t)

        ht = self.alpha(t) ** 2
        return (np.dot(z, embedding) - x) / np.maximum(ht, 1e-5)


class MengdiQR(nn.Module):
    """Architecture from Mengdi Wang paper, with learnable encoder/decoder linear layer and an MLP in the latent space.
    Allows option for known encoder/decoder."""

    n_neurons: int
    n_hidden: int
    output_dim: int
    latent_dim: int
    embedding: np.ndarray  # [latent_dim, d]

    @nn.compact
    def __call__(self, x: np.ndarray, t: float):
        if self.embedding is not None:
            embedding = self.embedding
        else:
            embedding = self.param(
                "encoder", nn.initializers.normal(), (self.latent_dim, self.output_dim)
            )  # [latent_dim, d]

        # compute QR factorizaiton to ensure orthogonality
        embedding = np.linalg.qr(embedding.T)[0]
        z = np.dot(x, embedding)
        z = MLP(self.n_neurons, self.n_hidden, self.latent_dim)(z, t)

        ht = self.alpha(t) ** 2
        return (np.dot(x, embedding.T) - x) / np.maximum(ht, 1e-5)


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    n_neurons: int
    n_hidden: int
    output_dim: int

    @nn.compact
    def __call__(self, x: np.ndarray, t: float):
        x = np.concatenate([x, np.array([t])])

        for _ in range(self.n_hidden):
            x = nn.Dense(self.n_neurons)(x)
            x = jax.nn.swish(x)

        return nn.Dense(self.output_dim)(x)


def setup_network(
    config: config_dict.ConfigDict,
) -> nn.Module:
    """Setup the neural network for the system.

    Args:
        config: Configuration dictionary.
    """

    if config.network_type == "barron" or config.network_type == "random_feature":
        return Barron(
            n_neurons=config.n_neurons,
            output_dim=config.d,
            random_weights=config.random_weights,
            random_biases=config.random_biases,
        )
    elif config.network_type == "mengdi":
        return Mengdi(
            n_neurons=config.n_neurons,
            n_hidden=config.n_hidden,
            output_dim=config.d,
            latent_dim=config.latent_dim,
            embedding=None,
            alpha=config.interp.alpha,
        )
    elif config.network_type == "mengdi_QR":
        return MengdiQR(
            n_neurons=config.n_neurons,
            n_hidden=config.n_hidden,
            output_dim=config.d,
            latent_dim=config.latent_dim,
            embedding=None,
        )
    elif config.network_type == "mengdi_known":
        return Mengdi(
            n_neurons=config.n_neurons,
            n_hidden=config.n_hidden,
            output_dim=config.d,
            latent_dim=config.latent_dim,
            embedding=config.embedding,
            alpha=config.interp.alpha,
        )
    elif config.network_type == "mengdi_known_QR":
        return MengdiQR(
            n_neurons=config.n_neurons,
            n_hidden=config.n_hidden,
            output_dim=config.d,
            latent_dim=config.latent_dim,
            embedding=config.embedding,
        )

    elif config.network_type == "mlp":
        return MLP(
            n_neurons=config.n_neurons,
            n_hidden=config.n_hidden,
            output_dim=config.d,
        )
    else:
        raise ValueError(f"Network type {config.network_type} not recognized.")
