"""
Nicholas M. Boffi
8/19/24

Code for learning diffusion models and investigating the
resulting low-dimensional structure.
"""

import sys

sys.path.append("../../py")

import jax
import jax.numpy as np
import numpy as onp
import dill as pickle
from typing import Tuple, Callable, Dict
from ml_collections import config_dict
from copy import deepcopy
import argparse
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import functools
from tqdm.auto import tqdm as tqdm
import wandb
from flax.jax_utils import replicate, unreplicate
import optax
import common.losses as losses
import common.updates as updates
import common.networks as networks
import common.samplers as samplers
import common.interpolant as interpolant
import common.gmm as gmm
from typing import Callable, Tuple
import time


####### sensible matplotlib defaults #######
mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.grid.which"] = "both"
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["grid.color"] = "0.8"
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["figure.figsize"] = (8, 4)
mpl.rcParams["figure.titlesize"] = 7.5
mpl.rcParams["font.size"] = 10
mpl.rcParams["legend.fontsize"] = 7.5
mpl.rcParams["figure.dpi"] = 300
############################################


Parameters = Dict[str, Dict]


def train_loop(
    x1s: np.ndarray,
    prng_key: np.ndarray,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    schedule: optax.GradientTransformation,
    update_fn: Callable,
    data: dict,
) -> None:
    """Carry out the training loop."""
    start_time = time.time()
    cfg = data["cfg"]

    # make sure we're on the GPU/TPU
    params = jax.device_put(data["params"], jax.devices("gpu")[0])
    if cfg.ndevices > 1:
        params = replicate(params)

    ema_params = {ema_fac: deepcopy(data["params"]) for ema_fac in cfg.ema_facs}

    for curr_epoch in tqdm(range(cfg.n_epochs)):
        print(f"Starting epoch {curr_epoch}. Time: {time.time() - start_time}s")
        pbar = tqdm(range(cfg.nbatches))
        for curr_batch in pbar:
            iteration = curr_batch + curr_epoch * cfg.nbatches

            loss_fn_args, prng_key = setup_loss_fn_args(
                x1s,
                prng_key,
                curr_batch,
                data["cfg"],
            )

            params, opt_state, loss_value, grads = update_fn(
                params, opt_state, opt, loss, loss_fn_args
            )

            ## compute EMA params
            if cfg.ndevices > 1:
                curr_params = unreplicate(params)
            else:
                curr_params = params
            ema_params = updates.update_ema_params(
                curr_params, ema_params, cfg.ema_facs
            )

            ## log to wandb
            data, prng_key = log_metrics(
                data,
                x1s,
                iteration,
                curr_params,
                ema_params,
                grads,
                schedule,
                loss_value,
                loss_fn_args,
                prng_key,
            )
            if iteration == 0:
                print(f"Metrics logged. Time: {time.time() - start_time}s")

            pbar.set_postfix(loss=loss_value)

            # dump one final time
            pickle.dump(
                data,
                open(
                    f"{cfg.output_folder}/{cfg.output_name}_{iteration//cfg.save_freq}.npy",
                    "wb",
                ),
            )


def make_gmm_plot(
    x1s: np.ndarray,
    params: Parameters,
    prng_key: np.ndarray,
    cfg: config_dict.ConfigDict,
) -> None:

    ## common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 4, 4
    fontsize = 12.5

    ## set up plot array
    titles = ["base and target", "model samples and target", "model alone"]

    ## extract target samples
    plot_x1s = x1s[: cfg.plot_bs]  # [plot_bs, d]

    ## draw mode samples
    x0s = cfg.sample_rho0(cfg.plot_bs, prng_key)
    prng_key = jax.random.split(prng_key)[0]
    ts = np.linspace(
        cfg.tmin, cfg.tmax, cfg.nsteps_sample + 1
    )  # just hard-code to 500 steps for simplicity

    if cfg.loss_type == "score":
        noises = jax.random.normal(
            prng_key, shape=(cfg.plot_bs, cfg.nsteps_sample, cfg.d)
        )
        xhats, _ = batch_sample(params, x0s, ts, noises)
        prng_key = jax.random.split(prng_key)[0]
    elif cfg.loss_type == "velocity":
        xhats, _ = batch_sample(params, x0s, ts)

    ## project to latent space for visualization
    if cfg.decoder is not None:
        plot_x1s = np.dot(plot_x1s, cfg.decoder)  # [plot_bs, latent_dim]
        xhats = np.dot(xhats, cfg.decoder)  # [plot_bs, latent_dim]
    else:
        plot_x1s = plot_x1s[:, :2]
        xhats = xhats[:, :2]

    ## construct the figure
    nrows = 1
    ncols = len(titles)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fw * ncols, fh * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for kk, ax in enumerate(axs.ravel()):
        if kk <= 1:
            ax.set_xlim([-7.5, 7.5])
            ax.set_ylim([-7.5, 7.5])
            ax.set_aspect("equal")
        ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
        ax.tick_params(axis="both", labelsize=fontsize)

    # do the plotting
    for jj in range(ncols):
        title = titles[jj]
        ax = axs[jj]
        ax.set_title(title, fontsize=fontsize)

        if jj == 0:
            ax.scatter(x0s[:, 0], x0s[:, 1], s=0.1, alpha=0.5, marker="o", c="black")
            ax.scatter(
                plot_x1s[:, 0], plot_x1s[:, 1], s=0.1, alpha=0.5, marker="o", c="C0"
            )

        if jj == 1:
            ax.scatter(
                plot_x1s[:, 0], plot_x1s[:, 1], s=0.1, alpha=0.5, marker="o", c="C0"
            )

            ax.scatter(
                xhats[:, 0],
                xhats[:, 1],
                s=0.1,
                alpha=0.5,
                marker="o",
                c="black",
            )

        if jj == 2:
            ax.scatter(
                xhats[:, 0],
                xhats[:, 1],
                s=0.1,
                alpha=0.5,
                marker="o",
                c="black",
            )

    wandb.log({"samples": wandb.Image(fig)})
    return prng_key


def make_loss_fn_args_plot(
    loss_fn_args: Tuple,
    cfg: config_dict.ConfigDict,
) -> None:
    """Make a plot of the loss function arguments."""
    tbatch, x0batch, x1batch = loss_fn_args

    # remove pmap reshaping
    x0batch = np.squeeze(x0batch)
    x1batch = np.squeeze(x1batch)
    tbatch = np.squeeze(tbatch)

    ## common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 4, 4
    fontsize = 12.5

    # compute xts
    xtbatch = cfg.interp.batch_calc_It(tbatch, x0batch, x1batch)

    ## set up plot array
    titles = [r"$x_0$", r"$x_1$", r"$x_t$", r"$t$"]

    ## project for visualization
    if cfg.decoder is not None:
        x1batch = np.dot(x1batch, cfg.decoder)
        xtbatch = np.dot(xtbatch, cfg.decoder)
    else:
        x1batch = x1batch[:, :2]
        xtbatch = xtbatch[:, :2]

    ## construct the figure
    nrows = 1
    ncols = len(titles)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fw * ncols, fh * nrows),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )

    for kk, ax in enumerate(axs.ravel()):
        if kk == (len(titles) - 1):
            ax.set_xlim([cfg.tmin, cfg.tmax])
            ax.set_ylim([-0.5, 0.5])
        else:
            ax.set_xlim([-7.5, 7.5])
            ax.set_ylim([-7.5, 7.5])

        ax.set_aspect("equal")
        ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
        ax.tick_params(axis="both", labelsize=fontsize)

    # do the plotting
    for jj in range(ncols):
        title = titles[jj]
        ax = axs[jj]
        ax.set_title(title, fontsize=fontsize)

        if jj == 0:
            ax.scatter(x0batch[:, 0], x0batch[:, 1], s=0.1, alpha=0.5, marker="o")
        elif jj == 1:
            ax.scatter(x1batch[:, 0], x1batch[:, 1], s=0.1, alpha=0.5, marker="o")
        elif jj == 2:
            ax.scatter(xtbatch[:, 0], xtbatch[:, 1], s=0.1, alpha=0.5, marker="o")
        elif jj == 3:
            ax.scatter(tbatch, np.zeros_like(tbatch), s=0.1, alpha=0.5, marker="o")

    wandb.log({"loss_fn_args": wandb.Image(fig)})
    return prng_key


@functools.partial(jax.jit, static_argnums=(3,))
def generate_exponentially_distributed_points(
    prng_key: np.ndarray, exp_lambda: float, tmax: float, num_points: int
) -> np.ndarray:
    """Sample exponentially, so that there are more points clustered near the target.
    Uses inverse transform sampling."""
    U = jax.random.uniform(prng_key, shape=(num_points,))
    X = (1 / exp_lambda) * np.log(1 + U * (np.exp(exp_lambda * tmax) - 1))
    return X


def setup_loss_fn_args(
    x1s: np.ndarray,  # [n, ...]
    prng_key: np.ndarray,
    curr_batch: int,
    cfg: config_dict.ConfigDict,
) -> Tuple:
    # draw x1s, ts, and x0s
    lb = cfg.bs * curr_batch
    ub = lb + cfg.bs
    x1batch = x1s[lb:ub]
    tkey, x0key = jax.random.split(prng_key, num=2)
    x0batch = cfg.sample_rho0(cfg.bs, x0key)

    if cfg.interpolant_type == "vp_diffusion":
        ## inverse transform sampling
        tbatch = generate_exponentially_distributed_points(
            tkey, cfg.exp_lambda, cfg.tmax - cfg.eps, cfg.bs
        )

        ## even discretization
        # tbatch = cfg.tmax - cfg.tmax*np.logspace(start=np.log(cfg.tmin / cfg.tmax), stop=0, num=cfg.bs, base=np.e)

        ## uniform in log-space
        # tbatch = jax.random.uniform(tkey, shape=(cfg.bs,), minval=np.log(cfg.tmin / cfg.tmax), maxval=0.0)
        # tbatch = cfg.tmax - cfg.tmax * np.exp(tbatch)
    else:
        tbatch = jax.random.uniform(
            tkey, shape=(cfg.bs,), minval=cfg.tmin, maxval=cfg.tmax
        )

    # handle case where batch size does not divide dataset size evenly
    curr_bs = x1batch.shape[0]
    if curr_bs < cfg.bs:
        tbatch = tbatch[:curr_bs]
        x0batch = x0batch[:curr_bs]

    # set up formatting for pmap
    if cfg.ndevices > 1:
        x0batch = x0batch.reshape((cfg.ndevices, -1, *x0batch.shape[1:]))
        x1batch = x1batch.reshape((cfg.ndevices, -1, *x1batch.shape[1:]))
        tbatch = tbatch.reshape((cfg.ndevices, -1))

    # set up the loss function arguments
    loss_fn_args = (tbatch, x0batch, x1batch)

    new_key = jax.random.split(x0key)[0]
    return loss_fn_args, new_key


def log_metrics(
    data: dict,
    x1s: np.ndarray,
    iteration: int,
    curr_params: Parameters,
    ema_params: Dict[float, Parameters],
    grads: Parameters,
    schedule: optax.GradientTransformation,
    loss_value: float,
    loss_fn_args: Tuple,
    prng_key: np.ndarray,
) -> None:
    """Log some metrics to wandb, make a figure, and checkpoint the parameters."""
    cfg = data["cfg"]
    if cfg.ndevices > 1:
        grads = unreplicate(grads)

    wandb.log(
        {
            f"loss": loss_value,
            f"grad": losses.compute_grad_norm(grads),
            f"learning_rate": schedule(iteration),
        }
    )

    if (iteration % cfg.visual_freq) == 0:
        if "gmm" in cfg.target or "ind_comp" in cfg.target:
            prng_key = make_gmm_plot(x1s, curr_params, prng_key, cfg)
            make_loss_fn_args_plot(loss_fn_args, cfg)

    if (iteration % cfg.save_freq) == 0:
        data["params"] = jax.device_put(curr_params, jax.devices("cpu")[0])
        data["ema_params"] = jax.device_put(ema_params, jax.devices("cpu")[0])
        pickle.dump(
            data,
            open(
                f"{cfg.output_folder}/{cfg.output_name}_{iteration//cfg.save_freq}.npy",
                "wb",
            ),
        )

    return data, prng_key


def setup_loss(
    cfg: config_dict.ConfigDict,
) -> Callable:
    if cfg.loss_type == "score":

        @losses.mean_reduce
        @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
        def loss(params, t, x0, x1):
            return losses.score_loss(
                params,
                t,
                x0,
                x1,
                net=cfg.net,
                interp=cfg.interp,
            )

    elif cfg.loss_type == "velocity":

        @losses.mean_reduce
        @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
        def loss(params, t, x0, x1):
            return losses.vel_loss(
                params,
                t,
                x0,
                x1,
                net=cfg.net,
                interp=cfg.interp,
            )

    else:
        raise ValueError("Specified loss is not implemented.")

    return loss


def setup_sampler(cfg: config_dict.ConfigDict) -> Callable:
    """Partial complete the sampler to allow jitting with fixed-parameter networks."""
    if cfg.loss_type == "score":

        @jax.jit
        def batch_sampler(params, x0s, ts, noises):
            return samplers.batch_rollout_rev_sde(params, x0s, ts, noises, cfg.net)

    elif cfg.loss_type == "velocity":

        @jax.jit
        def batch_sampler(params, x0s, ts):
            return samplers.batch_rollout_pflow(params, x0s, ts, cfg.net)

    else:
        raise ValueError("Specified loss is not implemented.")

    return batch_sampler


def setup_base(
    cfg: config_dict.ConfigDict, ex_input: np.ndarray
) -> config_dict.ConfigDict:
    """Set up the base (Gaussian) density for the system."""

    @functools.partial(jax.jit, static_argnums=(0,))
    def sample_rho0(bs: int, key: np.ndarray):
        return jax.random.normal(key, shape=(bs, *ex_input.shape))

    cfg.sample_rho0 = sample_rho0

    return cfg


def setup_target(cfg: config_dict.ConfigDict, prng_key: np.ndarray) -> np.ndarray:
    """Set up the target density for the system."""
    if "gmm" in cfg.target:
        weights, means, covs = gmm.setup_gmm(cfg.target, cfg.latent_dim)

        # generate embedding via QR decomposition
        if cfg.latent_dim == cfg.d or "ind_comp" in cfg.target:
            cfg.embedding = np.eye(cfg.d)
        else:
            random_mat = jax.random.normal(prng_key, shape=(cfg.d, cfg.latent_dim))
            Q, _ = onp.linalg.qr(random_mat, mode="complete")
            print(f"Random matrix shape: {random_mat.shape}")
            print(f"Q shape: {Q.shape}")
            cfg.embedding = Q[:, : cfg.latent_dim].T
            print(f"Embedding shape: {cfg.embedding.shape}")
            prng_key = jax.random.split(prng_key)[0]

        cfg.decoder = cfg.embedding.T

        cfg.sample_rho1 = functools.partial(
            gmm.sample_gmm,
            weights=weights,
            means=means,
            covariances=covs,
            embedding=cfg.embedding,
        )

        key_num = cfg.n + 1
        key_shape = (key_num, 2)

    elif "ind_comp" in cfg.target:
        # compute the number of independent components
        assert (cfg.d % cfg.latent_dim) == 0
        cfg.num_ind_components = cfg.d // cfg.latent_dim

        weights, means, covs, prng_key = gmm.setup_ind_components(
            cfg.num_ind_components, prng_key
        )
        cfg.embedding = cfg.decoder = None

        cfg.sample_rho1 = functools.partial(
            gmm.sample_ind_components,
            weights=weights,
            means=means,
            covariances=covs,
        )

        key_num = cfg.num_ind_components * (cfg.n + 1)
        key_shape = (cfg.num_ind_components, cfg.n + 1, 2)
    else:
        raise ValueError("Specified target density is not implemented.")

    n_samples = cfg.n
    keys = jax.random.split(prng_key, num=key_num)
    keys = keys.reshape(key_shape)
    x1s = cfg.sample_rho1(n_samples, keys)
    prng_key = jax.random.split(keys.reshape(-1, 2)[-1])[0]

    return cfg, x1s, prng_key


def initialize_network(
    ex_input: np.ndarray, prng_key: np.ndarray, cfg: config_dict.ConfigDict
) -> Tuple[Parameters, np.ndarray]:
    ex_t = 0.0
    params = {"params": cfg.net.init(prng_key, ex_input, ex_t)["params"]}
    prng_key = jax.random.split(prng_key)[0]
    print(f"Number of parameters: {jax.flatten_util.ravel_pytree(params)[0].size}")
    return params, prng_key


def parse_command_line_arguments():
    """Process command line arguments and set up associated simulation parameters."""
    parser = argparse.ArgumentParser(
        description="Simple experiments for diffusion models with low-dimensional structure.."
    )
    parser.add_argument("--bs", type=int)
    parser.add_argument("--plot_bs", type=int)
    parser.add_argument("--visual_freq", type=int)
    parser.add_argument("--save_freq", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument("--target", type=str)
    parser.add_argument("--latent_dim", type=int)
    parser.add_argument("--tmin", type=float)
    parser.add_argument("--tmax", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--nsteps_sample", type=int)
    parser.add_argument("--decay_steps", type=int)
    parser.add_argument("--network_type", type=str)
    parser.add_argument("--n_neurons", type=int)
    parser.add_argument("--n_hidden", type=int)
    parser.add_argument("--interpolant_type", type=str)
    parser.add_argument("--exp_lambda", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--loss_type", type=str)
    parser.add_argument("--wandb_name", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--slurm_id", type=int)

    return parser.parse_args()


def setup_config_dict():
    args = parse_command_line_arguments()
    cfg = config_dict.ConfigDict()

    ## grab command line arguments
    cfg.clip = 1.0
    cfg.bs = args.bs
    cfg.plot_bs = args.plot_bs
    cfg.visual_freq = args.visual_freq
    cfg.save_freq = args.save_freq
    cfg.n = args.n
    cfg.d = args.d
    cfg.target = args.target
    cfg.latent_dim = args.latent_dim
    cfg.tmin = args.tmin
    cfg.tmax = args.tmax
    cfg.eps = args.eps
    cfg.nsteps_sample = args.nsteps_sample
    cfg.decay_steps = args.decay_steps
    cfg.steps_per_epoch = cfg.n // cfg.bs
    if cfg.steps_per_epoch * cfg.bs != cfg.n:
        cfg.steps_per_epoch += 1
    cfg.n_epochs = cfg.decay_steps // cfg.steps_per_epoch
    cfg.network_type = args.network_type
    cfg.n_neurons = args.n_neurons
    cfg.n_hidden = args.n_hidden
    cfg.interpolant_type = args.interpolant_type
    cfg.exp_lambda = args.exp_lambda
    cfg.learning_rate = args.learning_rate
    cfg.loss_type = args.loss_type
    cfg.wandb_name = f"{args.wandb_name}_{args.slurm_id}"
    cfg.wandb_project = args.wandb_project
    cfg.output_name = f"{args.output_name}_{args.slurm_id}"
    cfg.output_folder = args.output_folder
    cfg.slurm_id = args.slurm_id

    ## set up the rest of the config
    cfg.nbatches = cfg.n // cfg.bs
    cfg.ema_facs = [0.9999]
    cfg.ndevices = jax.local_device_count()

    return cfg


if __name__ == "__main__":
    print("Entering main. Setting up config dict and PRNG key.")
    prng_key = jax.random.PRNGKey(42)
    cfg = setup_config_dict()

    print("Config dict set up. Setting up target.")
    cfg, x1s, prng_key = setup_target(cfg, prng_key)

    print("Target set up. Setting up base.")
    cfg = setup_base(cfg, x1s[0])

    print("Setting up the interpolant.")
    cfg.interp = interpolant.setup_interpolant(cfg)

    print("Setting up network and initializing.")
    if cfg.network_type == "random_feature":
        cfg.n_neurons = 2 * cfg.n_neurons
        cfg.random_weights = jax.random.normal(
            prng_key, shape=(cfg.d + 1, cfg.n_neurons)
        ) / np.sqrt(cfg.d + 1)
        prng_key = jax.random.split(prng_key)[0]

        cfg.random_biases = jax.random.normal(prng_key, shape=(cfg.n_neurons,))
        prng_key = jax.random.split(prng_key)[0]
    else:
        cfg.random_weights = cfg.random_biases = None

    cfg.net = networks.setup_network(cfg)
    params, prng_key = initialize_network(x1s[0], prng_key, cfg)

    if cfg.network_type == "random_feature":
        print(f"Random weights: {cfg.net.random_weights}")

    print("Freezing config dict.")
    cfg = config_dict.FrozenConfigDict(cfg)  # freeze the config

    ## define optimizer
    print("Setting up optimizer.")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=cfg.learning_rate,
        peak_value=cfg.learning_rate,
        warmup_steps=0,
        decay_steps=int(cfg.decay_steps),
    )

    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.radam(learning_rate=schedule),
    )

    ## set up update function
    update_fn = updates.pupdate if cfg.ndevices > 1 else updates.update

    ## set up the loss function
    print("Setting up loss function.")
    loss = setup_loss(cfg)

    ## set up the sampler
    print("Setting up sampler.")
    batch_sample = setup_sampler(cfg)

    # for parallel training
    opt_state = opt.init(params)
    if cfg.ndevices > 1:
        opt_state = replicate(opt_state)
    print("Optimizer set up.")

    ## set up weights and biases tracking
    print("Setting up wandb.")
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        config=cfg.to_dict(),
    )
    print("Wandb set up.")

    ## train the model
    data = {
        "params": params,
        "cfg": cfg,
    }

    train_loop(x1s, prng_key, opt, opt_state, schedule, update_fn, data)
