#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/nb3397/results/structured_diffusion/flower_gmm/11_17_24/output/%A_%a.out
#SBATCH --output=/scratch/nb3397/results/structured_diffusion/flower_gmm/11_17_24/output/%A_%a.err

## convenience definitions
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif
OVERLAY_FILE=/scratch/nb3397/technical/singularity_setup/jax_4_30_24.ext3:ro
PY_DIR=/scratch/nb3397/projects/structured_diffusions/py/launchers
output=/scratch/nb3397/results/structured_diffusion/flower_gmm/11_17_24
launcher_file=learn.py

# remove all modules
module purge

# sweep over two loss types using job arrays
ind=$SLURM_ARRAY_TASK_ID
ds=(2 4 8 16 32 64)
network_architectures=('barron' 'random_feature' 'mlp')

## get length of lists for indexing
n_network_architectures=${#network_architectures[@]}

## index ind into two-dimensional array
i=$((ind / n_network_architectures))
j=$((ind % n_network_architectures))

d=${ds[$i]}
network_type=${network_architectures[$j]}
n_neurons=1024

singularity exec --nv --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
    /bin/bash -c "source /ext3/env.sh; \
	cd $PY_DIR; \
	python $launcher_file \
    --bs 250000 \
    --plot_bs 2048 \
    --visual_freq 5000 \
    --save_freq 10000 \
    --n 1000000 \
    --d $d \
    --target 'flower_gmm' \
    --latent_dim 2 \
    --tmin 0.0 \
    --tmax 5.0 \
    --eps 1e-5 \
    --nsteps_sample 1024 \
    --decay_steps 250000 \
    --network_type $network_type \
    --n_neurons $n_neurons \
    --n_hidden 2 \
    --interpolant_type 'vp_diffusion' \
    --exp_lambda 1.0 \
    --learning_rate 0.0005 \
    --loss_type 'score' \
    --wandb_name '11_17_24_arch_sweep' \
    --wandb_project 'structured_diffusion' \
    --output_name '11_17_24_arch_sweep' \
    --output_folder $output \
    --slurm_id $ind"
exit
