#!/bin/bash
#SBATCH -p gpu
#SBATCH -J training_example
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=4
#SBATCH --time=7:00:00:00

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=11010

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

srun train_multi_deep.py --config-name flow_matching_base

