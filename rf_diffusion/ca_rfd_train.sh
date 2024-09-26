#!/bin/bash 

container="./exec/bakerlab_rf_diffusion_aa.sif"

export MASTER_PORT=12336
### get the first node name as master address - customized for vgg slurm
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)


apptainer exec --nv $container ./train_multi_deep.py --config-name=RFdiffusion_CA.yaml
