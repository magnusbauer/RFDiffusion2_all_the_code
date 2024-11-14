#!/bin/bash

echo helen_test_whs_sym.sh

export PYTHONPATH=.

SYM=T
NSUB=999
NRES=60
SRAD=12
FRAD=0

# for SYM in I2 I3 I5; do

OUT=test_splitsym_${SYM}_${NRES}_r${SRAD}_s${NSUB}
echo $OUT

/home/sheffler/sw/MambaForge/envs/rfdsym312/bin/python ./rf_diffusion/run_inference.py \
--config-name=sym \
inference.num_designs=1 \
sym.kind='rf_diffusion' \
inference.cautious=False \
diffuser.T=25 \
sym.symid=$SYM \
inference.write_trajectory=true \
inference.ckpt_path='/data/models/RFD_45.pt' \
viz.settings.enabled=false \
viz.settings.showfrac=0.2 \
viz.rfold_iter_end=true \
viz.diffusion_step=true \
contigmap.contigs=[\'$NRES\'] \
sym.max_nsub=$NSUB \
sym.start_radius=$SRAD \
inference.output_prefix=./$OUT/$OUT \

# sym.asu_to_best_frame=true \
# sym.recenter_xt_chains_on_px0=true \
# sym.recenter_for_diffusion=true  \
# sym.force_radius="\"diffuse:[[0.3,0.4]]:[$FRAD,None]\"" \

# inference.input_pdb='rf_diffusion/test_data/1qys.pdb' \
# inference.conditions.radius_of_gyration_v2.active=True \
# inference.ckpt_path='/data/models/RFD_45.pt' \
# inference.ckpt_path='/data/models/helen_RFD_32.pt' \

# sym.rf_asym_only=\'diffuse:[[0,0.5]]\' \

# done
						       
echo DONE
