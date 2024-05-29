#!/bin/bash

#echo RUNNING TESTS: $TESTS
#CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 PYTHONPATH=.. $NICE ./exec/bakerlab_rf_diffusion_aa.sif -mpytest --durations=10 --disable-warnings -m "not nondeterministic" --cov rf_diffusion -k "$TESTS" 2>&1
#exit 0

PARALLEL=$1

if [[ $PARALLEL == "parallel" ]]; then
   TESTS='not test_loss_grad and not test_call_speed'
   N="-n$SLURM_CPUS_PER_TASK"
   if [[ -z $SLURM_CPUS_PER_TASK ]]; then N="-n12"; fi
   NICE=nice
elif [[ $PARALLEL == "notparallel" ]]; then
   TESTS='test_loss_grad or test_call_speed'
   N=''
   NICE=''
else
   echo UNKNOWN PARALLEL $PARALLEL
   exit -1
fi

echo RUNNING TESTS: $TESTS
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=.. $NICE ./exec/bakerlab_rf_diffusion_aa.sif -mpytest --durations=10 --disable-warnings -m "not nondeterministic" --cov rf_diffusion -k "$TESTS" $N 2>&1
