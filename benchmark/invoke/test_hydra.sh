#!/bin/bash 

echo "ARG 1: $1";
resume=true
if [[ $1 == "--resume" ]]; then
        echo "Resuming test"
        resume=true
else
        echo "Running fresh test"
fi

benchmark_json='bench_07-26.json'

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
benchmark_dir="$(dirname "$script_dir")"
repo_dir="$(dirname "$benchmark_dir")"

outdir=$benchmark_dir'/test_output'
if [ "$resume" != true ]; then
        echo "Deleting previous test run outputs"
        rm -r $outdir
fi
mkdir $outdir
cd $outdir

export PYTHONPATH=$repo_dir:$PYTHONPATH

DIFFUSION_ARGS="--config-name=aa diffuser.T=2 inference.input_pdb=$benchmark_dir/input/gaa.pdb inference.ligand=LG1 contigmap.contigs='[4-4,A518-519]'"

$repo_dir/benchmark/pipeline.py \
    outdir="$outdir"/out/ \
    use_ligand=True \
    slurm_submit=True \
    in_proc=True \
    start_step=compile \
    sweep.command_args=\""$DIFFUSION_ARGS"\" \
    sweep.num_per_condition=2 \
    sweep.num_per_job=2 \
    mpnn.num_seq_per_target=1 \
    score.run=af2 \
    
