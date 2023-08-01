#!/bin/bash 

#outdir='/net/scratch/'$USER'/aa_diffusion/benchmark/test'
echo "ARG 1: $1";
resume=false
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

$repo_dir/benchmark/pipeline.py \
        --in_proc \
        --use_ligand \
        --num_per_condition 2 --num_per_job 2 --out $outdir/out/run \
        --args "--config-name=aa diffuser.T=2 contigmap.length=200-200 inference.input_pdb=$benchmark_dir/input/gaa.pdb inference.ligand=LG1 contigmap.contigs=[4-4,A518-519] contigmap.length=6-6" \
        --num_seq_per_target 1 --af2_gres=gpu:a6000:1 -p cpu --no_tmalign --score_scripts='af2'
