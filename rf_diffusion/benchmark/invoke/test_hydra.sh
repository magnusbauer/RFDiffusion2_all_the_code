#!/bin/bash 

resume=false
config=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --resume)
            resume=true
            shift
            ;;
        --config)
            config="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
done

echo "Resume: $resume"
echo "Config: $config"


script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
benchmark_dir="$(dirname "$script_dir")"
repo_dir="$(dirname "$benchmark_dir")"

outdir=$benchmark_dir'/'$config
if [ "$resume" != true ]; then
        echo "Deleting previous test run outputs"
        rm -r $outdir
fi
mkdir $outdir
cd $outdir

export PYTHONPATH=$repo_dir:$PYTHONPATH

$repo_dir/benchmark/pipeline.py \
        --config-name=$config \
        outdir="$outdir"/out/ \
        in_proc=True
