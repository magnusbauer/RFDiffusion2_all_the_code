#!/bin/bash 

resume=false
config="test_config"

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

outdir=$benchmark_dir'/test/'$config
if [ "$resume" != true ]; then
        echo "Deleting previous test run outputs"
        echo "outdir: $outdir"
        rm -r $outdir
fi
mkdir $outdir

$repo_dir/benchmark/pipeline.py \
        --config-name=$config \
        outdir="$outdir/out/"
