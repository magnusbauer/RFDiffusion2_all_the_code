# export PY=$HOME/miniconda3/envs/rfd/bin/python
export PY=$HOME/sw/MambaForge/envs/TEST/bin/python
export PYTHONPATH=$HOME/rfdsym
NAME=filter_test_pca
rm -rf ./tmp/$NAME
$PY run_inference.py \
    +viz.rfold_iter_end=False \
    +viz.diffusion_step=False \
    +sieve.PCA.max_bigsmallratio=2.5 \
    contigmap.contigs=[\'60\'] \
    diffuser.T=20 \
    inference.num_designs=13 \
    inference.cautious=False \
    inference.output_prefix=./tmp/$NAME/$NAME \
    
    # +sieve.SS.min_helix=0.75 \
