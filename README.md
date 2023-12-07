# Setup
git clone git@github.com:baker-laboratory/rf_diffusion.git PKG_DIR

cd rf_diffusion

git submodule init

git submodule update --init

## Temporary hack
export PYTHONPATH="${PYTHONPATH}:PKG_DIR"

cd rf_diffusion

## Verify tests pass
apptainer exec /software/containers/users/dtischer/rf_diffusion_aa.sif pytest --disable-warnings -s -m "not nondeterministic"


# Simple inference pipeline run

## Running inference
To run a simple pipeline with no mpnn/scoring for the tip atom case:

`/software/containers/users/dtischer/rf_diffusion_aa.sif rf_diffusion/benchmark/pipeline.py --config-name=retroaldolase_demo_nodigs`

This will print the directory the designs are created in:
ic| conf.outdir: '/tmp/USERNAME/DATE_retroaldolase_demo'


## Viewing designs
First, start pymol:

`PYMOL_RPCHOST='0.0.0.0' PYMOL_BIN -R`

`PYMOL_BIN` on the digs is: `/software/pymol-2/bin/pymol`

Find your hostname with
`hostname -I`

Then run:
`/software/containers/users/dtischer/rf_diffusion_aa.sif rf_diffusion/dev/show_bench.py --clear=True 'DATE/*.pdb' --pymol_url=http://HOSTNAME:9123`

You should see an enzyme like this render in your pymol session:
![retroaldolase_demo](images/demo_output_retroaldolase.png)

To render some of the nice colors, you may need to add the files in `pymol_config` to your `.pymolrc`

