# Setup
Let's define some paths. Begin by `cd`ing to any directory you like.
```
REPO_NAME="rf_diffusion_repo"  # Set REPO_NAME to be anything you like.
REPO_DIR="$PWD/$REPO_NAME"
```
Now clone the repo.
```
git clone -b aa git@github.com:baker-laboratory/rf_diffusion.git $REPO_NAME
cd $REPO_DIR
git submodule init
git submodule update --init
```
## Verify tests pass
```
export PYTHONPATH="${PYTHONPATH}:$REPO_DIR"
cd $REPO_DIR/rf_diffusion
apptainer exec /software/containers/users/dtischer/rf_diffusion_aa.sif pytest --disable-warnings -s -m "not nondeterministic"
```
## A note about running scripts in this repo
Many of the python scripts in this repo are executable. That is, you don't need to do `python some_script.py` or `my_container.sif some_script.py`. The are several environmental variables and flags that need to be set properly for the scripts to run. Rather than asking the user to set these correctly each time (which is error prone), we have a script to do this prep work under the hood for you! 

Any scripts (like `rf_diffusion/benchmark/pipeline.py` and `rf_diffusion/run_inference.py`) that have the shebang line `#!/net/software/containers/users/dtischer/shebang_rf_se3_diffusion_dev.sh` can be executed directly. If you need to run a script without that line, you need to do
```
export PYTHONPATH="${PYTHONPATH}:$REPO_DIR"
/usr/bin/apptainer run --nv --slurm --env PYTHONPATH="\$PYTHONPATH:$PYTHONPATH" /net/software/containers/users/dtischer/rf_diffusion_aa.sif path/to/script.py ...
```

# Simple inference pipeline run
## Running inference
To run a demo of some of the inference capabilities, including enzyme design from tip atoms, enzyme design from tip atoms of unknown sequence position, ligand binder design, traditional contiguous motif scaffolding, and molecular glue design (binder to protein:small_molecule complex).  (See `$REPO_DIR/rf_diffusion/benchmark/demo.json` for how these tasks are declared)

`$REPO_DIR/rf_diffusion/benchmark/pipeline.py --config-name=demo_only_design`

This will print the directory the designs are created in:
ic| conf.outdir: OUTDIR

Once the pipeline finishes (check sjobs for an array job named `sweep_hyperparameters`), view the designs:

## Viewing designs
First, start pymol:

`PYMOL_RPCHOST='0.0.0.0' PYMOL_BIN -R`

`PYMOL_BIN` on the digs is: `/software/pymol-2/bin/pymol`

Find your hostname with
`hostname -I`

Then run:
`$REPO_DIR/rf_diffusion/dev/show_bench.py --clear=True 'OUTDIR/*.pdb' --pymol_url=http://HOSTNAME:9123`

You should see multiple designs (such as this enzyme design) render in your pymol session:
![retroaldolase_demo](images/demo_output_retroaldolase.png)

To render some of the nice colors, you may need to add the files in `pymol_config` to your `.pymolrc`

## Running inference (OUTSIDE OF DIGS)
To run a simple pipeline with no mpnn/scoring for the tip atom case:

`$REPO_DIR/rf_diffusion/benchmark/pipeline.py --config-name=retroaldolase_demo_nodigs`

## Running catalytic constraint benchmarking

Put your un-mpnned designs in a folder, call it $MY_FOLDER

Each design is expected to be a .pdb, with a .trb file with the same file prefix.
The trb file is expected to contain a pickle that has the following structure:

```
{'con_hal_pdb_idx': [('A', 114), ('A', 115), ('A', 85)],
 'con_ref_pdb_idx': [('A', 1), ('A', 2), ('A', 3)],
 'con_ref_idx0': array([0, 1, 2]),
 'con_hal_idx0': array([113, 114,  84]),
 'config': {
	'contigmap': {'contig_atoms': "{'A1':'C','A2':'N,CA,CB,OG','A3':'NE2,CE1,ND1,CG,CD2'}"},
  	'inference': {
		'input_pdb': '/net/scratch/ahern/se3_diffusion/benchmarks/2023-12-13_02-40-19_sh_benchmark_1_bb_both/input/siteC.pdb',
   		'ligand': 'mu2'
		}
	}
}
```

Run python `./benchmark/pipeline.py --config-name=catalytic_constraints_from_designs outdir=$MY_FOLDER`

This will produce a metrics dataframe: $METRICS_DATAFRAME_PATH

Use $METRICS_DATAFRAME_PATH in the provided analysis notebook `notebooks/analyze_catalytic_constraints.ipynb` to analyze success on the various catalytic constraints.

If you do not have the dependencies to run this notebook in your default kernel, use this sif as a kernel `/net/software/containers/users/dtischer/rf_diffusion_aa.sif` following instructions in https://wiki.ipd.uw.edu/it/digs/apptainer#jupyter

## Running catalytic constraint design + benchmarking
`$REPO_DIR/rf_diffusion/benchmark/pipeline.py --config-name=sh_benchmark_1_tip-true_selfcond-false_seqposition_truefalse_T150`

This will make 50 * 2 [+/- sequence position] * 6 [6 different active site descriptions] = 600 designs = 600 * 8 (MPNN runs/design) = 4800 sequences

All motifs are tip atom motifs for 150 timesteps with no self-conditioning


