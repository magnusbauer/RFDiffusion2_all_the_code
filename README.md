# Setup
git clone git@github.com:baker-laboratory/rf_diffusion.git PKG_DIR

cd rf_diffusion

git submodule init

git submodule update --init

## Temporary hack to set the python path
export PYTHONPATH="${PYTHONPATH}:PKG_DIR"

cd rf_diffusion

## Verify tests pass
apptainer exec /software/containers/users/dtischer/rf_diffusion_aa.sif pytest --disable-warnings -s -m "not nondeterministic"


# Simple inference pipeline run
## Running inference
To run a demo of some of the inference capabilities, including enzyme design from tip atoms, enzyme design from tip atoms of unknown sequence position, ligand binder design, traditional contiguous motif scaffolding, and molecular glue design (binder to protein:small_molecule complex).  (See `PKG_DIR/rf_diffusion/benchmark/demo.json` for how these tasks are declared)

cd PKG DIR

`/software/containers/users/dtischer/rf_diffusion_aa.sif rf_diffusion/benchmark/pipeline.py --config-name=demo_design_only`

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
`/software/containers/users/dtischer/rf_diffusion_aa.sif rf_diffusion/dev/show_bench.py --clear=True 'OUTDIR/*.pdb' --pymol_url=http://HOSTNAME:9123`

You should see multiple designs (such as this enzyme design) render in your pymol session:
![retroaldolase_demo](images/demo_output_retroaldolase.png)

To render some of the nice colors, you may need to add the files in `pymol_config` to your `.pymolrc`

## Running inference (OUTSIDE OF DIGS)
To run a simple pipeline with no mpnn/scoring for the tip atom case:

`/software/containers/users/dtischer/rf_diffusion_aa.sif rf_diffusion/benchmark/pipeline.py --config-name=retroaldolase_demo_nodigs`

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

