# Info

You caught us in an early state. Brian C literally just pushed this out to the internet because we've spent far too long cleaning it up for release.

In theory you can get this code to run, but there might be some hiccups until we get this all looking good.


# Setup

## Dependencies
rf_diffusion now includes a pre-commit hook that runs ruff and yapf to autoformat files. If you run into an error, just run "pip install ruff yapf," or otherwise install the two packages. Most existing files are, for now, grandfathered out of autoformatting via inclusion in the .yapf_exclude file. If you really don't want your new files autoformatted, you can add them to that file. Linting and formatting will be applied before each commit. After the initial run, which may take ten seconds or so, formatting will be done incrementally and shouldn't slow you down.

## Paths
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
apptainer exec exec/bakerlab_rf_diffusion_aa.sif pytest --disable-warnings -s -m "not nondeterministic"
```
## A note about running scripts in this repo
Many of the python scripts in this repo are executable. That is, you don't need to do `python some_script.py` or `my_container.sif some_script.py`. The are several environmental variables and flags that need to be set properly for the scripts to run. Rather than asking the user to set these correctly each time (which is error prone), we have a script to do this prep work under the hood for you! 

Any scripts (like `rf_diffusion/benchmark/pipeline.py` and `rf_diffusion/run_inference.py`) that have the shebang line `#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'` can be executed directly. If you need to run a script without that line, you need to do
```
export PYTHONPATH="${PYTHONPATH}:$REPO_DIR"
/usr/bin/apptainer run --nv --slurm --env PYTHONPATH="\$PYTHONPATH:$PYTHONPATH" path/to/rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif path/to/script.py ...
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

If you do not have the dependencies to run this notebook in your default kernel, use this sif as a kernel `rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif` following instructions in https://wiki.ipd.uw.edu/it/digs/apptainer#jupyter

## Running catalytic constraint design + benchmarking
`$REPO_DIR/rf_diffusion/benchmark/pipeline.py --config-name=sh_benchmark_1_tip-true_selfcond-false_seqposition_truefalse_T150`

This will make 50 * 2 [+/- sequence position] * 6 [6 different active site descriptions] = 600 designs = 600 * 8 (MPNN runs/design) = 4800 sequences

All motifs are tip atom motifs for 150 timesteps with no self-conditioning

# Mid-trajectory filters

Sometimes your diffusion goals are hard for the network perform, but easy for you to evaluate. An even better case is where it's easy to tell if the network **is going to get it wrong** very early in the trajectory.

Mid-trajectory filters allow you to detect and restart trajectories that are going poorly.

## Universal filter flags

These flags control the behavior of how your run will progress. `filter.max_steps_per_design` might be the safer of the two since it can give you an upperbound on your runtime.
```
filters:
  max_attempts_per_design: 10 # If filters enabled, for a given design, try at most this many times before giving up
  max_steps_per_design: 100 # If filters enabled, for a given design, take at most this many diffusion steps (cumulative across failures) before giving up
```

These flags control the scorefile that is produced from filters. Typically you'll either use the scorefile to save scores so you know which outputs are best or in preparation to figure out how to filter.

```
inference:
  write_scorefile: True # Write a scorefile if there are scores to write
  scorefile_delimiter: ' ' # ',' implies .csv, ' ' implies .sc
  write_scores_to_trb: True # If scores are present, write to trb
```

## Filter instantiation

To add filters to your runs, you need to add them to `filters.names` and then configure them through `filters.configs`

Here's an example where we'll add a ChainBreak filter at t=20:

```
filters:
  names:
    - ChainBreak
  configs:
    ChainBreak:
      t: 20
      C_N_dist_limit: 1.8
```

If you want to have multiple copies of the same kind of filter. You can use `NewName:ClassName` to rename them in the names.

```
filters:
  names:
    - EarlyBreaks:ChainBreak
  configs:
    EarlyBreaks:
      t: '40,30,25'
      C_N_dist_limit: 2.5
```

In general, all filters have the following default fields:

```
t: # A comma separated list of t steps to activate at
suffix: # A string suffix to add to this value in the scorefile
prefix: # A string prefix to add to this value in the scorefile
verbose: # A bool (default false) as to whether or not this filter should print logging info
```

## Available Filters

This list will almost certainly be out of date at some point. `rf_diffusion/rf_diffusion/filters.py` will always be up-to-date.

### InterchainClashFilter filter

Look for overlapping protein backbones between chains

Configs:
```

chainA -- Which is the first chain we'll look at. None for all
chainB -- Which is the second chain we'll look at. None for all
max_bb_clashes -- How many backbone clashes are acceptable between two chains
clash_dist -- At what distance do we consider CAs to be clashing
use_px0 -- Default True. Use px0 as the structure to look for clashes in
```

Reports:
```
max_clashes -- The most backbone clashes we found between two chains
```

### ChainBreak filter

Finds the largest C->N atom gap in your protein

Configs:
```
C_N_dist_limit -- The limit (in angstroms) at which this filter will fail. (1.7 is a pretty good choice)
monitor_chains -- A string of comma separated numbers (starting at 0) of which chains to monitor. Binder design would use '0'
use_px0 -- Default True. Use px0 as the structure to look for chainbreak in
```

Reports:
```
max_chain_gap -- The largest C->N atom gap in the chains we are monitoring
```

### BBGPSatisfaction filter

Determines the satisfaction level of your Backbone Guideposts. Currently assumes all guidepost are backbone.

Configs:
```
gp_max_error_cut -- The limit (in angstroms) of the worst CA-CA guidepost mismatch that's allowable
gp_rmsd_cut -- The limit (in angstroms) of the RMSD of all CA-CA guidepost matches that's allowable
use_px0 -- Default True. Use px0 as the structure to look for chainbreak in
```

Reports:
```
gp_max_error -- The worst CA-CA guidepost mismatch
gp_rmsd -- The RMSD of all CA-CA guidepost matches
```

# Benchmarking guide

The ODE solver returns the guideposted, atomized protein.

The backbone is idealized.

The protein is deatomized.

The guideposts are placed.  A greedy search matches each guidepost to its nearest remaining C-alpha.
If `inference.guidepost_xyz_as_design_bb == True`, then the guidepost coordinates overwrite the matched backbone.  Otherwise only the sidechain (including C-beta) coordinates of the guidepost are used.

If `inference.idealize_sidechain_outputs == True` then all atomized sidechains are idealized.  This amounts to finding the set of torsions angles that minimizes the RMSD between the unidealized residue and the residue reconstructed from those torsion angles.  Note: these torsions are the full rf2aa torsion set which includes not only torsions but also bends and twists e.g. C-Beta bend which can adopt values which would be of higher-strain than that seen in nature.

The protein at this point has sequence and structure for the motif regions but only backbone (N,Ca,C,O,C-Beta) coordinates for diffused residues (as well as any non-protein components e.g. small molecules)

Sequence is fit using LigandMPNN in a ligand-aware, motif-rotamer-aware mode.  LigandMPNN also performs packing.  LigandMPNN attempts to keep the motif rotamers unchanged, however the pack uses a more conservative set of torsions than rf2aa (i.e. fewer DoF) to pack the rotamers and thus there is often some deviation between the rf2aa-idealized and ligandmpnn-idealized motif rotamers.  The idealization gap between the diffusion-output rotamer set and the rf2aa-idealized rotamer set can be found with metrics key: `metrics.IdealizedResidueRMSD.rmsd_constellation`.  The corresponding gap between the rf2aa-idealized (or not idealized if `inference.idealize_sidechain_outputs == False`) rotamer set and the ligandmpnn-idealized rotamer set can be found with metrics key: `motif_ideality_diff`.

Motif recapitulation metrics:

The following metrics follow a formula:

contig_rmsd_a_b_s

a,b: the proteins being compared:
	- des: The MPNN packed protein
	- pred: The AF2 prediction
	- ref: The input PDB
With the caveat that 'ref' is always omitted from the name.

s: the comparison type:
	- '': backbone (N, Ca, C)
	- 'c_alpha': Ca
	- 'full_atom': All heavy atoms
	- 'motif_atom': Only motif heavy atoms


# Running the enzyme benchmark

We crawled M-CSA for 41 enzymes where all reactants and products are present to create this benchmark.  Only positon-agnostic tip atoms and partial ligand positions are provided to the network.

100 designs for each case are created.

Run it with:

`./benchmark/pipeline.py --config-name=enzyme_bench_n41`

# Debugging

## pipeline.py
If your outdir is `/a/b/c/` then slurm logs appear at: `/a/b/SLURMJOBID_SLURMJOBARRAYINDEX_jobtype.log`

# PPI

You can compare protein-protein binder design between RFDAM and RFDiffusion by generating length 100 binders against the 5 binder design benchmark cases described in the RFDiffusion paper, by running this command:

`./benchmark/pipeline.py -cn ppi_comparison_pilot`

To visualize the trajectories created during the "sweep" step in PYMOL:
`./dev/show_bench.py --clear=1 'YOUR_OUTPUT_DIR_HERE/*.trb' --key=name --ppi=1 --des=0`

To visualize the designs as cartoons once the MPNN step is complete in PYMOL:
`./dev/show_bench.py --clear=1 'YOUR_OUTPUT_DIR_HERE/*.trb' --key=name --ppi=1 --mpnn_packed=1 --des=0 --cartoon=1 --structs='{}'`

## CA RFdiffusion

Example shell scripts for running inference with CA RFdiffusion (diffusion + refinement) are in
```
# inference
rf_diffusion/examples/inference/ca_rfd_diffuse.sh
rf_diffusion/examples/inference/ca_rfd_refine.sh
# training
rf_diffusion/examples/train/train_ca_rfd.sh
```
Below is reccomended reading for better understanding inference. 

### CA RFdiffusion -- inference
The following are some general notes and things to look out for when running inference with CA RFdiffusion.

Here's an example .sh script for submitting a CA RFdiffusion diffusion run:
```
output_pref="./experiments/out"

apptainer exec --nv ./exec/bakerlab_rf_diffusion_aa.sif python run_inference.py \
    --config-name="RFdiffusion_CA_inference" \
    inference.output_prefix=${output_pref} \
    inference.num_designs=30
```

Lets look at the `RFdiffusion_CA_inference.yaml` file. 

We first see some stuff related to running diffusion:
```
diffuser: 
  type: 'legacy'
  T: 50
  
  r3: 
    min_b: 0.01
    max_b: 0.07
    coordinate_scaling: 0.25
    T: ${..T}
    schedule_kwargs: {}
    var_scale: 0.05
    noise_scale: 0.05 
```

Things you can touch here are `T` (although 50 is reccomended) and `noise_scale` (Linna An showed 0.05 is best for compute efficiency). 

There's nothing fancy about contigs strings with CA RFdiffusion, but we will reference this one below:
```
contigmap:  
  contigs: ['30,A1-4,40,A5-5,40,A6-6,40,A7-7,40']
```

Now, look at these inference parameters:
```
inference: 
  output_prefix: ./experiments/caRFD_test
  ckpt_path: '/mnt/projects/ml/ca_rfd/BFF_7_w_new_conf.pt'
  input_pdb: ./test_data/siteC.pdb
  str_self_cond: 1
  ij_visible: 'abcde' # e is the ligand
  length: 90-125
  ligand: mu2
  write_trajectory: true
  recenter_xt: true 
  num_designs: 15
  cautious: true 
  guidepost_xyz_as_design: false
```

The most important flag to understand here is `ij_visible`. `ij_visible` specifies which motif chunks within the contigs string are going to be constrained with respect to each other rigidly. The syntax for writing `ij_visible` is to group together motif chunks you want constrained rigidly together, and separate groups with dashes. In the above example, all motif chunks including the ligand are constrained with respect to each other, which is why there's only one group and no dashes. If you wanted the first two motif chunks constrained w.r.t each other, and then the next two + the ligand constrained rigidly together, your flag would be `ij_visible: 'ab-cde'`. If you want each chunk free to move w.r.t to all the other chunks, you could do `ij_visible: 'a-b-c-d-e'`. If you're curious about pushing this to the limits, ask Alexis Courbet about constraining 26+ individual amino acid chunks for funnels. 

#### More on ij_visible letters
Each letter in `ij_visible` corresponds to a motif chunk in the contigs string. Specifically, the left-most contig chunk in the contigs string corresponds to letter `a` (in this example, `A1-4` is chunk `a`). The next chunk over corresponds to `b` (`A5-5` in this example). The `ij_visible` letters corresponding to other chunks in the contigs string just increases one letter at a time in the alphabet for each position further in the contigs string. **IF YOU HAVE A LIGAND IN YOUR DESIGN** (e.g., enzymes, small molecule binding, etc), you should include an extra letter in `ij_visible` which corresponds to the ligand, and that letter should be the letter in the alphabet just after the letter corresponding to the last contig chunk in the contigs string. In the example here, the ligand letter is `e` because there are 4 contig chunks in the contigs string corresponding to `a,b,c` and `d`. 

### CA RFdiffusion -- refinement
After you run the diffusion step of CA RFdiffusion, you have created a good "CA trace", in which only the CA positions matter and the N,C,O positions are nonsense. So, you run the "refinement" step. 

In this step, you just need to specify; 
1. The path to the .pdb file you want to refine 
2. The number of refinement outputs you'd like to do per input. 2 is reccomended, but you can do as many as you want. They are really quick to run, and the outputs differ slightly (1-3 RMSD) from each other.
3. The ligand (if you had one).

See `test_ca_rfd_refinement` config yaml. Here is an example submission in a bash script:
```
#bin/bash

pdb='path/to/some_pdb_with_trb_file_next_to_it.pdb'
CKPT='/mnt/projects/ml/ca_rfd/BFF_3_w_new_conf.pt'

apptainer exec --nv ./exec/bakerlab_rf_diffusion_aa.sif python run_inference.py \
    --config-name=test_ca_rfd_refinement \
    inference.num_designs=2 \
    inference.input_pdb=$pdb \
    inference.ligand='mu2' \
    inference.ckpt_path=$CKPT
```
