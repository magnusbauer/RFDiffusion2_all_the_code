#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'
#
# Takes a folder of pdb & trb files, generates list of AF2 prediction & scoring
# jobs on batches of those designs, and optionally submits slurm array job and
# outputs job ID
# 

import sys
import os
import glob
import copy
import pandas as pd
import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig

script_dir = os.path.dirname(os.path.realpath(__file__))
from rf_diffusion.benchmark.util import slurm_tools

@hydra.main(version_base=None, config_path='configs/', config_name='score_designs')
def main(conf: HydraConfig) -> list[int]:
    '''
    ### Expected conf keys ###
    datadir:        Folder of designs to score.
    trb_dir:        Folder containing .trb files (if not same as datadir).
    filenames:      A path to a list of PDBs to score, rather than scoring everything in datadir.
    chunk:          How many designs to score in each job.
    tmp_pre:        Name prefix of temporary files with lists of designs to score.
    pipeline:       Pipeline mode - submit the next script to slurm with a dependency on jobs from this script. <True, False>
    run:            Comma-separated (no whitespace) list of scoring scripts to run (e.g. "af2,pyrosetta"). <"af2", pyrosetta", "chemnet", "rosettalig">

    slurm:
        J:          Job name
        p:          Partition
        gres:       Gres specification
        submit:     False = Do not submit slurm array job, only generate job list. <True, False>
        in_proc:    Run slurm array job on the current node? <True, False>
        keep_logs:  Keep the slurm logs? <True, False>
    '''
    if conf.filenames:
        filenames = [l.strip() for l in open(conf.filenames).readlines()]
    else:
        filenames = sorted(glob.glob(conf.datadir+'/*.pdb'))
    if len(filenames)==0: sys.exit('No pdbs to score. Exiting.')

    conf.run = conf.run.split(',')

    if conf.chunk == -1:
        conf.chunk = len(filenames)

    job_ids = []

    if 'protein_metrics' in conf.run:
        # General metrics
        job_fn = conf.datadir + '/jobs.score.protein_metrics.list'
        job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
        for i in np.arange(0,len(filenames),conf.chunk):
            tmp_fn = f'{conf.datadir}/{conf.tmp_pre}.protein_metrics.{i}'
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+conf.chunk, len(filenames))):
                    print(filenames[j], file=outf)
            print(f'/usr/bin/apptainer run --nv --bind /software/mlfold/alphafold:/software/mlfold/alphafold --bind /net/databases/alphafold/params/params_model_4_ptm.npz:/software/mlfold/alphafold-data/params/params_model_4_ptm.npz /software/containers/mlfold.sif {script_dir}/util/af2_metrics.py --use_ptm '\
                  f'--outcsv {conf.datadir}/protein_metrics.csv.{i} '\
                  f'--trb_dir {conf.trb_dir} '\
                  f'{tmp_fn}', file=job_list_file)

        # submit job
        if conf.slurm.submit: 
            job_list_file.close()
            if conf.slurm.J is not None:
                job_name = conf.slurm.J 
            else:
                job_name = 'af2_'+os.path.basename(conf.datadir.strip('/'))
            af2_job, proc = slurm_tools.array_submit(job_fn, p = conf.slurm.p, gres=None if conf.slurm.p=='cpu' else conf.slurm.gres, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc)
            if af2_job > 0:
                job_ids.append(af2_job)
            print(f'Submitted array job {af2_job} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to AF2-predict {len(filenames)} designs')


    # AF2 predictions
    if 'af2' in conf.run:
        job_fn = conf.datadir + '/jobs.score.af2.list'
        job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
        already_ran = {}
        for i in np.arange(0,len(filenames),conf.chunk):
            tmp_fn = f'{conf.datadir}/{conf.tmp_pre}.{i}'
            input_filenames = []
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+conf.chunk, len(filenames))):
                    input_filenames.append(filenames[j])
                    print(filenames[j], file=outf)
            outcsv = f'{conf.datadir}/af2_metrics.csv.{i}'
            job = (f'/usr/bin/apptainer run --nv --bind /software/mlfold/alphafold:/software/mlfold/alphafold --bind /net/databases/alphafold/params/params_model_4_ptm.npz:/software/mlfold/alphafold-data/params/params_model_4_ptm.npz /software/containers/mlfold.sif {script_dir}/util/af2_metrics.py --use_ptm '\
                  f'--outcsv {outcsv} '\
                  f'--trb_dir {conf.trb_dir} '\
                  f'{tmp_fn}')
            print(job, file=job_list_file)
            def outputs_exist(outcsv=outcsv, input_filenames=input_filenames):
                if not os.path.exists(outcsv):
                    return False
                df = pd.read_csv(outcsv)
                name_set = set(df['name'])
                for i, input_filename in enumerate(input_filenames):
                    name = os.path.basename(input_filename).removesuffix('.pdb')
                    if name not in name_set:
                        return False
                return True

            already_ran[job] = copy.deepcopy(outputs_exist)

        # submit job
        if conf.slurm.submit: 
            job_list_file.close()
            if conf.slurm.J is not None:
                job_name = conf.slurm.J 
            else:
                job_name = 'af2_'+os.path.basename(conf.datadir.strip('/'))
            af2_job, proc = slurm_tools.array_submit(job_fn, p = conf.slurm.p, gres=None if conf.slurm.p=='cpu' else conf.slurm.gres, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc, already_ran=already_ran)
            if af2_job > 0:
                job_ids.append(af2_job)
            print(f'Submitted array job {af2_job} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to AF2-predict {len(filenames)} designs')

    # Rosetta metrics
    if 'pyrosetta' in conf.run:
        # pyrosetta metrics (rog, SS)
        job_fn = conf.datadir + '/jobs.score.pyr.list'
        job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
        for i in np.arange(0,len(filenames),conf.chunk):
            tmp_fn = f'{conf.datadir}/{conf.tmp_pre}.pyr.{i}'
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+conf.chunk, len(filenames))):
                    print(filenames[j], file=outf)
            print(f'apptainer exec /software/containers/pyrosetta.sif python {script_dir}/util/pyrosetta_metrics.py '\
                  f'--outcsv {conf.datadir}/pyrosetta_metrics.csv.{i} '\
                  f'{tmp_fn}', file=job_list_file)

        # submit job
        if conf.slurm.submit: 
            job_list_file.close()
            if conf.slurm.J is not None:
                job_name = conf.slurm.J 
            else:
                job_name = 'pyr_'+os.path.basename(conf.datadir.strip('/'))
            pyr_job, proc = slurm_tools.array_submit(job_fn, p = 'cpu', gres=None, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc)
            if pyr_job > 0:
                job_ids.append(pyr_job)
            print(f'Submitted array job {pyr_job} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to get PyRosetta metrics for {len(filenames)} designs')

    # Ligand metrics (chemnet)
    if 'chemnet' in conf.run:
        job_fn = conf.datadir + '/jobs.score.chemnet.list'
        job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
        chemnet_script = '/net/databases/lab/chemnet/arch.22-10-28/DALigandDock_v03.py'
        for i in range(0, len(filenames), conf.chunk):
            tmp_fn = f'{conf.datadir}/{conf.tmp_pre}.chemnet.{i}'
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+conf.chunk, len(filenames))):
                    print(filenames[j], file=outf)
            print(f'apptainer exec --nv /software/containers/users/aivan/dlchem.sif python {chemnet_script} '\
                  f'-n 10 --ifile {tmp_fn} '\
                  f'--odir {conf.datadir}/chemnet/ '\
                  f'--ocsv {conf.datadir}/chemnet_scores.csv.{i} ',
                  file=job_list_file)

        # submit job
        if conf.slurm.submit:
            job_list_file.close()
            if conf.slurm.J is not None:
                job_name = conf.slurm.J
            else:
                pre = 'chemnet_'
                job_name = pre + os.path.basename(conf.datadir.strip('/')) 
            cn_job, proc = slurm_tools.array_submit(job_fn, p = conf.slurm.p, gres=None if conf.slurm.p=='cpu' else conf.slurm.gres, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc)
            if cn_job > 0:
                job_ids.append(cn_job)
            print(f'Submitted array job {cn_job} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to ChemNet-predict {len(filenames)} designs')

    # Ligand metrics (rosetta)
    if False:  #'rosettalig' in conf.run: No current sif file has pyrosetta and pytorch.
        job_fn = conf.datadir + '/jobs.score.rosettalig.list'
        job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
        rosettalig_script = script_dir+'/util/rosetta_ligand_metrics.py'
        for i in range(0, len(filenames), conf.chunk):
            tmp_fn = f'{conf.datadir}/{conf.tmp_pre}.rosettalig.{i}'
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+conf.chunk, len(filenames))):
                    print(filenames[j], file=outf)
            print(f'apptainer exec /software/containers/mpnn_binder_design.sif python {rosettalig_script} '\
                  f'--list {tmp_fn} '\
                  f'--outdir {conf.datadir}/rosettalig/ '\
                  f'--outcsv {conf.datadir}/rosettalig_scores.csv.{i} ',
                  file=job_list_file)

        # submit job
        if conf.slurm.submit:
            job_list_file.close()
            if conf.slurm.J is not None:
                job_name = conf.slurm.J
            else:
                pre = 'rosetta_lig_'
                job_name = pre + os.path.basename(conf.datadir.strip('/')) 
            lig_job, proc = slurm_tools.array_submit(job_fn, p = 'cpu', gres=None, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc)
            if lig_job > 0:
                job_ids.append(lig_job)
            print(f'Submitted array job {lig_job} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to compute Rosetta ligand metrics on {len(filenames)} designs')

    if 'rosetta_gen_ff' in conf.run:
        job_fn = conf.datadir + '/jobs.score.rosetta_gen_ff.list'
        job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
        rosettalig_script = script_dir+'/util/rosetta_gen_ff.py'
        for i in range(0, len(filenames), conf.chunk):
            tmp_fn = f'{conf.datadir}/{conf.tmp_pre}.rosetta_gen_ff.{i}'
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+conf.chunk, len(filenames))):
                    print(filenames[j], file=outf)
            print(f'apptainer exec --bind /databases:/databases /software/containers/users/ahern/atom_diff_4.1.sif python -u {rosettalig_script} '\
                  f'--list {tmp_fn} '\
                  f'--trb_dir {conf.trb_dir} '\
                  f'--outdir {conf.datadir}/rosetta_gen_ff/ '\
                  f'--outcsv {conf.datadir}/rosetta_gen_ff.csv.{i} ',
                  file=job_list_file)

        # submit job
        if conf.slurm.submit:
            job_list_file.close()
            if conf.slurm.J is not None:
                job_name = conf.slurm.J
            else:
                pre = 'rosetta_gen_ff_'
                job_name = pre + os.path.basename(conf.datadir.strip('/')) 
            lig_job, proc = slurm_tools.array_submit(job_fn, p = 'cpu', gres=None, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc)
            if lig_job > 0:
                job_ids.append(lig_job)
            print(f'Submitted array job {lig_job} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to compute Rosetta gen ff metrics on {len(filenames)} designs')

    from collections import defaultdict
    if 'chai1' in conf.run:
        job_fn = conf.datadir + '/jobs.score.chai1.list'
        job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
        already_ran = {}
        filenames_by_i = defaultdict(list)
        for i in np.arange(0,len(filenames),conf.chunk):
            tmp_fn = f'{conf.datadir}/{conf.tmp_pre}.chai1.{i}'
            input_filenames = []
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+conf.chunk, len(filenames))):
                    input_filenames.append(filenames[j])
                    print(filenames[j], file=outf)
                    filenames_by_i[i].append(filenames[j])
            outdir = f'{conf.datadir}/chai1/out'
            os.makedirs(outdir, exist_ok=True)

            # Resolve chai1 path, e.g. /home/ahern/reclone/rf_diffusion_dev/lib/chai/predict.py
            chai1_script = os.path.join(script_dir, '../../lib/chai/predict.py')
            job = (f'/usr/bin/apptainer run --nv --bind /net/software/lab/chai:/net/software/lab/chai /net/software/lab/chai/chai_apptainer/chai.sif {chai1_script} '\
                  f'--output_dir {outdir} '\
                  f'--pdb_paths_file {tmp_fn}\
                  --allow_ccd_pdb_mismatch')
            print(job, file=job_list_file)
            def outputs_exist(input_filenames=filenames_by_i[i]):
                def expected_outputs(input_pdb):
                    name = os.path.basename(input_pdb).removesuffix('.pdb')
                    return [
                        f'{outdir}/pred.{name}_model_idx_{i}.pdb'
                        for i in range(5)
                    ]
                for input_pdb in input_filenames:
                    for expected in expected_outputs(input_pdb):
                        if not os.path.exists(expected):
                            return False
                return True

            already_ran[job] = copy.deepcopy(outputs_exist)

        # submit job
        if conf.slurm.submit: 
            job_list_file.close()
            if conf.slurm.J is not None:
                job_name = conf.slurm.J 
            else:
                job_name = 'chai1_'+os.path.basename(conf.datadir.strip('/'))
            chai1_job, proc = slurm_tools.array_submit(job_fn, p = conf.chai1.slurm.p, gres=None if conf.chai1.slurm.p=='cpu' else conf.chai1.slurm.gres, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc, already_ran=already_ran, mem=conf.chai1.slurm.mem)
            if chai1_job > 0:
                job_ids.append(chai1_job)
            print(f'Submitted array job {chai1_job} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to chai1-predict {len(filenames)} designs')

    return job_ids

if __name__ == "__main__":
    main()
