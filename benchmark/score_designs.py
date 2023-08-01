#!/net/scratch/ahern/shebangs/shebang_rf_se3_diffusion.sh
#
# Takes a folder of pdb & trb files, generates list of AF2 prediction & scoring
# jobs on batches of those designs, and optionally submits slurm array job and
# outputs job ID
# 

import sys, os, argparse, itertools, json, glob
import numpy as np
from icecream import ic
import hydra
from hydra.core.hydra_config import HydraConfig

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, 'util'))
import slurm_tools

@hydra.main(version_base=None, config_path='configs/', config_name='score_designs')
def main(conf: HydraConfig) -> None:
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

    # AF2 predictions
    if 'af2' in conf.run:
        job_fn = conf.datadir + '/jobs.score.af2.list'
        job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
        for i in np.arange(0,len(filenames),conf.chunk):
            tmp_fn = f'{conf.datadir}/{conf.tmp_pre}.{i}'
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+conf.chunk, len(filenames))):
                    print(filenames[j], file=outf)
            print(f'/usr/bin/apptainer run --nv --bind /software/mlfold/alphafold:/software/mlfold/alphafold --bind /net/databases/alphafold/params/params_model_4_ptm.npz:/software/mlfold/alphafold-data/params/params_model_4_ptm.npz /software/containers/mlfold.sif {script_dir}/util/af2_metrics.py --use_ptm '\
                  f'--outcsv {args.datadir}/af2_metrics.csv.{i} '\
                  f'--trb_dir {args.trb_dir} '\
                  f'{tmp_fn}', file=job_list_file)

        # submit job
        if conf.slurm.submit: 
            job_list_file.close()
            if conf.slurm.J is not None:
                job_name = conf.slurm.J 
            else:
                job_name = 'af2_'+os.path.basename(conf.datadir.strip('/'))
            af2_job, proc = slurm_tools.array_submit(job_fn, p = conf.slurm.p, gres=None if conf.slurm.p=='cpu' else conf.slurm.gres, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc)
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
            print(f'apptainer exec /software/containers/pyrosetta.sif python {rosettalig_script} '\
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
            cn_job, proc = slurm_tools.array_submit(job_fn, p = 'cpu', gres=None, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc)
            print(f'Submitted array job {cn_job} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to compute Rosetta ligand metrics on {len(filenames)} designs')


if __name__ == "__main__":
    main()
