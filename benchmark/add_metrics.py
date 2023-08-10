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

@hydra.main(version_base=None, config_path='configs/', config_name='add_metrics')
def main(conf: HydraConfig) -> list[int]:
    '''
    ### Expected conf keys ###
    datadir:        Folder of designs to score.
    filenames:      A path to a list of PDBs to score, rather than scoring everything in datadir.
    chunk:          How many designs to score in each job.
    tmp_pre:        Name prefix of temporary files with lists of designs to score.
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

    if conf.chunk == -1:
        conf.chunk = len(filenames)

    job_ids = []

    # General metrics
    job_fn = conf.datadir + '/jobs.metrics_per_design.list'
    job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
    for i in np.arange(0,len(filenames),conf.chunk):
        tmp_fn = f'{conf.datadir}/{conf.tmp_pre}.metrics_per_design.{i}'
        with open(tmp_fn,'w') as outf:
            for j in np.arange(i,min(i+conf.chunk, len(filenames))):
                print(filenames[j], file=outf)
        print(f'{os.path.join(script_dir, "per_design_metrics.py")} '\
                f'--outcsv {conf.datadir}/metrics/per_design/csv.{i} '\
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
        print(f'Submitted array job {af2_job} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to compute per-design metrics for {len(filenames)} designs')

    return job_ids


