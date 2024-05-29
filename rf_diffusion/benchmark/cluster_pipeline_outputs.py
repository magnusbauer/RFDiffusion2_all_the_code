#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'

import os
import sys
import re
import glob

# from rf_diffusion.benchmark import util
from rf_diffusion.benchmark.util.slurm_tools import array_submit
import hydra
from hydra.core.hydra_config import HydraConfig

script_dir = os.path.dirname(os.path.realpath(__file__))+'/'

@hydra.main(version_base=None, config_path='configs/', config_name='cluster_pipeline_outputs')
def main(conf: HydraConfig) -> list[int]:
    '''
    ### Expected conf keys ###
    pipeline_outdir:    Out dir of the pipeline.

    slurm:
        J:              Job name
        submit:         False = Do not submit slurm array job, only generate job list. <True, False>
        in_proc:        Run slurm array job on the current node? <True, False>
        keep_logs:      Keep the slurm logs? <True, False>
    '''

    # ID all conditions
    conditions = {re.findall('cond\d+', pdb_fn)[0] for pdb_fn in glob.glob(f'{conf.pipeline_outdir}/*cond*.pdb')}

    # Make foldseek clustering jobs on all pdbs from the same condition.
    job_fn = f'{conf.pipeline_outdir}/jobs.clustering.list'
    job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
    for condition in conditions:
        cluster_outdir = f'{conf.pipeline_outdir}/foldseek_clustering/{condition}'
        os.makedirs(cluster_outdir, exist_ok=True)
        condition_pdbs = glob.glob(f'{conf.pipeline_outdir}/*{condition}*.pdb')
        print(f'{script_dir}/foldseek_clustering.py --pdbs {" ".join(condition_pdbs)} --out_dir {cluster_outdir}',
               file=job_list_file)

    # submit job
    job_ids = []
    if conf.slurm.submit:
        job_list_file.close()
        if conf.slurm.J is not None:
            job_name = conf.slurm.J
        else:
            pre = 'foldseek_cluster_'
            job_name = pre + os.path.basename(conf.pipeline_outdir.strip('/'))

        try:
            job_id, proc = array_submit(job_fn, p='cpu', gres=None, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc)
            if job_id > 0:
                job_ids.append(job_id)
            print(f'Submitted array job {job_id} with {len(conditions)} jobs to cluster the backbones '
                'from each condition.')
        except Exception as excep:
            if 'No k-mer could be extracted for the database' in str(excep):
                print('WARNING: Some generated protein was too short for foldseek-based clustering (<14 aa). '
                      'This often occurs when running the pipeline unit test. NBD')
            else:
                sys.exit(excep)
                  
    return job_ids

if __name__ == '__main__':
    main()
