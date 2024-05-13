#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'
#
# Breakup a foldseek job on a dir of many pdbs in several chunks, which run as separate jobs.

import os
import sys
import glob
from rf_diffusion.benchmark.util.slurm_tools import array_submit
import hydra
from hydra.core.hydra_config import HydraConfig

script_dir = os.path.dirname(os.path.realpath(__file__))+'/'

def split_list(l, idx):
    '''Split one list into two at the given index.'''
    l1 = l[:idx]
    l2 = l[idx:]
    return l1, l2

@hydra.main(version_base=None, config_path='configs/', config_name='chunkify_foldseek_pdb')
def main(conf: HydraConfig) -> list[int]:
    '''
    ### Expected conf keys ###
    pdb_dir:        Dir of pdbs. Too many to do in one foldseek job.
    chunk:          Number of pdbs to pass to each foldseek job

    slurm:
        J:          Job name
        submit:     False = Do not submit slurm array job, only generate job list. <True, False>
        in_proc:    Run slurm array job on the current node? <True, False>
        keep_logs:  Keep the slurm logs? <True, False>
    '''

    # Make the slurm task file
    job_fn = f'{conf.pdb_dir}/jobs.foldseek.list'
    job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout

    remaining_pdbs = glob.glob(f'{conf.pdb_dir}/*.pdb')
    chunk_number = 0
    while remaining_pdbs:
        chunk_pdbs, remaining_pdbs = split_list(remaining_pdbs, conf.chunk)
        chunk_outdir = f'{conf.pdb_dir}/foldseek_pdb/chunk{chunk_number}'
        print(f'{script_dir}/foldseek_pdb.py --pdbs {" ".join(chunk_pdbs)} --out_dir {chunk_outdir}',
              file=job_list_file)
        chunk_number += 1

    # submit job
    job_ids = []
    if conf.slurm.submit:
        job_list_file.close()
        if conf.slurm.J is not None:
            job_name = conf.slurm.J
        else:
            pre = 'foldseek_pdb_'
            job_name = pre + os.path.basename(conf.pdb_dir.strip('/'))
        
        try:
            job_id, proc = array_submit(job_fn, p='cpu', gres=None, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc)
            if job_id > 0:
                job_ids.append(job_id)
            print(f'Submitted array job {job_id} with {chunk_number} jobs to compute the '
                f'similarity of {len(glob.glob(f"{conf.pdb_dir}/*.pdb"))} designs to the PDB.')
        except Exception as excep:
            if 'No k-mer could be extracted for the database' in str(excep):
                print('WARNING: Some generated protein was too short for foldseek (<14 aa). '
                      'This often occurs when running the pipeline unit test. NBD')
            else:
                sys.exit(excep)

    return job_ids


if __name__ == '__main__':
    main()