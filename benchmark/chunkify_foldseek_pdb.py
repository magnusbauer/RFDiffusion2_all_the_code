#!/net/scratch/ahern/shebangs/shebang_rf_se3_diffusion.sh
#
# Breakup a foldseek job on a dir of many pdbs in several chunks, which run as separate jobs.

import argparse
import os
import sys
import glob
from util.slurm_tools import array_submit

script_dir = os.path.dirname(os.path.realpath(__file__))+'/'

def split_list(l, idx):
    '''Split one list into two at the given index.'''
    l1 = l[:idx]
    l2 = l[idx:]
    return l1, l2

def main():
    # Parse args
    p = argparse.ArgumentParser()
    p.add_argument('--pdb_dir', help='Dir of pdbs. Too many to do in one foldseek job.')
    p.add_argument('--chunk', type=int, default=500, help='# of pdbs to pass to each foldseek job')
    p.add_argument('--no_submit', dest='submit', action="store_false", default=True, help='Do not submit slurm array job, only generate job list.')
    p.add_argument('--no_logs', dest='keep_logs', action="store_false", default=True, help='Don\'t keep slurm logs.')
    p.add_argument('--in_proc', dest='in_proc', action="store_true", default=False, help='Do not submit slurm array job, run in current process.')
    p.add_argument('-J', type=str, help='name of slurm job')

    args = p.parse_args()

    # Make the slurm task file
    job_fn = f'{args.pdb_dir}/jobs.foldseek.list'
    job_list_file = open(job_fn, 'w') if args.submit else sys.stdout

    remaining_pdbs = glob.glob(f'{args.pdb_dir}/*pdb')
    chunk_number = 0
    while remaining_pdbs:
        chunk_pdbs, remaining_pdbs = split_list(remaining_pdbs, args.chunk)
        chunk_outdir = f'{args.pdb_dir}/foldseek_pdb/chunk{chunk_number}'
        print(f'{script_dir}/foldseek_pdb.py --pdbs {" ".join(chunk_pdbs)} --out_dir {chunk_outdir}',
              file=job_list_file)
        chunk_number += 1

    # submit job
    if args.submit:
        job_list_file.close()
        if args.J is not None:
            job_name = args.J
        else:
            pre = 'foldseek_pdb_'
            job_name = pre + os.path.basename(args.pdb_dir.strip('/'))
        
        try:
            cn_job, proc = array_submit(job_fn, p = 'cpu', gres=None, log=args.keep_logs, J=job_name, in_proc=args.in_proc)
        except Exception as excep:
            if 'No k-mer could be extracted for the database' in str(excep):
                print('WARNING: Some generated protein was too short for foldseek (<14 aa). '
                      'This often occurs when running the pipeline unit test. NBD')
                sys.exit(0)

        print(f'Submitted array job {cn_job} with {chunk_number} jobs to compute the '
              f'similarity of {len(glob.glob(f"{args.pdb_dir}/*pdb"))} designs to the PDB.')

if __name__ == '__main__':
    main()