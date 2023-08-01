#!/net/scratch/ahern/shebangs/shebang_rf_se3_diffusion.sh

import argparse
import os
import sys
import re
import glob
from util.slurm_tools import array_submit

script_dir = os.path.dirname(os.path.realpath(__file__))+'/'

def main():
    # Parse args
    p = argparse.ArgumentParser()
    p.add_argument('--pipeline_outdir', help='Out dir of the pipeline.')
    p.add_argument('--chunk', type=int, default=500, help='# of pdbs to pass to each foldseek job')
    p.add_argument('--no_submit', dest='submit', action="store_false", default=True, help='Do not submit slurm array job, only generate job list.')
    p.add_argument('--no_logs', dest='keep_logs', action="store_false", default=True, help='Don\'t keep slurm logs.')
    p.add_argument('--in_proc', dest='in_proc', action="store_true", default=False, help='Do not submit slurm array job, run in current process.')
    p.add_argument('-J', type=str, help='name of slurm job')
    args = p.parse_args()

    # ID all conditions
    conditions = {re.findall('cond\d+', pdb_fn)[0] for pdb_fn in glob.glob(f'{args.pipeline_outdir}/*cond*pdb')}

    # Make foldseek clustering jobs on all pdbs from the same condition.
    job_fn = f'{args.pipeline_outdir}/jobs.clustering.list'
    job_list_file = open(job_fn, 'w') if args.submit else sys.stdout
    for condition in conditions:
        cluster_outdir = f'{args.pipeline_outdir}/foldseek_clustering/{condition}'
        os.makedirs(cluster_outdir, exist_ok=True)
        condition_pdbs = glob.glob(f'{args.pipeline_outdir}/*{condition}*pdb')
        print(f'{script_dir}/foldseek_clustering.py --pdbs {" ".join(condition_pdbs)} --out_dir {cluster_outdir}',
               file=job_list_file)

    # submit job
    if args.submit:
        job_list_file.close()
        if args.J is not None:
            job_name = args.J
        else:
            pre = 'foldseek_cluster_'
            job_name = pre + os.path.basename(args.pipeline_outdir.strip('/'))

        try:
            cn_job, proc = array_submit(job_fn, p = 'cpu', gres=None, log=args.keep_logs, J=job_name, in_proc=args.in_proc)
        except Exception as excep:
            if 'No k-mer could be extracted for the database' in str(excep):
                print('WARNING: Some generated protein was too short for foldseek (<14 aa). '
                      'This often occurs when running the pipeline unit test. NBD')
                sys.exit(0)
                  
        print(f'Submitted array job {cn_job} with {len(conditions)} jobs to cluster the backbones '
               'from each condition.')

if __name__ == '__main__':
    main()