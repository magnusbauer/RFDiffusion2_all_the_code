#!/net/software/containers/users/dtischer/shebang_rf_se3_diffusion.sh

import argparse
import os
import re
import glob
from util.slurm_tools import array_submit

script_dir = os.path.dirname(os.path.realpath(__file__))+'/'

def main():
    # Parse args
    p = argparse.ArgumentParser()
    p.add_argument('--pipeline_outdir', help='Out dir of the pipeline.')
    args = p.parse_args()

    # ID all conditions
    conditions = {re.findall('cond\d+', pdb_fn)[0] for pdb_fn in glob.glob(f'{args.pipeline_outdir}/*cond*pdb')}

    # Make foldseek clustering jobs on all pdbs from the same condition.
    # TODO: Follow array submittion pattern in score_designs.py
    job_list_fn = f'{args.pipeline_outdir}/jobs.clustering.list'
    with open(job_list_fn, 'w') as f:
        for condition in conditions:
            cluster_outdir = f'{args.pipeline_outdir}/foldseek_clustering/{condition}'
            os.makedirs(cluster_outdir, exist_ok=True)
            condition_pdbs = glob.glob(f'{args.pipeline_outdir}/*{condition}*pdb')
            cmd = f'{script_dir}/foldseek_clustering.py --pdbs {" ".join(condition_pdbs)} --out_dir {cluster_outdir} \n'
            f.write(cmd)

    array_submit(job_list_fn, p='cpu', gres=None, J='foldseek_clustering')

if __name__ == '__main__':
    main()