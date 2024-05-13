#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'
#
# Get the structurally most similar matches for a list/dir of pdbs to the PDB.

import argparse
import os
import sys
import subprocess
import pandas as pd

def main():
    # Parse args
    p = argparse.ArgumentParser()
    p.add_argument('--pdbs', nargs='+', help='Space separated list of PDBs to structurally cluster.')
    p.add_argument('--pdb_dir', help='Directory of PDBs to structurally cluster.')
    p.add_argument('--out_dir', default='./', help='Directory for outputs.')

    # Pass through args to foldseek easy-cluster. Use '/software/foldseek/foldseek easy-search -h' for documentation.
    p.add_argument('--format-output', dest='format_output', default='query,target,alntmscore,qcov,tcov')
    p.add_argument('-c', default=0.0)
    p.add_argument('--alignment-type', dest='alignment_type', default=1)
    args = p.parse_args()

    # Make a directory of links to the desired pdbs - Foldseek cannot take a list of pdb.
    if args.pdbs is not None:
        args.pdb_dir = f'{args.out_dir}/query_pdbs'
        os.makedirs(args.pdb_dir, exist_ok=True)
        pdbs_abs_path = [os.path.realpath(x) for x in args.pdbs]
        os.system(f'ln -s {" ".join(pdbs_abs_path)} {args.pdb_dir}')

    # Run foldseek in a subprocess
    FOLDSEEK = '/net/software/foldseek/bin/foldseek'
    TARGET_DB = '/projects/omics/ianh/databases/foldcomp/pdb'
    cmd = (f'{FOLDSEEK} easy-search {args.pdb_dir} {TARGET_DB} {args.out_dir}/aln.m8 {args.out_dir}/foldseek_tmp '
           f'-c {args.c} --alignment-type {args.alignment_type} --format-output {args.format_output}')

    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = proc.stdout.decode()
    print(out)
    if proc.returncode != 0: 
        sys.exit(proc.stderr.decode())

    # Only record best match (in a csv with field names)
    df = pd.read_csv(f'{args.out_dir}/aln.m8', sep='\t', names=args.format_output.split(','))
    idx = df.groupby(by='query')['alntmscore'].idxmax().values
    df_best_match = df.iloc[idx]
    df_best_match.to_csv(f'{args.out_dir}/best_aln.csv', index=False)

if __name__ == '__main__':
    main()


