#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'

import argparse
import os
import sys
import subprocess

def main():
    # Parse args
    p = argparse.ArgumentParser()
    p.add_argument('--pdbs', nargs='+', help='Space separated list of PDBs to structurally cluster.')
    p.add_argument('--pdb_dir', help='Directory of PDBs to structurally cluster.')
    p.add_argument('--out_dir', default='./', help='Directory for outputs.')
    p.add_argument('--cautious', action='store_true', default=False, help='If the expected output already exists, do no recreate it')

    # Pass through args to foldseek easy-cluster. Use '/software/foldseek/foldseek easy-cluster -h' for documentation.
    p.add_argument('--tmscore-threshold', dest='tmscore_threshold', default=0.5)
    p.add_argument('-c', default=0.5)
    args = p.parse_args()

    # Does the output exist?
    if args.cautious and os.path.exists(f'{args.out_dir}/foldseek_clustering.tsv'):
        print('WARNING: Clustering outputs ({args.out_dir}/foldseek_clustering.tsv) already exist. Skipping foldseek clustering.')
        return

    # Make a directory of links to the desired pdbs - Foldseek cannot take a list of pdb.
    if args.pdbs is not None:
        args.pdb_dir = f'{args.out_dir}/query_pdbs'
        os.makedirs(args.pdb_dir, exist_ok=True)

        pdbs_abs_path = [os.path.realpath(x) for x in args.pdbs]
        os.system(f'ln -s {" ".join(pdbs_abs_path)} {args.pdb_dir}')

    # Run foldseek in a subprocess
    FOLDSEEK = '/net/software/foldseek/bin/foldseek'
    cmd = (f'{FOLDSEEK} easy-cluster {args.pdb_dir} {args.out_dir}/foldseek_clustering {args.out_dir}/foldseek_tmp '
           f'-c {args.c} --alignment-type 1 --tmscore-threshold {args.tmscore_threshold}')

    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = proc.stdout.decode()
    print(out)
    if proc.returncode != 0: 
        sys.exit(proc.stderr.decode())

if __name__ == '__main__':
    main()
