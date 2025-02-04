#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'
"""
Script to run geometry from per_sequence_metrics.py and save the results to a csv file.
"""

from rf_diffusion.benchmark.per_sequence_metrics import geometry
import fire
import os 
import glob
import pandas as pd
import tqdm

def compute_geometry_df(pdbs):
    csv = []
    for pdb in tqdm.tqdm(pdbs):
        o = geometry(pdb)
        o['pdb_path'] = pdb
        csv.append(o)
    df = pd.DataFrame.from_records(csv)
    return df

def main(input_dir: str, output_csv: str = None):
    if output_csv is None:
        output_csv = input_dir + '/geometry_metrics.csv'
    output_csv = os.path.abspath(output_csv)
    input_dir = os.path.abspath(input_dir)

    pdbs = glob.glob(input_dir + '/*.pdb')
    if len(pdbs) == 0:
        raise ValueError(f"No pdb files found in {input_dir}")
    print('Found', len(pdbs), 'pdb files in', input_dir)

    df = compute_geometry_df(pdbs)

    df.to_csv(output_csv, index=False)
    print('Wrote to', output_csv)
    print('Done.')


if __name__ == '__main__':
    fire.Fire(main)