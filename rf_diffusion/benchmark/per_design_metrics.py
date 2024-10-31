#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'

import os
import sys

# Hack for autobenching
PKG_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
SE3_DIR = os.path.join(PKG_DIR, 'lib/se3_flow_matching')
sys.path.append(SE3_DIR)

import pandas as pd
import fire
from tqdm import tqdm

import rf_diffusion.dev.analyze
import analysis.metrics
from rf_diffusion.dev import benchmark as bm


def main(pdb_names_file, outcsv=None):
    with open(pdb_names_file, 'r') as fh:
        pdbs = [pdb.strip() for pdb in fh.readlines()]
    
    df = get_metrics(pdbs)

    print(f'Outputting computed metrics to {outcsv}')
    os.makedirs(os.path.dirname(outcsv), exist_ok=True)
    df.to_csv(outcsv)


def get_metrics(pdbs):
    records = []
    for pdb in tqdm(pdbs):
        record = {}
        row = rf_diffusion.dev.analyze.make_row_from_traj(pdb[:-4])
        record['name'] = row['name']

        traj_metrics = bm.get_inference_metrics_base(bm.get_trb_path(row), regenerate_cache=False)
        traj_t0_metrics = traj_metrics[traj_metrics.t==traj_metrics.t.min()]
        assert len(traj_t0_metrics) == 1
        traj_t0_metrics = traj_t0_metrics.iloc[0].to_dict()
        record.update(traj_t0_metrics)

        # Ligand distance
        if row['inference.ligand']:
            for af2, c_alpha in [
                (False, True),
                # (True, True),
                # (True, True)
            ]:
                dgram = rf_diffusion.dev.analyze.get_dist_to_ligand(row, af2=af2, c_alpha=c_alpha) # [P, L]
                maybe_af2 = 'af2' if af2 else 'des'
                maybe_c_alpha = 'c-alpha' if c_alpha else 'all-atom'
                record[f'ligand_dist_{maybe_af2}_{maybe_c_alpha}'] = dgram.min(-1)[0].tolist() # [P]
                record[f'ligand_dist_{maybe_af2}_{maybe_c_alpha}_min'] = dgram.min().item()

        # Secondary structure and radius of gyration
        record.update(analysis.metrics.calc_mdtraj_metrics(pdb))
        # Broken due to residue indexing.
        # record['rigid_loss'] = rigid_loss(row)

        records.append(record)

    df = pd.DataFrame.from_records(records)
    return df

if __name__ == '__main__':
    fire.Fire(main)
