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
import torch

import rf_diffusion.dev.analyze
import analysis.metrics
from rf_diffusion import aa_model
from rf_diffusion import atomize
from rf_diffusion.chemical import ChemicalData as ChemData
from rf_diffusion import bond_geometry
from rf_diffusion.dev import benchmark as bm


def main(pdb_names_file, outcsv=None):
    with open(pdb_names_file, 'r') as fh:
        pdbs = [pdb.strip() for pdb in fh.readlines()]
    
    df = get_metrics(pdbs)

    print(f'Outputting computed metrics to {outcsv}')
    os.makedirs(os.path.dirname(outcsv), exist_ok=True)
    df.to_csv(outcsv)

# TODO: rigids
def rigid_loss(r):
    trb = rf_diffusion.dev.analyze.get_trb(r)
    
    def get_motif_indep(pdb, motif_i):
        indep = aa_model.make_indep(pdb)
        is_motif = aa_model.make_mask(motif_i, indep.length())
        indep_motif, _ = aa_model.slice_indep(indep, is_motif)
        return indep_motif
    
    
    indep_motif_native = get_motif_indep(r['inference.input_pdb'], trb['con_ref_idx0'])
    indep_motif_des = get_motif_indep(rf_diffusion.dev.analyze.get_design_pdb(r), trb['con_hal_idx0'])
    i_inv = torch.argsort(torch.tensor(trb['con_hal_idx0']))
    i_inv = torch.argsort(i_inv)
    aa_model.rearrange_indep(indep_motif_des, i_inv)
    
    # ic(
    #     indep_motif_des.seq,
    #     indep_motif_native.seq,
    # )
    
    is_atom_str_shown = {}
    for i in range(indep_motif_des.length()):
        res = indep_motif_des.seq[i]
        atom_names = [n.strip() for n in ChemData().aa2long[res][:14] if n is not None]
        is_atom_str_shown[i] = atom_names
    is_res_str_shown = torch.zeros((indep_motif_des.length(),)).bool()
    true_atomized, is_diffused, is_masked_seq, atomizer = atomize.atomize_and_mask(indep_motif_native, is_res_str_shown, is_atom_str_shown)
    pred_atomized, _, _, _                              = atomize.atomize_and_mask(indep_motif_des, is_res_str_shown, is_atom_str_shown)

    rigid_losses = bond_geometry.calc_rigid_loss(true_atomized, pred_atomized.xyz, is_diffused)

    return rigid_losses.get('motif_atom_determined', torch.tensor(float('nan'))).item()

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
