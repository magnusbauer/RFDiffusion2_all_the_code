#!/net/software/containers/users/dtischer/shebang_rf_se3_diffusion_dev.sh

import os
# import sys
# script_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.join(script_dir, '..'))
# sys.path.append(os.path.join(script_dir, '../..'))


import pandas as pd
import fire
from icecream import ic
from tqdm import tqdm
import torch

from rf_diffusion.dev import analyze
# import analysis.metrics
from rf_diffusion.inference import utils
# import rf2aa.util


def main(pdb_names_file, outcsv=None):
    with open(pdb_names_file, 'r') as fh:
        pdbs = [pdb.strip() for pdb in fh.readlines()]
    
    df = get_metrics(pdbs)

    print(f'Outputting computed metrics to {outcsv}')
    os.makedirs(os.path.dirname(outcsv), exist_ok=True)
    df.to_csv(outcsv)

# def get_protein_bond_feats(seq):

#     for i, res in enumerate(seq):
#         for j, bond in enumerate(rf2aa.chemical.aabonds[res]):
#             start_idx = rf2aa.chemical.aabondsaa2long[res].index(bond[0])
#             end_idx = aa2long[res].index(bond[1])
#             if (i, start_idx) not in ra2ind or (i, end_idx) not in ra2ind:
#                 #skip bonds with atoms that aren't observed in the structure
#                 continue
#             start_idx = ra2ind[(i, start_idx)]
#             end_idx = ra2ind[(i, end_idx)]

#             # maps the 2d index of the start and end indices to btype
#             bond_feats[start_idx, end_idx] = aabtypes[res][j]
#             bond_feats[end_idx, start_idx] = aabtypes[res][j]

# def get_motif(row, mpnn=True):

#     if mpnn:
#         design_pdb = dev.analyze.get_design_pdb(row)
#     else:
#         design_pdb = dev.analze.get_diffusion_pdb(row)
#     ic(design_pdb)
#     design_info = utils.process_target(design_pdb, center=False, parse_hetatom=True)
#     # des = design_info['xyz_27']
#     # hydrogens = des[:, 14:]
#     # ic(type(hydrogens))
#     # has_hydrogen = (~torch.isnan(hydrogens)).any()
#     # ic(has_hydrogen)
#     des = design_info['xyz_27']
#     mask = design_info['mask_27']

def get_metrics(pdbs):
    records = []
    for pdb in tqdm(pdbs):
        record = {}
        ic(pdb)
        row = analyze.make_row_from_traj(pdb[:-4])
        # ic(row['mpnn_index'])
        ic(pdb, row['name'], row['mpnn_index'])
        record['name'] = row['name']
        record['mpnn_index'] = row['mpnn_index']

        design_pdb = analyze.get_design_pdb(row)
        ic(design_pdb)
        design_info = utils.process_target(design_pdb, center=False, parse_hetatom=True)
        # des = design_info['xyz_27']
        # hydrogens = des[:, 14:]
        # ic(type(hydrogens))
        # has_hydrogen = (~torch.isnan(hydrogens)).any()
        # ic(has_hydrogen)
        des = design_info['xyz_27']
        mask = design_info['mask_27']
        des[~mask] = torch.nan
        dgram = torch.sqrt(torch.sum((des[None, None,:,:,:] - des[:,:,None,None, :]) ** 2, dim=-1))
        dgram = torch.nan_to_num(dgram, 999)
        # ic(dgram.shape)
        
        # is_n = torch.full_like(dgram, False)
        # is_n[:,:,:,0] = True
        # is_n[:,0,:,:] = True
        # is_c = torch.full_like(dgram, False)
        # is_c[:,1,:,1] = True
        # is_c[:,1,:,1] = True
        # is_cn = torch.full_like(dgram, False)
        # is_cn = torch.
        # L = des.shape[0]
        # res_diff = (torch.arange(L)[None, :] - torch.arange(L)[None, :])
        # is_upstream_neighbor = res_diff == 1
        # is_downstream_neighbor = res_diff == -1
        # is_peptide_bond = 


        # Ignore backbone-backbone distance, as ligandmpnn is not responsible for this.
        bb2bb = torch.full(dgram.shape, False)
        bb2bb[:, :4, :, :4] = True
        # ic(bb2bb.dtype)
        dgram[bb2bb] = 999

        dgram = dgram.min(dim=3)[0]
        dgram = dgram.min(dim=1)[0]
        # ic(dgram)
        # ic(dgram.shape)

        dgram.fill_diagonal_(999)
        min_dist = dgram.min()
        record['res_to_res_min_dist'] = min_dist.item()
        is_dist = torch.ones_like(dgram).bool()
        is_dist = torch.triu(is_dist, diagonal=1)
        dists = dgram[is_dist]
        clash_dist = 2 # roughly 2 VDW radii
        
        n_pair_clash = torch.sum(dists < clash_dist).item()
        record['n_pair_clash'] = n_pair_clash
        res_clash = (dgram < clash_dist).any(dim=-1)
        record['n_res_clash'] = res_clash.sum().item()
        record['fraction_res_clash'] = res_clash.float().mean().item()
        record['res_clash'] = res_clash.tolist()
        pair_clashes = (dgram < clash_dist)

        diffusion_pdb = analyze.get_diffusion_pdb(row)
        diffusion_info = utils.process_target(diffusion_pdb, center=False, parse_hetatom=True)
        diff = diffusion_info['xyz_27']
        diff[~mask] = torch.nan

        trb = analyze.get_trb(row)
        des_motif = des[trb['con_hal_idx0']]
        diff_motif = diff[trb['con_hal_idx0']]
        flat_des = des_motif[~des_motif.isnan().any(dim=-1)]
        flat_diff = diff_motif[~diff_motif.isnan().any(dim=-1)]

        mpnn_motif_dist = ((flat_des - flat_diff) ** 2).sum(dim=-1) ** 0.5
        record['motif_ideality_diff'] = mpnn_motif_dist.mean().item()

        records.append(record)

    df = pd.DataFrame.from_records(records)
    return df

if __name__ == '__main__':
    fire.Fire(main)