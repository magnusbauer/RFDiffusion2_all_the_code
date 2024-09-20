#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'

import sys
import os
from icecream import ic

sys.path.append('/home/ahern/tools/pdb-tools/')
sys.path.append('/home/ahern/projects/pagan/swiss_army_knife/')
import protein as sak
from rf_diffusion.dev import analyze
import shutil
import glob
from tqdm import tqdm
import fire
from pdbtools import *
from rf_diffusion import aa_model
import torch
import assertpy

def get_input_aligned_pdb(row, out_path=None):
    input_pdb = analyze.get_input_pdb(row)
    des_pdb = analyze.get_design_pdb(row)
    des_p = sak.parse_pdb(des_pdb)
    self_idx, other_idx = analyze.get_idx_motif(row, mpnn=False)
    if len(self_idx):
        input_p = sak.parse_pdb(input_pdb)
        trb = analyze.get_trb(row)
        other_chains = [ch for ch, _ in trb['con_ref_pdb_idx']]
        self_ch = 'A'
        other_ch_idx = tuple(zip(other_chains, other_idx))

        other_ch_idx = []
        for abs_i in other_idx:
            other_ch_idx.append(input_p.get_relative_index_from_absolute(abs_i))
        # Defensive programming
        assertpy.assert_that(
            [ch for ch, _ in other_ch_idx]
        ).is_equal_to(other_chains)

        des_p = des_p.aligned_to_chain_idx_tuples(input_p, self_ch, self_idx, other_ch_idx)
        # We omit the oxygen here as it should have already been placed.
        des_p.chains[self_ch].xyz[self_idx, 4:] = input_p.get_xyz_by_ch_idx(other_ch_idx)[:, 4:]

    aligned_path = des_p.write_pdb(out_path)
    return aligned_path

def center_ligands(row, des_pdb, out_path=None):
    # des_pdb = analyze.get_design_pdb(row)
    ligands = row['inference.ligand']
    indep, metadata = aa_model.make_indep(des_pdb, ligands, return_metadata=True)
    trb = analyze.get_trb(row)
    potential_xyz = trb['motif_substrate_atoms']
    xyz_het_com = trb['xyz_het_com']
    # xyz_het_com = xyz_het.mean(dim=0)
    indep.xyz[indep.is_sm, 1] -= xyz_het_com

    diff = potential_xyz - indep.xyz[indep.is_sm, 1]
    ic(torch.max(diff))
    assert torch.max(diff) < 0.01
    out_path = out_path or des_pdb
    ic('writing centered ligands:', out_path)
    indep.write_pdb(out_path, ligand_name_arr=metadata['ligand_names'])    

def get_input_aligned_pdb_with_ligand(row, out_path):
    self_idx, other_idx = analyze.get_idx_motif(row, mpnn=False)
    has_motif = len(self_idx)
    ic(has_motif)
    pdb = get_input_aligned_pdb(row)
    substrate_name = row['inference.ligand']
    input_pdb = analyze.get_input_pdb(row)
    with open(input_pdb) as fh, open(pdb) as aligned:
        o = pdb_selresname.run(fh, substrate_name)
        o = pdb_selhetatm.run(o)
        o = pdb_merge.run([o, aligned])
        o = pdb_sort.run(o, [])
        # o = pdb_tidy.run(o)

        o = (e for e in o if not e.startswith('ANISOU'))
        
        with open(out_path, 'w') as of:
            for l in o:
                of.write(l)
    
    if not has_motif:
        center_ligands(row, out_path, out_path=out_path)
    shutil.copy(get_trb(analyze.get_design_pdb(row)), get_trb(out_path))
        
def get_trb(pdb):
    return pdb[:-4] + '.trb'

def main(input_dir, output_dir=None, prefix='', cautious=True):
    if os.path.abspath(input_dir) == os.path.abspath(output_dir):
        out_by_in = graft_all(input_dir, output_dir=None, prefix=prefix, cautious=cautious)
        for in_path, out_path in out_by_in.items():
            shutil.move(out_path, in_path)
    else:
        graft_all(input_dir, output_dir, prefix, cautious=cautious)
    print(f'grafted PDBs from {input_dir} to {output_dir}')

def graft_all(input_dir, output_dir=None, prefix='', cautious=True):
    '''
    For each PDB in the input directory, create a PDB with the native motif sidechains grafted onto the design.
    '''
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'grafted')
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    pdbs_to_graft = glob.glob(os.path.join(input_dir, '*.pdb'))
    pdbs_to_graft.sort()
    out_by_in = {}
    for pdb in tqdm(pdbs_to_graft):
        input_path = pdb
        output_path= os.path.join(output_dir, prefix + os.path.split(pdb)[1])
        graft(input_path, output_path, cautious=cautious)
        out_by_in[input_path] = output_path
    return out_by_in

def graft(input_path, output_path, cautious=True):
    ic('graft', input_path, output_path)
    assert input_path != output_path
    if os.path.exists(output_path) and cautious:
        return
    row = analyze.make_row_from_traj(input_path[:-4])
    get_input_aligned_pdb_with_ligand(row, output_path)
    

if __name__ == '__main__':
    fire.Fire(main)
