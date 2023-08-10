#!/net/scratch/ahern/shebangs/shebang_rf_se3_diffusion.sh

import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
sys.path.append('/home/ahern/tools/pdb-tools/')
sys.path.append('/home/ahern/projects/pagan/swiss_army_knife/')
import protein as sak
from dev import analyze
import shutil
import glob
from icecream import ic
from tqdm import tqdm
import fire
from pdbtools import *

def get_input_aligned_pdb(row, out_path=None):
    input_pdb = analyze.get_input_pdb(row)
    des_pdb = analyze.get_design_pdb(row)
    input_p = sak.parse_pdb(input_pdb)
    des_p = sak.parse_pdb(des_pdb)
    self_idx, other_idx = analyze.get_idx_motif(row, mpnn=False)
    trb = analyze.get_trb(row)
    other_ch = trb['con_ref_pdb_idx'][0][0]
    self_ch = 'A'
    des_p = des_p.aligned_to_chain_idxs(input_p, self_ch, self_idx, other_ch, other_idx)
    des_p.chains[self_ch].xyz[self_idx, 3:] = input_p[other_ch].xyz[other_idx, 3:]
    aligned_path = des_p.write_pdb(out_path)
    return aligned_path
    

def get_input_aligned_pdb_with_ligand(row, out_path):
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
    
    shutil.copy(get_trb(analyze.get_design_pdb(row)), get_trb(out_path))
        
def get_trb(pdb):
    return pdb[:-4] + '.trb'

def main(input_dir, output_dir=None, prefix=''):
    if os.path.abspath(input_dir) == os.path.abspath(output_dir):
        out_by_in = graft_all(input_dir, output_dir=None, prefix=prefix)
        for in_path, out_path in out_by_in.items():
            shutil.move(out_path, in_path)
    else:
        graft_all(input_dir, output_dir, prefix)
    print(f'grafted PDBs from {input_dir} to {output_dir}')

def graft_all(input_dir, output_dir=None, prefix=''):
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
        graft(input_path, output_path)
        out_by_in[input_path] = output_path
    return out_by_in

def graft(input_path, output_path):
    assert input_path != output_path
    if os.path.exists(output_path):
        return
    row = analyze.make_row_from_traj(input_path[:-4])
    get_input_aligned_pdb_with_ligand(row, output_path)
    

if __name__ == '__main__':
    fire.Fire(main)
