#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'

import sys
import os
sys.path.append('/home/ahern/tools/pdb-tools/')
import glob
from icecream import ic
from tqdm import tqdm
import fire
from pdbtools import *
import io

from rf_diffusion.dev import analyze

def main(input_dir, output_dir=None, prefix='', cautious=True):
    '''
    For each PDB in the input directory, create a PDB where the ligand is on chain B.
    '''
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'renumbered_chains')
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    pdbs_to_graft = glob.glob(os.path.join(input_dir, '*.pdb'))
    pdbs_to_graft.sort()
    for pdb in tqdm(pdbs_to_graft):
        out_pdb = os.path.join(output_dir, prefix + os.path.split(pdb)[1])
        if cautious and os.path.exists(out_pdb):
            continue
        row = analyze.make_row_from_traj(pdb[:-4])
        trb = analyze.get_trb(row)
        ligands = trb['config']['inference']['ligand'].split(',')
        hets = []
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for ligand, chain in zip(ligands, alphabet[1:]):
            with open(pdb) as fh:
                het = pdb_selresname.run(fh, [ligand])
                het = pdb_rplchain.run(het, ('A', chain))
                het = list(het)
                ic(ligand, len(het))
                het = io.StringIO(''.join(het))
                hets.append(het)
        with open(pdb) as fh, open(pdb):
            prot = pdb_delhetatm.run(fh)
            o = pdb_merge.run([prot] + hets)
            # o = pdb_sort.run(o, [])
            # o = pdb_tidy.run(o) -- pdb_tidy deletes CONECT records, do not run
            o = [e for e in o]
        
        with open(out_pdb, 'w') as of:
            for l in o:
                of.write(l)

if __name__ == '__main__':
    fire.Fire(main)
