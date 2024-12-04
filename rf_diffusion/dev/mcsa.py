# Run with /home/dimaio/apptainer/SE3nv-20240912.sif
import os
from datetime import datetime
from rf_diffusion import mask_generator
import networkx as nx

import copy
from itertools import groupby
import itertools
import torch
import urllib
from rf_diffusion.dev import analyze
from rf2aa import util as rf2aa_util
from rf_diffusion.chemical import ChemicalData as ChemData
import numpy as np
import pandas as pd
import re
from rf_diffusion import move_ORI

from rf_diffusion.inference import utils as inference_utils
import biotite.structure.io.pdb as biotite_pdb

import sys
sys.path.append('/home/ahern/third_party/cifutils')


max_span = 180

aa_123 = {val:key for key,val in ChemData().aa_321.items()}

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

class ExpectedError(Exception):
    pass
class LoggedError(Exception):
    pass

np.random.seed(0)

def current_datetime_string():
    # Get the current date and time
    now = datetime.now()
    # Format it as a string
    return now.strftime("%Y-%m-%d_%H-%M-%S")

def load_datasets():
    parity = pd.read_csv('/home/ahern/Downloads/keggCpd_pdb_sep17.csv', skiprows=10)

    import io

    mcsa = pd.read_csv('/home/ahern/Downloads/curated_data_mod.csv', on_bad_lines='skip')
    # print(f'{mcsa.shape=}')

    # Cleanup reactants/products parse errors
    # mcsa_clean_path = '/home/ahern/Downloads/curated_data_mod_1.csv'
    with open('/home/ahern/Downloads/curated_data_mod.csv', 'r') as fh:
        mcsa_lines = fh.read()
    lines = mcsa_lines.split('\n')

    lines_filtered = []
    for l in lines:
        m = re.compile(r'(.*,.*,.*,.*,.*,.*,C\d+,\d+,)(.*)(,,,,)').match(l)
        if m:
            l = m[1] + m[2].replace(',', '_') + m[3]
        lines_filtered.append(l)

    mcsa_buffer = io.StringIO('\n'.join(lines_filtered))


    mcsa = pd.read_csv(mcsa_buffer, on_bad_lines='skip')
    # print(f'{mcsa.shape=}')

    # MCSA alteration
    mcsa['EC1'] = mcsa.apply(lambda r: r['EC'].split('.')[0], axis=1)
    bad_ids = ['M0700']
    mcsa = mcsa[~mcsa['M-CSA ID'].map(lambda x: np.isin(x, bad_ids))].reset_index(drop=True)

    return dict(
        parity=parity,
        mcsa=mcsa,
    )


def atom_array_to_pdb(atom_array):

    pdb_file = biotite_pdb.PDBFile()
    pdb_file.set_structure(atom_array)

def pdb_to_atom_array(pdb):
    parsed = biotite_pdb.PDBFile.read(pdb)
    return parsed.get_structure(include_bonds=True)[0]

def crop_structure(
    input_pdb,
    output_pdb,
    contig_atoms,
    ligands,
):

    parsed = biotite_pdb.PDBFile.read(input_pdb)
    atom_array = parsed.get_structure()[0]
    def get_chain_res(atom):
        return f'{atom.chain_id}{atom.res_id}'
    
    chain_res = np.array([get_chain_res(atom) for atom in atom_array])
    motif_chain_res = list(contig_atoms.keys())
    is_motif_residue = np.isin(chain_res, motif_chain_res)
    is_motif_ligand = np.isin(atom_array.res_name, list(ligands) + ['ORI'])
    
    is_motif_ligand = atom_array.hetero * is_motif_ligand
    atom_array_filtered = atom_array[is_motif_ligand + is_motif_residue]

    file = biotite_pdb.PDBFile()
    file.set_structure(atom_array_filtered)
    file.write(output_pdb)

def get_theozyme(mcsa_id, pdb_id, datasets, input_pdb_dir):
    '''
    Parameters:
        mcsa_id: M-CSA ID
        pdb_id: PDB ID
        datasets: dictionary like {'parity': parity_df, ...}
    
    Returns:
        string of command line arguments to run this benchmark, i.e. "inference.input_pdb=asdf contigmap.contig_atoms=..."
    '''

    motif_dict, meta = get_motif(
        df=datasets['mcsa'],
        parity=datasets['parity'],
        mcsa_id=mcsa_id,
        pdb_id=pdb_id,
    )

    motif_dict['contig_atoms'] = sample_contig_atoms(
        pdb = motif_dict['pdb'],
        ligands = motif_dict['ligands'], # {'APC', 'MG', 'PH2'}},
        motif_selector = motif_dict['motif_selector'], # [('A', 82), ('A', 92), ('A', 95), ('A', 97)],
    )

    motif_dict['inference.partially_fixed_ligand'] = sample_ligand_atoms(
        motif_selector=motif_dict['motif_selector'],
        pdb = motif_dict['pdb'],
        ligand_chain_res_id = motif_dict['ligand_res_id'], # [('APC', ('A', 282))]
    )

    for k in motif_dict['inference.partially_fixed_ligand'].keys():
        if k.isdigit():
            raise ExpectedError(f'numeric ligand resname: {k=}')

    pdb_name = f'{mcsa_id}_{pdb_id}'
    current_pdb = motif_dict['pdb']
    meta['pdb_unprocessed'] = current_pdb[:]

    # # Strip repeated ligands
    # processed_dir_0 = os.path.join(input_pdb_dir, 'processed_0_ligand_unrepeated')
    # os.makedirs(processed_dir_0, exist_ok=True)
    # current_pdb = os.path.join(processed_dir_0, f'{pdb_name}.pdb')
    # strip_repeated_ligands(motif_dict['pdb'], current_pdb)


    # Deduplicate ligands from symmetry
    previous_pdb = current_pdb
    processed_dir_0 = os.path.join(input_pdb_dir, 'processed_0_ligand_deduped')
    os.makedirs(processed_dir_0, exist_ok=True)
    current_pdb = os.path.join(processed_dir_0, f'{pdb_name}.pdb')
    deduplicate_ligands(
        input_pdb=previous_pdb,
        output_pdb=current_pdb,
        ligand_res_id=motif_dict['ligand_res_id'],
    )

    # Center on motif
    previous_pdb = current_pdb
    processed_dir_1 = os.path.join(input_pdb_dir, 'processed_1_ori_placed')
    os.makedirs(processed_dir_1, exist_ok=True)
    current_pdb = os.path.join(processed_dir_1, f'{pdb_name}.pdb')
    move_ORI.center_pdb_on_atoms(
        input_pdb=previous_pdb,
        output_pdb=current_pdb,
        contig_atoms=motif_dict['contig_atoms'],
    )

    # Cropped
    previous_pdb = current_pdb
    processed_dir_2 = os.path.join(input_pdb_dir, 'processed_2_cropped')
    os.makedirs(processed_dir_2, exist_ok=True)
    current_pdb = os.path.join(processed_dir_2, f'{pdb_name}.pdb')
    crop_structure(
        input_pdb=previous_pdb,
        output_pdb=current_pdb,
        contig_atoms=motif_dict['contig_atoms'],
        ligands=motif_dict['ligands']
    )

    command_line_args = ""

    raw_args = get_raw_args(
        pdb_path = current_pdb,
        motif_selector = motif_dict['motif_selector'],
        contig_atoms = motif_dict['contig_atoms'],
        ligands=motif_dict['ligands'],
        partially_fixed_ligand=motif_dict['inference.partially_fixed_ligand'])
    
    return command_line_args, raw_args, meta

def get_native_contig_str(
    motif, #ch_resi
    terminal=10,
    inter_chain=20,
):
    c = np.full(len(motif)*2+1, '', dtype=object)
    is_motif = (np.arange(len(c)) % 2).astype(bool)
    c[is_motif] = [f'{ch}{r}-{r}' for ch, r in motif]
    interc = []
    for ((ch_a, r_a), (ch_b, r_b)) in pairwise(motif):
        if ch_a != ch_b:
            interc.append(inter_chain)
            continue
        interc.append(r_b - r_a - 1)
    L = sum(interc) + len(motif)

    # span = max(max_span, L)
    span = max_span

    if L < span:
        terminal = span - L
        terminal_a = terminal // 2
        terminal_b = terminal - terminal_a
    else:
        terminal_a = 0
        terminal_b = 0
    # ic( sum(interc), len(motif), L, terminal_a, terminal_b)
    interc = [terminal_a] + interc + [terminal_b]
    c[~is_motif] = interc
    return ','.join(map(str, c)), span
    

def native_contig_span(by_chain):
    l = 20
    l += 20 * (len(by_chain) - 1)
    for ch, resis in by_chain.items():
        res_i = [r for ch, r in resis]
        l += max(res_i) - min(res_i) + 1
    return l

class KnownFaultyPDB(ExpectedError):
    pass

import scipy.spatial.distance

def get_closest_chain_res_id(prospects, target_xyz):
    lig_res_ids = set(zip(prospects.chain_id, prospects.res_id))
    dist_by_res_id = {}
    for chain_id, res_id in lig_res_ids:
        prospect_res = prospects[(prospects.chain_id == chain_id) & (prospects.res_id == res_id)]
        dist_by_res_id[(chain_id, res_id)] = scipy.spatial.distance.cdist(target_xyz, prospect_res.coord).min()
    return min(dist_by_res_id, key=dist_by_res_id.get)

def get_motif(
    df, # M-CSA dataframe
    parity, # parity dataframe
    mcsa_id, pdb_id, base_dir='/home/ahern/datasets/mcsa_triads_3/'):

    label = f"{mcsa_id}_{pdb_id}"
    print(f"{label=}")

    new_2tdt = '/home/ahern/Downloads/2tdt-assembly1_processed.pdb'

    meta = {}
    data = df[
        (df['M-CSA ID'] == mcsa_id) &
        (df['PDB'] == pdb_id)].reset_index(drop=True)
    data = data.copy()
    resi = data[data['residue/reactant/product/cofactor']=='residue']
    motif_resis = resi[['chain/kegg compound', 'resid/chebi id', 'PDB code']].drop_duplicates()
    pdb_id = data.iloc[0]['PDB']
    pdb = os.path.join(base_dir, f'{pdb_id}.pdb')
    if pdb == '/home/ahern/datasets/mcsa_triads_3/2tdt.pdb':
        pdb = new_2tdt
    try:
        if not os.path.exists(pdb):
            urllib.request.urlretrieve(f'http://files.rcsb.org/download/{pdb_id}.pdb', pdb)
    except Exception as e:
        raise LoggedError(f'httperror for {pdb_id}') from e
    motif = []
    seq = []
    for _, r in motif_resis.iterrows():
        motif.append((r['chain/kegg compound'], int(r['resid/chebi id']), r['PDB code']))
    motif.sort()
    seq = [e[-1] for e in motif]
    motif = [e[:-1] for e in motif]
    if pdb == new_2tdt:
        motif = [(ch if ch != 'AA' else 'B', i) for ch,i in motif]
    r = {'motif_selector': motif, 'source': 'ec' + data.iloc[0]['EC1'], 'pdb': os.path.join(base_dir, f'{pdb_id}.pdb'), 'name': data.iloc[0]['M-CSA ID'], 'seq': seq}
    try:
        feats = inference_utils.parse_pdb(pdb)
        # atom_array = pdb_to_atom_array(pdb)
    except Exception as e:
        raise e

    pdb_to_i = {p: i for i, p in enumerate(feats['pdb_idx'])}
    # chain_res_set = set((atom.chain_id, atom.res_id) for atom in atom_array)
    for p in r['motif_selector']:
        if p not in pdb_to_i:
        # if p not in chain_res_set:
            raise ExpectedError(f'missing motif residue: {p} not in pdb_to_i for: {label}')
            # raise ExpectedError("Bad grape") from exc

    motif_xyz = analyze.get_motif_from_pdb(pdb, motif)
    motif_ca_xyz = motif_xyz[:, 1]
    ca_dist = torch.cdist(motif_ca_xyz[None], motif_ca_xyz[None])
    max_ca_dist = ca_dist.max()
    max_allowed_ca_dist = 20
    if max_ca_dist > max_allowed_ca_dist:
        raise ExpectedError(f'max_ca_dist too big: {max_ca_dist} > {max_allowed_ca_dist}')

    by_chain = {ch: list(g) for ch, g in groupby(motif, key=lambda e: e[0])}
    # if ec1 == '3' and len(by_chain) == 1:
    #     raise LoggedError(f'want multichain for ec3')
    meta['motif_by_chain'] = by_chain

#     motif_seq_dist = (abs(a-b) (_, a), (_,b) in pairwise(motif))
#     if any(d > cutoff for d in motif_seq_dist):
#         raise ExcpectedError(f'seq distance 
    
#         if abs(a-b) < cutoff:
#             print(f'motif {motif} too easy')
#             do_continue = True
#     if do_continue:
#         continue

    saved_outputs = {}

    
    cofactors = data[data['residue/reactant/product/cofactor'].isin(['cofactor'])]
    cofactor_pdb_codes = cofactors['PDB code'].unique().tolist()
    # disallowed_cofactors = ["HOH"]
    # if any(disallowed_cofactor in cofactor_pdb_codes for disallowed_cofactor in disallowed_cofactors):
    #     raise LoggedError(f'has one of the disallowed cofactors: {disallowed_cofactors=}: {cofactor_pdb_codes=}')
    
    # From manual inspection
    incorrect_cofactors_by_mcsa = {
        'M0711': ['NAP'],
        'M0255': ['NAD'],
    }
    incorrect_cofactors = incorrect_cofactors_by_mcsa.get(mcsa_id, [])
    for incorrect_cofactor in incorrect_cofactors:
        cofactor_pdb_codes.remove(incorrect_cofactor)
    ignored_cofactors = ["HOH"]
    cofactor_pdb_codes = [c for c in cofactor_pdb_codes if c not in ignored_cofactors]
    # print(f'{cofactor_pdb_codes=}')
    
    reactants_products = data[data['residue/reactant/product/cofactor'].isin(['reactant'])]
    saved_outputs['reactants_products'] = reactants_products
    saved_outputs['parity'] = parity
    reactant_parity = pd.merge(reactants_products, parity, left_on=['PDB', 'chain/kegg compound'], right_on=['pdb', 'kegg_cpd'])
    reactants_deduped = reactant_parity.drop_duplicates(['kegg_cpd'])
    reactants_deduped = reactants_deduped.drop_duplicates(['het'])
    saved_outputs['reactant_parity_saved'] = reactant_parity

    if len(reactants_deduped) == 0:
        raise LoggedError('has no reactant or product')

    reactant_pdb_codes = reactants_deduped['het'].tolist()
    # print(f'{reactant_pdb_codes=} {cofactor_pdb_codes=}')
    ligands = set(reactant_pdb_codes + cofactor_pdb_codes)
    
    # native_span = native_contig_span(by_chain)
    # if native_span > max_span:
    #     raise LoggedError(f'motif {by_chain} has span {native_span} > {max_span}')

    dist_to_het, het = analyze.get_dist_to_het(pdb, motif, ligands)
    # print(f'{pdb=}, {motif=}, {ligands=}, {dist_to_het.shape=}, {len(het)=}')
    dist_to_het_min = torch.min(dist_to_het)
    if dist_to_het_min > 3:
        raise ExpectedError(f'dist to het min too high: {dist_to_het_min}, {label=}')
    r['ligands'] = ligands


    atom_array = pdb_to_atom_array(pdb)
    chain_res = np.array([f'{atom.chain_id}_{atom.res_id}' for atom in atom_array])
    motif_chain_res = [f'{ch}_{r}' for ch, r in motif]
    is_motif_residue = np.isin(chain_res, motif_chain_res)
    motif_xyz = atom_array.coord[is_motif_residue]

    for lig in ligands:
        if (atom_array.res_name == lig).sum() == 0:
            pdbs_missing_cofactors = {
                '4eay': ['FE'],
            }
            known_faulty_pdbs = ['1qpr']
            if pdb_id in known_faulty_pdbs:
                raise KnownFaultyPDB(f'{lig=} not present in known faulty PDB: {pdb_id=}')
            if lig in pdbs_missing_cofactors.get(pdb_id, []):
                raise ExpectedError(f'known missing cofactor: {lig} in {label}')
            raise Exception(f'{lig=} not present in PDB: {pdb_id=}.')
    
    ligand_res_id = []
    for lig in ligands:
        ligand_res_id.append(
            (lig, get_closest_chain_res_id(atom_array[atom_array.res_name == lig], motif_xyz))
        )
    r['ligand_res_id'] = ligand_res_id
    # print(f'{ligand_res_id=}')

    dist_to_lig = {}
    for lig, (chain_id, res_id) in ligand_res_id:
        ligand_atoms = atom_array[(atom_array.res_name == lig) & (atom_array.res_id == res_id) & (atom_array.chain_id == chain_id)]
        dist_to_lig[lig] = scipy.spatial.distance.cdist(motif_xyz, ligand_atoms.coord).min()
    
    pairwise_distance_min = 6.0
    print(f'{dist_to_lig=}')
    # if any(dist_to_lig.values() > pairwise_distance_min):
    violations = {k: v for k, v in dist_to_lig.items() if v > pairwise_distance_min}
    if len(violations):
        raise ExpectedError(f'The following pairwise distances exceed {pairwise_distance_min=}: {violations}.  all distainces: {dist_to_lig}')

    # xyz_groups = {}
    # xyz_groups['motif'] = analyze.get_xyz_nonhet(pdb, motif)
    # for lig in ligands:
    #     het_xyz, het_names = analyze.get_xyz_het(pdb, [lig])
    #     print(f'{pdb=}')
    #     pdbs_missing_cofactors = {
    #         '4eay': ['FE'],
    #     }
    #     known_faulty_pdbs = ['1qpr']
    #     if len(het_xyz) == 0:
    #         if pdb_id in known_faulty_pdbs:
    #             raise KnownFaultyPDB(f'{lig=} not present in known faulty PDB: {pdb_id=}')
    #         if lig in pdbs_missing_cofactors.get(pdb_id, []):
    #             raise ExpectedError(f'known missing cofactor: {lig} in {label}')
    #         raise Exception(f'{lig=} not present in PDB: {pdb_id=}.')
    #     xyz_groups[lig] = torch.tensor(het_xyz, dtype=torch.float)
    # pairwise_distances = {}
    # for (a_name, a_xyz), (b_name, b_xyz) in itertools.combinations(xyz_groups.items(), 2):
    #     print(f'{a_name=} {a_xyz.shape=} {b_name=} {b_xyz.shape=}')
    #     distogram = torch.cdist(a_xyz, b_xyz)
    #     pairwise_distances[(a_name, b_name)] = torch.min(distogram)
    # print(f'{pairwise_distances=}')

    # pairwise_distance_min = 6.0
    # violations = []
    # for pair, distance in pairwise_distances.items():
    #     if a_name != 'motif' or b_name != 'motif':
    #         continue # We only care about motif-ligand distances
    #     if distance > pairwise_distance_min:
    #         violations.append((pair, distance))
    # if len(violations):
    #     raise LoggedError(f'The following pairwise distances exceed {pairwise_distance_min=}: {violations}')

    got_seq = rf2aa_util.seq2chars([feats['seq'][pdb_to_i[p]] for p in r['motif_selector']])
    want_seq = ''.join(ChemData().aa_321[e.upper()] for e in r['seq'])
    got_seq = np.array([aa_123[e] for e in got_seq])
    want_seq = np.array([aa_123[e] for e in want_seq])

    # got_seq = '-'.join(aa_123[e] for e in got_seq)
    # want_seq = '-'.join(aa_123[e] for e in want_seq)
    # print(got_seq, want_seq)
    assert len(got_seq) == len(want_seq)
    acceptable_mutation_rate = 1/3 + 1e-3
    assert (got_seq != want_seq).mean() < acceptable_mutation_rate

    # if got_seq != want_seq:
    #     # acceptable_mutants = 
    #     acceptable_mutation_rate = 1/3 + 1e-3
    #     if got_seq
    #     # Manually checked.
    #     if pdb_id != '2bif':
    #         raise LoggedError(f'got_seq != want_seq: {got_seq} != {want_seq}')
    return r, meta

def sample_contig_atoms(
    # name, # M0151
    pdb, # /home/ahern/datasets/mcsa_triads_3/1q0n.pdb
    ligands, # {'APC', 'MG', 'PH2'}},
    motif_selector, # [('A', 82), ('A', 92), ('A', 95), ('A', 97)],
    seq=None, # ['Arg', 'Arg', 'Asp', 'Asp'],
):

    het_xyz, het_names = analyze.get_xyz_het(pdb, ligands)
    feats = inference_utils.parse_pdb(pdb)
    pdb_to_i = {p: i for i, p in enumerate(feats['pdb_idx'])}
    i_to_pdb = {v:k for k,v in pdb_to_i.items()}
    motif_i = [pdb_to_i[p] for p in motif_selector]
    motif_seq = feats['seq'][motif_i]
    # got_seq =  ChemData().seq2chars(motif_seq)
    # got_seq = '-'.join(rf2aa.chemical.aa_123[e] for e in got_seq)
    # want_seq = ''.join(rf2aa.chemical.aa_321[e.upper()] for e in r['seq'])
    # want_seq = '-'.join(rf2aa.chemical.aa_123[e] for e in want_seq)
    
    contig_atoms = {}
    for i, seq in zip(motif_i, motif_seq):
        xyz_i = feats['xyz'][i]
        dist_sidechain_ligand = torch.tensor(xyz_i[:,None,...] - het_xyz[None,...]).pow(2).sum(dim=-1).sqrt()
        closest_atom = torch.argmin(dist_sidechain_ligand.min(dim=-1)[0]).item()
        n_bonds = np.random.randint(1, 3)
        atom_names = mask_generator.get_atom_names_within_n_bonds(seq, closest_atom, n_bonds)
        atom_names = [a.strip() for a in atom_names]
        ch, idx = i_to_pdb[i]
        contig_atoms[f'{ch}{idx}'] = atom_names
        # if r['name'] == 'M0151':
        #     print(f'{ch}{idx}, {n_bonds=}')
    
    return contig_atoms

def nearest_fraction_by_bonds(atom_array, atom_id, fraction_shown, min_shown=1):
    # atom = atom_array[atom_array.atom_id == atom_id]
    atom_idx = np.where(atom_array.atom_id == atom_id)[0][0]
    print(f'{atom_idx=}')

    print(f'{atom_array.bonds=}')

    # G = nx.from_edgelist(atom_array.bonds.adjacency_matrix())
    G = nx.from_numpy_matrix(atom_array.bonds.adjacency_matrix())
    visited = [atom_idx]
    for depth, nodes_at_depth in nx.bfs_successors(G, atom_idx):
        visited.extend(nodes_at_depth)
        # if depth >= n_bonds:
        #     break
    
    assert len(visited) == len(atom_array), f"{len(visited)=} {len(atom_array)=}"
    n_atoms = len(atom_array)
    n_sampled = max(int(np.floor(fraction_shown*n_atoms)), min_shown)
    return atom_array[visited[:n_sampled]]

def get_bonds_longer_than(atom_array, max_bond_length=3):
    violations = []
    for i, atom in enumerate(atom_array):
        bondeds, bond_types = atom_array.bonds.get_bonds(i)
        for bonded, bond_type in zip(bondeds, bond_types):
            atom2 = atom_array[bonded]
            dist = np.linalg.norm(atom.coord - atom2.coord)
            if dist > max_bond_length:
                violations.append(atom.atom_id, atom2.atom_id, bond_type, dist)
    return violations

def sample_ligand_atoms(
        pdb,
        motif_selector,
        ligand_chain_res_id,
        low_frac=0.0,
        high_frac=1.0,
    ):

    atom_array = pdb_to_atom_array(pdb)
    print(f'{pdb=}')
    print(f'{atom_array.bonds=}')
    atom_array.set_annotation('atom_id', np.arange(len(atom_array))+1)

    def get_chain_res(atom):
        return f'{atom.chain_id}_{atom.res_id}'
    
    chain_res = np.array([get_chain_res(atom) for atom in atom_array])
    motif_chain_res = list(f'{ch}_{res}' for ch, res in motif_selector)
    print(f'{motif_chain_res=}')
    print(f'{chain_res=}')
    is_motif_residue = np.isin(chain_res, motif_chain_res)
    motif_xyz = atom_array.coord[is_motif_residue]
    motif_com = motif_xyz.mean(axis=0)
    motif_xyz = motif_com[None, :]

    atom_names_by_ligand = {}
    for res_name, (chain_id, res_id) in ligand_chain_res_id:
        sm = atom_array[(atom_array.res_name == res_name) & (atom_array.chain_id == chain_id) & (atom_array.res_id == res_id)]
        bonds_too_long = get_bonds_longer_than(sm)
        if len(bonds_too_long):
            raise ExpectedError(f'ligand bond length violation: atoms in {res_name} chain {chain_id=} {res_id=} have bonds that are too long: {bonds_too_long}')

        # Filter out hydrogens
        sm = sm[sm.element != 'H']

        print(f'{motif_xyz.shape=}')
        print(f'{sm.coord.shape=}')
        sm_motif_atom_dist = scipy.spatial.distance.cdist(sm.coord, motif_xyz)
        print(f'{sm_motif_atom_dist.shape=}')
        sm_motif_dist = sm_motif_atom_dist.min(axis=1)
        closest_atom_index = sm_motif_dist.argmin()
        closest_atom = sm[closest_atom_index]
        nearest_atoms = nearest_fraction_by_bonds(sm, closest_atom.atom_id, np.random.uniform(low_frac, high_frac))
        atom_names_by_ligand[res_name] = nearest_atoms.atom_name.tolist()
    return atom_names_by_ligand
        

def strip_repeated_ligands(from_p, to_p):
    assert not os.path.exists(to_p)
    het_seen = {}
    with open(from_p, 'r') as from_fh:
        with open(to_p, 'w') as to_fh:
            for l in from_fh:
                if l.startswith('HETATM'):
                    ch = l[21]
                    resn = l[17:20].strip()
                    if resn not in het_seen:
                        het_seen[resn] = ch
                    if het_seen[resn] != ch:
                        continue
                to_fh.write(l)

def deduplicate_ligands(
    input_pdb,
    output_pdb,
    ligand_res_id,
):
    written = set()
    skipped = set()

    assert not os.path.exists(output_pdb)
    with open(input_pdb, 'r') as from_fh:
        with open(output_pdb, 'w') as to_fh:
            for l in from_fh:
                if l.startswith('HETATM'):
                    ch = l[21]
                    res_id = int(l[22:26].strip())
                    resn = l[17:20].strip()
                    key = (resn, (ch, res_id))
                    if key not in ligand_res_id:
                        skipped.add(key)
                        continue
                    written.add(key)
                to_fh.write(l)
    # print(f'{ligand_res_id=}')
    # print(f'{written=}')
    # print(f'{skipped=}')
    # skipped_no_water = set(k for k in skipped if k[0] != 'HOH')
    # print(f'{skipped_no_water=}')


import yaml
from yaml import Loader
def inline_yaml(obj):
    return yaml.dump(obj, line_break=False, width=99999, default_flow_style=True).strip().replace(' ','')


def get_raw_args(
    pdb_path,
    motif_selector,
    contig_atoms,
    ligands,
    partially_fixed_ligand,
):
    for motif in [motif_selector]:
        contig, span = get_native_contig_str(motif)
        contig_atoms_str = []
        for k, v in contig_atoms.items():
            v = ','.join(v)
            contig_atoms_str.append(f"\\'{k}\\':\\'{v}\\'")
        contig_atoms_str = ",".join(contig_atoms_str)
        contig_atoms_str = f'{{{contig_atoms_str}}}'
        contig_atoms_str = f'"\'{contig_atoms_str}\'\"'

        partially_fixed_ligand_str_0 = "'{ATH: [OH, C4, C3, C5, H4, C2, C6, O3]}'"
        partially_fixed_ligand_str = inline_yaml(partially_fixed_ligand)
        print(f'{partially_fixed_ligand=}')
        print(f'{partially_fixed_ligand_str=}')
        print(f'{partially_fixed_ligand_str_0=}')

        print(f'{inline_yaml(partially_fixed_ligand)=}')

        for k,v in partially_fixed_ligand.items():
            new_v = []
            for vv in v:
                if "'" in vv:
                    vv = f'\\"{vv}\\"'
                new_v.append(vv)
            partially_fixed_ligand[k] = new_v

        partially_fixed_ligand_str = inline_yaml(partially_fixed_ligand)
        print(f'{partially_fixed_ligand_str=}')

        # partially_fixed_ligand = "'{[ATH: [OH, C4, C3, C5, H4, C2, C6, O3]}'"
        o = f'''++inference.partially_fixed_ligand="{partially_fixed_ligand_str}"''' # WORKS

        o = f'''inference.input_pdb={pdb_path} inference.ligand=\\'{",".join(ligands)}\\' contigmap.contigs=[\\'{contig}\\'] contigmap.contig_atoms={contig_atoms_str} contigmap.length={span}-{span} ++inference.partially_fixed_ligand="{partially_fixed_ligand_str}"''' # WORKS

        # o = f'''"inference.input_pdb={pdb_path} contigmap.contigs=[\\\\'{contig}\\\\'] inference.ligand=\\\\'{",".join(ligands)}\\\\' contigmap.length={span}-{span} contigmap.contig_atoms={contig_atoms_str} inference.partially_fixed_ligand=\'{partially_fixed_ligand_str}\'"'''

        # o = f'''"inference.input_pdb={pdb_path} contigmap.contigs=[\\\\'{contig}\\\\'] inference.ligand=\\\\'{",".join(ligands)}\\\\' contigmap.length={span}-{span} contigmap.contig_atoms={contig_atoms_str} inference.partially_fixed_ligand=\\"{partially_fixed_ligand_str}\\""'''
        print("o:")
        print(o)

        return o

def get_command_line_args(
    pdb_path,
    motif_selector,
    contig_atoms,
    ligands,
    partially_fixed_ligand,
):
    for motif in [motif_selector]:
        contig, span = get_native_contig_str(motif)
        contig_atoms_str = []
        for k, v in contig_atoms.items():
            v = ','.join(v)
            contig_atoms_str.append(f"\\\\'{k}\\\\':\\\\'{v}\\\\'")
        contig_atoms_str = ",".join(contig_atoms_str)
        contig_atoms_str = f'{{{contig_atoms_str}}}'
        contig_atoms_str = f'\\"\'{contig_atoms_str}\'\\\"'

        partially_fixed_ligand_str_0 = "'{ATH: [OH, C4, C3, C5, H4, C2, C6, O3]}'"
        partially_fixed_ligand_str = inline_yaml(partially_fixed_ligand)
        print(f'{partially_fixed_ligand=}')
        print(f'{partially_fixed_ligand_str=}')
        print(f'{partially_fixed_ligand_str_0=}')

        print(f'{inline_yaml(partially_fixed_ligand)=}')

        for k,v in partially_fixed_ligand.items():
            new_v = []
            for vv in v:
                if "'" in vv:
                    vv = f'\\\\"{vv}\\\\"'
                new_v.append(vv)
            partially_fixed_ligand[k] = new_v

        partially_fixed_ligand_str = inline_yaml(partially_fixed_ligand)
        print(f'{partially_fixed_ligand_str=}')

        # partially_fixed_ligand = "'{[ATH: [OH, C4, C3, C5, H4, C2, C6, O3]}'"

        o = f'''"inference.input_pdb={pdb_path} contigmap.contigs=[\\\\'{contig}\\\\'] inference.ligand=\\\\'{",".join(ligands)}\\\\' contigmap.length={span}-{span} contigmap.contig_atoms={contig_atoms_str} inference.partially_fixed_ligand=\'{partially_fixed_ligand_str}\'"'''

        o = f'''"inference.input_pdb={pdb_path} contigmap.contigs=[\\\\'{contig}\\\\'] inference.ligand=\\\\'{",".join(ligands)}\\\\' contigmap.length={span}-{span} contigmap.contig_atoms={contig_atoms_str} inference.partially_fixed_ligand=\'{partially_fixed_ligand_str}\'"'''

        o = f'''"inference.input_pdb={pdb_path} contigmap.contigs=[\\\\'{contig}\\\\'] inference.ligand=\\\\'{",".join(ligands)}\\\\' contigmap.length={span}-{span} contigmap.contig_atoms={contig_atoms_str} inference.partially_fixed_ligand=\\"{partially_fixed_ligand_str}\\""'''
        print("o:")
        print(o)

        return o
        # return f'''"inference.input_pdb={pdb_path} contigmap.contigs=[\\\\'{contig}\\\\'] inference.ligand=\\\\'{",".join(ligands)}\\\\' contigmap.length={span}-{span} contigmap.contig_atoms={contig_atoms_str}"'''




        # partially_fixed_ligand = f"'{inline_yaml(partially_fixed_ligand)}'"
        # return f'''"inference.input_pdb={pdb_path} contigmap.contigs=[\\\\'{contig}\\\\'] inference.ligand=\\\\'{",".join(ligands)}\\\\' contigmap.length={span}-{span} contigmap.contig_atoms={contig_atoms_str} inference.partially_fixed_ligand={partially_fixed_ligand}"'''

def test_inference_from_mcsa(datasets):
    mcsa_id = 'M0054'
    pdb_id = '1qfe'
    config_dir = '/home/ahern/data/generative_modeling/mcsa/test'

    data_dir = os.path.join(config_dir, current_datetime_string())
    out_dir = os.path.join(data_dir, 'out')
    in_dir = os.path.join(data_dir, 'in')
    os.makedirs(data_dir)
    os.makedirs(out_dir)
    os.makedirs(in_dir)

    command_line_args = get_theozyme(mcsa_id, pdb_id, datasets, input_pdb_dir=in_dir)

    benchmark_json = os.path.join(data_dir, 'benchmark.json')
    benchmark_name = f"{mcsa_id}_{pdb_id}"
    with open(benchmark_json, 'w') as fh:
        fh.write("{")
        fh.write(f'    "{benchmark_name}":{command_line_args}')
        fh.write("}")

    # pipeline_conf = '/home/ahern/reclone/rf_diffusion_dev/rf_diffusion/benchmark/configs/demo_cfg_center_all_finetune_ec_4.yaml'
    pipeline_conf = 'demo_cfg_center_all_finetune_ec_4'

    print(f'./benchmark/pipeline.py --config-name={pipeline_conf} sweep.benchmark_json={benchmark_json} in_proc=1')

def get_benchmark_json(
        pdb_ec,
        datasets,
        config_dir = '/home/ahern/data/generative_modeling/mcsa/test',
        start_at_i = 0,
        ):

    data_dir = os.path.join(config_dir, current_datetime_string())
    out_dir = os.path.join(data_dir, 'out')
    in_dir = os.path.join(data_dir, 'in')
    os.makedirs(data_dir)
    os.makedirs(out_dir)
    os.makedirs(in_dir)

    benchmark_json = os.path.join(data_dir, 'benchmark.json')
    records = []
    with open(benchmark_json, 'w') as fh:
        fh.write("{\n")
        for i, (_, r) in enumerate(pdb_ec.iterrows()):
            new_r = copy.deepcopy(r)
            new_r['valid'] = True
            if i < start_at_i:
                continue
            print(f'##########################################{i=}')
            mcsa_id = r['M-CSA ID']
            pdb_id = r['PDB']
            try:
                command_line_args, raw_args, meta = get_theozyme(mcsa_id, pdb_id, datasets, input_pdb_dir=in_dir)
                # new_r['command_line_args'] = command_line_args
                # print(f'{command_line_args=}')
                # new_r['benchmark_value'] = eval(command_line_args)
                print(f'{raw_args=}')
                new_r['benchmark_value'] = raw_args
                for k, v in meta.items():
                    new_r[k] = v
            except Exception as e:
                if isinstance(e, ExpectedError):
                    new_r['error'] = e
                    new_r['valid'] = False
                else:
                    e.i = i
                    raise e
            benchmark_name = f"{mcsa_id}_{pdb_id}"
            new_r['benchmark_name'] = benchmark_name

            print(f"Valid: {new_r['valid']} {new_r['error'] if not new_r['valid'] else ''}")
            records.append(new_r)
            # if new_r['valid']:
            #     maybe_comma = ',' if i != last_i else ''
            #     fh.write(f'    "{benchmark_name}":{command_line_args}{maybe_comma}\n')
        fh.write("}")
    
    out = pd.DataFrame(records)
    out['data_dir'] = data_dir
    return out

def valid_for_benchmark(pdb_ec, datasets, start_at_i=0):
    records, benchmark_json, errors = get_benchmark_json(pdb_ec, datasets, start_at_i=start_at_i)
    return records[records['valid']]

    # return benchmark_json, errors

### Filtering functions

def good_catalytic_residue_type(mcsa):
    # resns = df.drop_duplicates(['M-CSA ID', 'residue/reactant/product/cofactor', 'chain/kegg compound', 'resid/chebi id'])
    df_residues = mcsa[mcsa['residue/reactant/product/cofactor']=='residue']
    mechanisms = df_residues.groupby(
            ['M-CSA ID', 'PDB']
        ).agg(
            location_set=pd.NamedAgg(column="function location/name", aggfunc=lambda x: set(x))
        ).reset_index()
    allowed = set(['side_chain', 'main_chain_amide', 'main_chain_carbonyl', 'main_chain'])
    mechanisms = mechanisms[mechanisms.apply(lambda x: len(x.location_set.difference(allowed))==0, axis=1)]
    return mechanisms[['M-CSA ID', 'PDB']].drop_duplicates()

ignorable_reactants = ['C00007', 'C00001', 'C00080']
def get_parity_reactants(mcsa_df, parity):
    reactants = mcsa_df[mcsa_df['residue/reactant/product/cofactor'].isin(['reactant'])]
    reactants = reactants[~reactants['chain/kegg compound'].isin(ignorable_reactants)]
    reactant_parity = pd.merge(reactants, parity, how='left', left_on=['PDB', 'chain/kegg compound'], right_on=['pdb', 'kegg_cpd'])
    reactants_deduped = reactant_parity.drop_duplicates(['M-CSA ID', 'PDB', 'chain/kegg compound'])
    return reactants_deduped

# def get_parity_reactants_2(mcsa_df, parity):
#     reactants = mcsa_df[mcsa_df['residue/reactant/product/cofactor'].isin(['reactant'])]
#     reactants = reactants[~reactants['chain/kegg compound'].isin(ignorable_reactants)]
#     reactant_parity = pd.merge(reactants, parity[parity['r_or_p'] == 'r'], how='left', left_on=['PDB', 'chain/kegg compound'], right_on=['pdb', 'kegg_cpd'])
#     reactants_deduped = reactant_parity.drop_duplicates(['M-CSA ID', 'PDB', 'chain/kegg compound'])
#     return reactants_deduped

def has_all_reactants_2(mcsa_df, parity):
    reactants_deduped = get_parity_reactants(mcsa_df, parity)
    parity_reactant_count = reactants_deduped.groupby(['M-CSA ID', 'PDB']).agg(
            count=pd.NamedAgg(column="EC", aggfunc=len),
            parity_count=pd.NamedAgg(column="kegg_cpd", aggfunc=lambda x: (~x.isna()).sum())
    ).reset_index()
    more_parity = parity_reactant_count[parity_reactant_count['count'] < parity_reactant_count['parity_count']]
    assert len(more_parity) == 0
    has_all_reactants = parity_reactant_count[parity_reactant_count['count'] == parity_reactant_count['parity_count']]
    
    return has_all_reactants[['M-CSA ID', 'PDB']].drop_duplicates()


def get_residue_counts(mcsa_df):
    resns = mcsa_df.drop_duplicates(['M-CSA ID', 'residue/reactant/product/cofactor', 'chain/kegg compound', 'resid/chebi id'])
    res_resns = resns[resns['residue/reactant/product/cofactor']=='residue']
    residue_counts = res_resns.groupby(['EC', 'M-CSA ID', 'PDB']).agg(
        residue_count=pd.NamedAgg(column="EC", aggfunc=len)).reset_index()
    return residue_counts

def residue_count_between(mcsa, low, high):
    # resns = mcsa.drop_duplicates(['M-CSA ID', 'residue/reactant/product/cofactor', 'chain/kegg compound', 'resid/chebi id'])
    # res_resns = resns[resns['residue/reactant/product/cofactor']=='residue']
    # residue_counts = res_resns.groupby(['EC1', 'M-CSA ID', 'PDB']).agg(
    #     count=pd.NamedAgg(column="EC", aggfunc=len)).reset_index()
    residue_counts = get_residue_counts(mcsa)
    residue_counts_reasonable = residue_counts[
        (residue_counts['residue_count'] >= low) & 
        (residue_counts['residue_count'] <= high)
    ]
    return residue_counts_reasonable[['M-CSA ID', 'PDB']].drop_duplicates()

def present_in_mcsa_and_parity(mcsa_df, parity):

    mcsa_deduped = mcsa_df.drop_duplicates(['EC', 'PDB'])
    parity_deduped = parity.drop_duplicates(['pdb', 'ec'])
    mcsa_and_parity = mcsa_deduped.merge(parity_deduped, how='inner', left_on=['EC', 'PDB'], right_on=['ec', 'pdb'])

    return mcsa_and_parity

def long_electron_transfer_chain(mcsa_df):
    mcsa_pdb = mcsa_df[['M-CSA ID', 'PDB']].drop_duplicates()
    ignore = pd.DataFrame(
        [
            ('M0105', '1vlb'), # Long electron transfer chain: https://pubs.acs.org/doi/10.1021/bi0510025
            ('M0134', '2toh'), # Substrate is itself (Meta-tyrosine), confusing to represent as a theozyme for PDB processing reasons
            ('M0178', '1bmf'), # H+ gradient
            ('M0230', '1gzg'), # Missing zinc
            ('M0324', '1tph'), # Misnumbered chains (A == 1, B == 2)
        ],
        columns=['M-CSA ID', 'PDB']
    )

    # print(f"{mcsa_pdb.shape=}")
    # print(f"{mcsa_pdb.merge(ignore, how='left', on=['M-CSA ID', 'PDB'], indicator=True)['_merge']=}")
    # print(f"{mcsa_pdb.merge(ignore, how='left', on=['M-CSA ID', 'PDB'], indicator=True).value_counts(['_merge'])=}")

    return mcsa_pdb.merge(ignore, how='left', on=['M-CSA ID', 'PDB'], indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)

import json
from rf_diffusion.dev import show_tip_row
cmd = analyze.cmd
def show_benchmark_json(bm_path):
    
    with open(bm_path) as fp:
        bm = json.load(fp)

    pdb_contig = []
    for name, args in bm.items():
        contig = re.search('contigs=\[\\\\\'(.*)\\\\\'\]', args).groups()[0]
        contigs = [c for c in contig.split(',') if c[0].isalpha()]
        motif = []
        for c in contigs:
            ch = c[0]
            a, b = list(map(int, c[1:].split('-')))
            for i in range(a, b+1):
                motif.append((ch, i))
        # print(motif)
        input_pdb = re.search('input_pdb=(.*\.pdb)\s', args).groups()[0]
        # input_pdb = os.path.join(input_dir, input_pdb)
        # print(f'{args=}')
        ligand = re.search('inference.ligand=\\\\\'(\S*)\\\\\'', args).groups()[0]
        contig_atoms = re.search('contigmap.contig_atoms=\"(\S*)\"', args).groups()[0]
        # partially_fixed_ligand_str = re.search('inference.partially_fixed_ligand=\'(.*)\'', args).groups()[0]
        partially_fixed_ligand_str = re.search('inference.partially_fixed_ligand=\"(.*)\"', args).groups()[0]
        partially_fixed_ligand = yaml.load(partially_fixed_ligand_str.replace('\\', ''), Loader=Loader)
        # print(f'{ligand=}')
        # return
        # print(f'{contig_atoms=}')
        # print(f'{eval(contig_atoms)=}')
        # print(f'{eval(eval(contig_atoms))=}')
        contig_atoms = eval(eval(contig_atoms))
        # print(input_pdb)
        pdb_contig.append((input_pdb, tuple(sorted(motif)), ligand, contig_atoms, partially_fixed_ligand))


    pdb_contig = sorted(pdb_contig)

    analyze.clear()

    structures = []
    for pdb, motif, ligands, contig_atoms, partially_fixed_ligand in pdb_contig:
        native = os.path.splitext(os.path.basename(pdb))[0]
        ref_idx = motif
        # cmd.load(pdb, native)
        # print(f'{pdb=}')
        # print(f'{analyze.to_selector(ref_idx)=}')
        # print(ligands)
        ligand_selector = f'({" or ".join("(resn " + l + ")" for l in ligands.split(","))})'
        # print(f'{ref_idx=}')
        cmd.do(f'load {pdb}, {native}; remove ({native} and not ({ligand_selector} or {analyze.to_selector(ref_idx)}))')
        # print(f'load {pdb}, {native}; remove ({native} and not ({ligand_selector} or {analyze.to_selector(ref_idx)}))')
        # cmd.do(f'load {pdb}, {native}; remove ({native} and not ((not resn {ligand}) or {analyze.to_selector(ref_idx)}))')
        # cmd.do(f'load {pdb}, {native}_het; remove (not resn {ligand})')
        # cmd.color('atomic')
        tip_selector = []
        for k, v in contig_atoms.items():
            ch = k[0]
            resi = k[1:]
            atom_names = v.split(',')
            tip_selector.append(show_tip_row.get_atom_selector(native, ch, resi, atom_names))
        
        tip_selector = ' or '.join(tip_selector)
        cmd.color('paper_melon', tip_selector)
        # tip_selector = f'{native} and ({tip_selector})'
        # cmd.do
        #     atom_sel = ' or '.join(f'name {a}' for a in atom_names)
        
        cmd.color('orange', f'{native} and hetatm and elem C')
        cmd.show_as('licorice')

        for res_name, atom_names in partially_fixed_ligand.items():
            atom_names_sel = ' or '.join(f'name {a}' for a in atom_names)
            cmd.color('paper_blue', f'{native} and resn {res_name} and ({atom_names_sel})')
        structures.append(analyze.Structure(native, motif))
        # break

    cmd.center(structures[0].name)
    cmd.zoom(structures[0].name)
    for s in structures[1:]:
        cmd.align(s.name, structures[0].name)

    cmd.set('grid_mode', 1)

    return pdb_contig


def write_benchmarks(validated):

    benchmark_dict = {}
    for _, r in validated.iterrows():
        # benchmark_dict[r['benchmark_name']] = eval(r['command_line_args'])
        benchmark_dict[r['benchmark_name']] =r['benchmark_value']

    data_dir = validated.iloc[0]['data_dir']
    benchmark_json_path = os.path.join(data_dir, 'benchmark.json')

    with open(benchmark_json_path, 'w') as f:
        json.dump(benchmark_dict, f, indent=4)
    
    return benchmark_json_path
