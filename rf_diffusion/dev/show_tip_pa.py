import os
import copy
from rf_diffusion.dev import show_tip_row
from rf_diffusion.dev.show_tip_row import OR
from rf_diffusion.dev import analyze
import math
cmd = analyze.cmd

import logging
logger = logging.getLogger(__name__)

def combine_selectors(objs, selectors):
    s = {}
    for o in objs:
        for k, sel in selectors.items():
            s[f'{o}_{k}'] = f'{o} and {sel}'
    return s


# def get_motif_spec(row):
#     trb = analyze.get_trb(row)
#     is_atom_motif = trb['motif']
#     print(f'{is_atom_motif=}')
#     idx = trb['indep']['idx']
#     print(idx)
    
#     atom_names_by_res_idx = {}
#     for i0, atom_names in is_atom_motif.items():
#         idx_pdb = idx[i0]
#         atom_names_by_res_idx[idx_pdb] = atom_names

#     return atom_names_by_res_idx

def get_motif_spec(row, traj=False):
    trb = analyze.get_trb(row)
    if traj and 'motif' in trb:
        is_atom_motif = trb['motif']
    else:
        is_atom_motif = trb.get('atomize_indices2atomname', {})
    
    if traj and 'is_atomized' in trb:
        for k in trb['is_atomized']:
            if k not in is_atom_motif:
                is_atom_motif[k] = []

    idx = trb['indep']['idx']
    atom_names_by_res_idx = {}
    for i0, atom_names in is_atom_motif.items():
        idx_pdb = idx[i0]
        atom_names_by_res_idx[idx_pdb] = atom_names

    if not traj:
        contig_res = [resi for ch, resi in trb['con_hal_pdb_idx']]
        for resi in contig_res:
            if resi not in atom_names_by_res_idx:
                atom_names_by_res_idx[resi] = 'RES'

    logger.debug(f'{traj=} {is_atom_motif=}')
    return atom_names_by_res_idx


# def load_pdbs(pdbs):
#     pymol_objects = {}
#     for label, pdb in pdbs.items():
#         assert os.path.exists(pdb), f'{pdb} does not exist'
#         name = f'{label}_{os.path.splitext(os.path.basename(pdb))[0]}'
#         cmd.load(pdb, name)
#         pymol_objects[label] = name
#     return pymol_objects

def load_pdbs(pdbs, name_by_pdb):
    pymol_objects = {}
    for label, pdb in pdbs.items():
        assert os.path.exists(pdb), f'{label}:{pdb} does not exist'
        name = name_by_pdb[pdb]
        cmd.load(pdb, name)
        pymol_objects[label] = name
    return pymol_objects
    return pdbs

# design='/home/ahern/projects/dev_rf_diffusion/debug/sh_10.pdb'
# design='/home/ahern/projects/dev_rf_diffusion/debug/sh_15.pdb'
# design='/home/ahern/projects/dev_rf_diffusion/debug/sh_22.pdb'
# design='/home/ahern/projects/dev_rf_diffusion/debug/sh_23.pdb'
# design='/mnt/home/ahern/projects/dev_rf_diffusion/debug/debug_17.pdb'
# design='/home/ahern/projects/dev_rf_diffusion/debug/lyso_epoch7_4.pdb'
# design='/home/ahern/projects/dev_rf_diffusion/debug/sh_single_15.pdb'
# design='/home/ahern/benchmarks/aa_diffusion/tip_atoms/220420_tip_pilot_2/out/run_cond5_0.pdb'

# pdb_prefix = os.path.splitext(design)[0]
# row=analyze.make_row_from_traj(pdb_prefix)

def clear():
    analyze.clear()

def is_rf_diff(row):
    if 'pdb_path' in row:
        if "inference_outputs" in row['pdb_path']:
            return False
    k = 'resume_scheduler'
    if k in row:
        if row[k] is None:
            return True
        return math.isnan(row[k])
    k = 'inference.contig_as_guidepost'
    if k in row:
        return math.isnan(row[k])
    return True

import random
def show(row, structs = {'X0'}, af2=False, chai1_best=False, chai1_index=False, des=True, des_color=None, hetatm_color=None, mpnn_packed=False, rosetta_lig=False, input=False, ga_lig=False, hydrogenated=False, unbond_motif=True, extras=None):
    logger.debug(f"{row.get('inference.contig_as_guidepost')=}")
    # x0_pdb = analyze.get_design_pdb(row)
    # print(f'{row["extras"]=}')
    extras = row.get('extras', '{}')
    extras = eval(extras)
    # print(extras)

    # traj = analyze.load_traj(row, traj_type, traj_type)
    # print(traj)
    # traj_types = ['X0', 'Xt']
    pdbs = {}
    if des:
        pdbs['des'] =  analyze.get_diffusion_pdb(row)
        # pdbs['des_raw'] = '/net/scratch/ahern/tip_atoms/rfd_retro_3_pilot/out/run_tip_3_lig_retroaldolase_cond0_0.pdb'
    
    name = row['name']

    structs = copy.deepcopy(structs)
    if 'unidealized' in structs:
        structs.remove('unidealized')
        pdbs['unidealized'] = os.path.join(row['rundir'], f'unidealized/{name}.pdb')

    for s in structs:
        suffix = s
        if s == 'X0':
            suffix = 'pX0'
        if s == 'Xt':
            suffix = 'Xt-1'
        # if 'pymol' in row:
        #     name = row['pymol']
        # s = f'{name}_{s}_{random.randint(0, 1000)}'
        pdbs[s] = os.path.join(row['rundir'], f'traj/{name}_{suffix}_traj.pdb')
    
    # for extra_name, path in extras.items():
    #     pdbs[extra_name] = path
    pdbs.update(extras)

    name = row['name']
    mpnn_i = row['mpnn_index']
    if mpnn_packed:
        possible_paths = []
        possible_paths.append(os.path.join(row['rundir'], 'ligmpnn/packed', f'{name}_{mpnn_i}_1.pdb'))
        possible_paths.append(os.path.join(row['rundir'], 'ligmpnn/packed', f'{name}_{mpnn_i}.pdb'))
        possible_paths.append(os.path.join(row['rundir'], 'mpnn/packed', f'{name}_{mpnn_i}_1.pdb'))
        for pdb in possible_paths:
            if os.path.exists(pdb):
                pdbs['mpnn_packed'] = pdb
                break
        else:
            raise Exception(f'could not find mpnn_packed pdb at any of {possible_paths}')
    
    if rosetta_lig:
        # mpnn_i = 0
        pdbs['mpnn_packed'] = os.path.join(row['rundir'], 'ligmpnn', 'rosettalig', f'{name}_{mpnn_i}_FR.pdb')
        name
    
    if ga_lig:
        # pdbs['ga_lig'] = os.path.join(row['rundir'], 'ligmpnn', 'packed', 'rosetta_gen_ff', f'{name}_{mpnn_i}_0001.pdb')
        for pdb in [
            os.path.join(row['rundir'], 'ligmpnn', 'packed', 'addh', 'rosetta_gen_ff', f'{name}_{mpnn_i}_0001.pdb'),
            os.path.join(row['rundir'], 'ligmpnn', 'packed', 'rosetta_gen_ff', f'{name}_{mpnn_i}_0001.pdb'),
        ]:
            if os.path.exists(pdb):
                ga_lig_pdb = pdb
                break
        else:
            raise Exception(f'ga_lig pdb not found: {pdb}')
        pdbs['ga_lig'] = ga_lig_pdb
        # pdbs['ga_lig'] = os.path.join(row['rundir'], 'ligmpnn', 'packed', 'addh', 'rosetta_gen_ff', f'{name}_{mpnn_i}_0001.pdb')

    
    if hydrogenated:
        # pdbs['mpnn_packed'] = os.path.join(row['rundir'], 'ligmpnn', 'packed', f'{name}_packed_1.pdb')
        pdbs['hydrogenated'] = os.path.join(row['rundir'], 'ligmpnn/packed/addh', f'{name}_{mpnn_i}.pdb')

    # ic(pdbs)
    
    # x0_pdb = os.path.join(row['rundir'], f'traj/{row["name"]}_Xt-1_traj.pdb')
    
    # des_pdb = analyze.get_design_pdb(row)
    # pdbs = [des_pdb]
    # pdbs.append(x0_pdb)
    # pdbs = [x0_pdb]
    
    
    scored = []
    if af2:
        af2 = analyze.get_af2(row)
        pdbs['af2'] = af2
        scored.append('af2')

    if chai1_best:
        chai1_best, has_chai1 = analyze.get_best_chai1_path(row)
        assert has_chai1, f'{chai1_best=} but no chai1 pdbs found'
        pdbs['chai1_best'] = chai1_best
        scored.append('chai1_best')
    
    if chai1_index:
        assert 'chai_model_idx' in row
        chai1_index= analyze.get_chai1_path(row, row['chai_model_idx'])
        pdbs['chai1_index'] = chai1_index
        scored.append('chai1_index')
    
    if input:
        pdbs['input'] = analyze.get_input_pdb(row)

    name = row['name']
    if 'pymol' in row:
        name = row['pymol']
    prefix = f'{name}_{random.randint(0, 1000)}'
    name_by_pdb = {}
    for label, pdb in pdbs.items():
        name_by_pdb[pdb] = f'{prefix}_{label}'
    pymol_objects = load_pdbs(pdbs, name_by_pdb)
    for from_name in scored:
        for k, align in [
                ('mpnn_packed', cmd.align),
                ('des', cmd.super),
                ('unidealized', cmd.super),
        ]:
            if k not in pymol_objects:
                continue
            align(pymol_objects[from_name], pymol_objects[k])
            break

    # print(pdbs, pymol_objects)
    
    obj_selectors = {}
    for label in pymol_objects:
        if label == 'input':
            trb = analyze.get_trb(row)
            atom_names_by_res_idx = {resi: 'RES' for ch, resi in trb['con_ref_pdb_idx']}
        else:
            is_traj = label in structs
            logger.debug(f'{is_rf_diff(row)=}')
            # print(f'{is_rf_diff(row)=}')
            if is_rf_diff(row):
                trb = analyze.get_trb(row)
                atom_names_by_res_idx = {resi: 'RES' for ch, resi in trb['con_hal_pdb_idx']}
            else:
                atom_names_by_res_idx = get_motif_spec(row, traj=is_traj)

        selectors = show_tip_row.get_selectors_2(atom_names_by_res_idx)
        obj_selectors[label] = selectors
    
    # Align the input to the design.
    if input:
        for k, align in [
                ('mpnn_packed', cmd.align),
                ('des', cmd.super),
                ('unidealized', cmd.super),
        ]:
            if k not in pymol_objects:
                continue
            align(
                pymol_objects['input'], pymol_objects[k])
            break
    # Uncomment to debug
    # ic(obj_selectors)

    for i, (label, pymol_name) in enumerate(pymol_objects.items(), start=1):
        selectors = obj_selectors[label]
        sels = combine_selectors([pymol_name], selectors)
        shown = sels.pop(f'{pymol_name}_shown')
        # cmd.hide('everything', pymol_name)
        cmd.show_as('licorice', shown)
        show_tip_row.color_selectors(sels, verbose=False, des_color=des_color, hetatm_color=hetatm_color)
        cmd.show('spheres', f'name CA and {pymol_name}')

    cmd.alter('name CA', 'vdw=2.0')
    cmd.set('sphere_transparency', 0.0)
    if unbond_motif:
        cmd.unbond('chain A', 'chain B')

    if af2:
        cmd.color('white', pymol_objects['af2'])
    if chai1_best:
        cmd.color('grey', pymol_objects['chai1_best'])
    if chai1_index:
        cmd.color('grey80', pymol_objects['chai1_index'])

    entities = {}
    for label, pymol_name in pymol_objects.items():
        entities[label] = PymolObj(pymol_name, obj_selectors[label])
    
    for k, e in entities.items():
        is_traj = k in structs
        if row.get('inference.contig_as_guidepost') and is_traj:
            gp = OR([
                e['residue_gp_motif'],
                e['sidechains_diffused'],
                e['sidechains_motif'],
            ])
            cmd.unbond(gp, f' NOT ({gp})')

            # DEBUG: hide the non-gp
            # cmd.hide('everything', f'{e.name} and not ({gp})')

    # Align the input to the design.
    if input:
        for k, align in [
                ('mpnn_packed', cmd.align),
                ('des', cmd.super),
                ('unidealized', cmd.super),
        ]:
            if k not in pymol_objects:
                continue
            input_selector = entities['input']['residue_motif']
            design_selector = entities[k]['residue_motif']
            align(
                input_selector, design_selector)
            break


    return entities

class PymolObj:
    def __init__(self, name, selectors):
        self.name = name
        self.selectors = selectors

    def __getitem__(self, k):
        return f'({self.name} and {self.selectors[k]})'
    
    def NOT(self, k):
        return f'{self.name} and not ({self.selectors[k]})'
    
    def inverse(self, k):
        return f'{self.name} and not ({self[k]})'
    
    def self_selectors(self):
        o = {}
        for k, _ in self.selectors.items():
            o[k] = self[k]
        return o
    
    def copy(self, new_name):
        cmd.copy(new_name, self.name)
        o = copy.deepcopy(self)
        o.name = new_name
        return o
