import os
from dev import show_tip_row
from dev import analyze
cmd = analyze.cmd

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
#     trb
    # is_atom_motif = trb['motif']
    # print(f'{is_atom_motif=}')
    idx = trb['indep']['idx']
#     print(idx)
    
    atom_names_by_res_idx = {}
    for i0, atom_names in is_atom_motif.items():
        idx_pdb = idx[i0]
        atom_names_by_res_idx[idx_pdb] = atom_names

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
        assert os.path.exists(pdb), f'{pdb} does not exist'
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
    analyze.sak.clear(cmd)
    cmd.do('@~/.pymolrc')

import random
def show(row, structs = {'X0'}, af2=False, des=True, des_color=None):
    # x0_pdb = analyze.get_design_pdb(row)


    traj_type = 'X0'
    # traj = analyze.load_traj(row, traj_type, traj_type)
    # print(traj)
    traj_types = ['X0', 'Xt']
    pdbs = {}
    if des:
        pdbs['des'] =  analyze.get_design_pdb(row)
    for s in structs:
        if s in traj_types:
            if s == 'X0':
                suffix = 'pX0'
            if s == 'Xt':
                suffix = 'Xt-1'
            name = row['name']
            # if 'pymol' in row:
            #     name = row['pymol']
            # s = f'{name}_{s}_{random.randint(0, 1000)}'
            pdbs[s] = os.path.join(row['rundir'], f'traj/{name}_{suffix}_traj.pdb')
    
    # x0_pdb = os.path.join(row['rundir'], f'traj/{row["name"]}_Xt-1_traj.pdb')
    
    # des_pdb = analyze.get_design_pdb(row)
    # pdbs = [des_pdb]
    # pdbs.append(x0_pdb)
    # pdbs = [x0_pdb]
    
    
    if af2:
        af2 = analyze.get_af2(row)
        pdbs['af2'] = af2
    
    name = row['name']
    if 'pymol' in row:
        name = row['pymol']
    prefix = f'{name}_{random.randint(0, 1000)}'
    name_by_pdb = {}
    for label, pdb in pdbs.items():
        name_by_pdb[pdb] = f'{prefix}_{label}'

    # print(f"{pdbs=}")
    
    pymol_objects = load_pdbs(pdbs, name_by_pdb)
    # print(f'{pymol_objects=}')

    # load_pdbs(pdbs)
    # pymol_objects = 
    if af2:
        cmd.align(pymol_objects['af2'], pymol_objects['des'])
    # print(pdbs, pymol_objects)
    
    obj_selectors = {}
    for label in pymol_objects:
        # traj = label in traj_types
        # print(f'{label=}')
        is_traj = label.split('_')[0] in traj_types
        atom_names_by_res_idx = get_motif_spec(row, traj=is_traj)
        # print(f'{atom_names_by_res_idx=}')
        selectors = show_tip_row.get_selectors_2(atom_names_by_res_idx)
        obj_selectors[label] = selectors
    # print(f'{pymol_objects=}')
    # sel_lens = {k:len(v) for k,v in obj_selectors['des'].items()}
    # sel_lens = len(obj_selectors['des'])
    # print(f'{sel_lens=}')
    
    
    # obj_0 = pymol_objects['des']
    # for _, obj in pymol_objects.items():
    #     cmd.align(f'{obj} and chain A', f'{obj_0} and chain A')
    # ic(selectors)
    
    
    # print(f'{selectors}')
    # print(

    for i, (label, pymol_name) in enumerate(pymol_objects.items(), start=1):
        selectors = obj_selectors[label]
        sels = combine_selectors([pymol_name], selectors)
        # print(f'{sels=}')
        shown = sels.pop(f'{pymol_name}_shown')
        cmd.show_as('licorice', shown)
        
        # print(f'{label=}')
        # print(f'{len(obj_selectors[label])}')
        # # print(f'{sels}')
        # print(f'{len(sels)}')
        palette = show_tip_row.color_selectors(sels, verbose=False, des_color=des_color)
        # cmd.set('grid_slot', i, obj)
    # cmd.set('grid_mode', 1)
    cmd.unbond('chain A', 'chain B')

    cmd.alter('name CA', 'vdw=2.0')
    cmd.set('sphere_transparency', 0.1)
    cmd.show('spheres', 'name CA')
    if af2:
        cmd.color('white', pymol_objects['af2'])
    return pymol_objects, obj_selectors