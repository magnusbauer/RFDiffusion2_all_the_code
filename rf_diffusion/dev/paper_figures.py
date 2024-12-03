import itertools

import json
import re
from rf_diffusion.dev import show_tip_row
from rf_diffusion.dev import analyze


from rf_diffusion.dev import pymol
# pymol.init('http://10.64.100.67:9123')

import os



from rf_diffusion.dev import show_bench
from rf_diffusion.dev import show_tip_pa

from rf_diffusion.dev.pymol import cmd
from rf_diffusion.dev import paper_vis


from rf_diffusion.dev.paper_vis import *
pymol.init('http://localhost:9123')

def show_benchmark_objs(
        bm,
        input_dir='/home/ahern/reclone/rf_diffusion_staging/rf_diffusion/benchmark/input',
        n_show=20,
        palette_json=default_palette_json,
        prefix='',
        palette_v=None,
        ):
    palette = get_pymol_palette(palette_json)
    if palette_v is None:
        palette_v = show_tip_row.PymolPalette(cmd, 'viridis', start_val=0, stop_val=n_show*2)

    pdb_contig = []
    for name, args in itertools.islice(bm.items(), n_show):
        contig = re.search('contigs=\[\\\\\'(.*)\\\\\'\]', args).groups()[0]
        contigs = [c for c in contig.split(',') if c[0].isalpha()]
        motif = []
        for c in contigs:
            ch = c[0]
            a, b = list(map(int, c[1:].split('-')))
            for i in range(a, b+1):
                motif.append((ch, i))
        input_pdb = re.search('input_pdb=(.*\.pdb)\s', args).groups()[0]
        input_pdb = os.path.join(input_dir, input_pdb)
        ligand = re.search('inference.ligand=\\\\\'([^=]*)\\\\\'', args).groups()[0]
        contig_atoms = re.search('contigmap.contig_atoms=\"(.*)\"', args).groups()[0]
        contig_atoms = eval(eval(contig_atoms))
        pdb_contig.append((input_pdb, tuple(sorted(motif)), ligand, contig_atoms))

    structures = []
    tip_selectors = []
    inv_rotamer_selectors = []
    objs = []
    for i, (pdb, motif, ligands, contig_atoms) in enumerate(pdb_contig):
        native =  prefix + os.path.splitext(os.path.basename(pdb))[0]
        ref_idx = motif
        ligand_selector = f'({" or ".join("(resn " + l + ")" for l in ligands.split(","))})'
        cmd.do(f'load {pdb}, {native}; remove ({native} and not ({ligand_selector} or {analyze.to_selector(ref_idx)}))')
        cmd.show_as('licorice', 'not hetatm')
        cmd.set('sphere_transparency', 0.)
        tip_selector = []
        for k, v in contig_atoms.items():
            ch = k[0]
            resi = k[1:]
            atom_names = v.split(',')
            tip_selector.append(show_tip_row.get_atom_selector(native, ch, resi, atom_names))
        
        tip_selector = ' or '.join(tip_selector)
        tip_selectors.append(tip_selector)
        
        ligand_color = palette.name(0)
        inv_rotamer_selector = f'{native} and not hetatm and not ({tip_selector})'
        inv_rotamer_selectors.append(inv_rotamer_selector)
        cmd.set('stick_transparency', 0.3, inv_rotamer_selector)
        
        cmd.color(ligand_color, 'hetatm')
        cmd.color(palette_v.name(i), inv_rotamer_selector )
        cmd.color(palette.name(1), tip_selector)

        # cmd.set('stick_color', ligand_color, 'hetatm and elem C')
        # cmd.set('stick_color', palette.name(0), f'{native} and not hetatm')
        # cmd.set('stick_color', palette.name(4), tip_selector)
        # cmd.set('stick_color', 'grey90', 'resi 1')
        structures.append(analyze.Structure(native, motif))

        objs.append(show_tip_pa.PymolObj(native, dict(
            sidechains_motif=tip_selector,
            sidechains_diffused=inv_rotamer_selector,
            lig='hetatm'
        )))
        # break

    cmd.center(structures[0].name)
    cmd.zoom(structures[0].name)
    for s in structures[1:]:
        cmd.align(s.name, structures[0].name)

    cmd.unbond('hetatm', 'not hetatm')
    cmd.remove('hydrogens')

    cmd.unbond('hetatm', 'not hetatm')
    cmd.remove('hydrogens')
    cmd.set('valence', 1)
    return objs

def color_inv_rot(objs, palette_v=None):
    if palette_v is None:
        palette_v = show_tip_row.PymolPalette(cmd, 'viridis', start_val=0, stop_val=len(objs)*2)
    for i,o in enumerate(objs):
        cmd.set('stick_color', palette_v.name(i), o.name)
        cmd.set('sphere_color', palette_v.name(i), o.name)
        cmd.set('stick_transparency', 0.3, o.name)
        cmd.set('stick_transparency', 0.0, o['lig'])
        cmd.set('stick_transparency', 0.0, o['sidechains_motif'])

from collections import OrderedDict

# analyze.clear()


spectrum_palette_json = dict(list(default_palette_json.items())[:2])
non_spectrum_palette_json = dict(list(default_palette_json.items())[2:])

def show_rfd_processing(df, bm_filt):
    set_pymol_settings()
    cmd.set('grid_mode', 1)
    obj_by_panel = {}


    # Panel 1: Only motif
    grid_slot = 1
    objs = show_benchmark_objs(bm_filt, palette_json=non_spectrum_palette_json, n_show=1, prefix='panel_1_')
    obj_by_panel[grid_slot] = objs
    e = objs[0]
    cmd.remove(e['sidechains_diffused'])

    # Panel 2: Inverse rotamers
    grid_slot += 1
    n_show = 5
    palette_v = show_tip_row.PymolPalette(cmd, 'plasma', start_val=0, stop_val=1.5*n_show - 1)
    objs = show_benchmark_objs(bm_filt, palette_json=non_spectrum_palette_json, n_show=n_show, prefix='panel_2_', palette_v=palette_v)
    obj_by_panel[grid_slot] = objs
    color_inv_rot(objs, palette_v)
    for o in objs:
        cmd.set('grid_slot', grid_slot, o.name)

    # Panel 3: Inverse rotamer selected
    grid_slot += 1
    objs = show_benchmark_objs(bm_filt, palette_json=non_spectrum_palette_json, n_show=1, prefix='panel_3_')
    obj_by_panel[grid_slot] = objs
    e = objs[0]
    e.selectors['sidechains_motif'] = f"({e['sidechains_motif']}) OR ({e['sidechains_diffused']})"
    color_inv_rot(objs)
    for o in objs:
        cmd.set('grid_slot', grid_slot, o.name)

    # Panel 4-7: Sequence placement
    n_motif = 4
    backbone = '(name CA or name N or name C)'
    for i in range(n_motif):
        grid_slot += 1
        objs = show_benchmark_objs(bm_filt, palette_json=non_spectrum_palette_json, n_show=1, prefix=f'panel_{grid_slot}_')
        obj_by_panel[grid_slot] = objs
        e = objs[0]
        e.selectors['sidechains_motif'] = f"({e['sidechains_motif']}) OR ({e['sidechains_diffused']})"
        color_inv_rot(objs)
        cmd.remove(f"{e.name} AND NOT resi {i+1}")
        cmd.set('grid_slot', grid_slot, e.name)

        if i == 0:
            first_model = e
        else:
            cmd.do(f"align {e.name} and {backbone}, {first_model.name} and {backbone}")

    # Panel 8: Sequence placed
    # grid_slot += 1
    # objs = show_benchmark_objs(bm_filt, palette_json=non_spectrum_palette_json, n_show=1, prefix='panel_8_')
    # obj_by_panel[grid_slot] = objs
    # e = objs[0]
    # e.selectors['sidechains_motif'] = f"({e['sidechains_motif']}) OR ({e['sidechains_diffused']})"
    # color_inv_rot(objs)
    # for o in objs:
    #     cmd.set('grid_slot', grid_slot, o.name)


    # Global coloring
    for i, objs in obj_by_panel.items():
        for e in objs:
            paper_vis.fix_spheres()
            paper_vis.color_lig_sidechain(e)
            paper_vis.color_lig_atom_spheres(e)
            paper_vis.color_motif_atom_spheres(e)


    # Panel 8: Sequence placed
    grid_slot += 1
    all_entities = show_bench.show_df(
        df,
        structs={},
        # structs={'X0'},
        # structs={'unidealized'},
        # structs={'unidealized'},
        # structs={'unidealized'},
        des=True,
        af2=False,
        mpnn_packed=False,
        return_entities=True,
    )
    obj_by_panel[grid_slot] = list(all_entities[0].values())
    e = all_entities[0]['des']
    e.selectors['sidechains_motif'] = e.selectors['residue_motif']
    paper_vis.visualize_design_entities(all_entities)
    cmd.color('paper_teal', e['sidechains_motif'])
    cmd.set('grid_slot', grid_slot, e.name)
    extended_selection = 2
    cmd.hide('cartoon', f"{e.name}")
    cmd.show('cartoon', f"{e['sidechains_motif']} extend {extended_selection}")
    cmd.cartoon('tube',  f"{e['sidechains_motif']} extend {extended_selection}")

    # Panel 9: design
    grid_slot += 1
    all_entities = show_bench.show_df(
        df,
        structs={},
        # structs={'X0'},
        # structs={'unidealized'},
        # structs={'unidealized'},
        # structs={'unidealized'},
        des=True,
        af2=False,
        mpnn_packed=False,
        return_entities=True,
    )
    obj_by_panel[grid_slot] = list(all_entities[0].values())
    e = all_entities[0]['des']
    e.selectors['sidechains_motif'] = e.selectors['residue_motif']
    paper_vis.visualize_design_entities(all_entities)
    for entities in all_entities:
        for e in entities.values():
            cmd.color('paper_teal', e['sidechains_motif'])
            cmd.set('grid_slot', grid_slot, e.name)

    motif_obj = obj_by_panel[1][0]
    cmd.center(motif_obj.name)
    paper_vis.set_pymol_settings()

    cmd.alter("name CA", 'vdw=2.5')
    cmd.do('rebuild')
    cmd.set('ray_opaque_background', 0)

    return obj_by_panel

def load_rfdiffusion():

    df = analyze.combine('/net/scratch/ahern/se3_diffusion/benchmarks/2024-08-16_01-49-10_rfd_enzyme_3_ec_invrot/compiled_metrics.csv')
    enzyme_class = 'enzyme_class'
    def get_enzyme_class(row):
        for i in range(1, 6):
            if f'ec{i}' in row['benchmark'] or f'EC{i}' in row['benchmark']:
                return f'EC{i}'
        if row['benchmark'] == 'retroaldolase':
            return 'retroaldolase'
        raise Exception(f"{row['benchmark']=} not recognized as one of EC1-5 or retroaldolase")

    df[enzyme_class] = df.apply(get_enzyme_class, axis=1)
    df = df[df[enzyme_class] == 'EC2']
    df = df.nsmallest(n=1, columns='backbone_aligned_allatom_rmsd_af2_ref_all_sym_resolved')

    bm_path = '/home/ahern/reclone/rf_diffusion_staging/rf_diffusion/benchmark/ecbench_invrot_newEC2_relative.json'

    with open(bm_path) as fp:
        bm = json.load(fp)
    
    n_per_ec = 1000

    bm_filt = {}
    for k, v in bm.items():
        if k[-1] != '2':
            continue
        if 500 <= int(k.split('_')[2]) <= 500 + n_per_ec:
            bm_filt[k] = v
            bm_filt[k] += "contigmap.contig_atoms=\"'{\\'A1\\':\\'OD1,CG,CB,OD2\\',\\'B2\\':\\'OD2,CG\\',\\'C3\\':\\'NE,CZ,CD\\',\\'D4\\':\\'NH2,CZ,NH1,NE\\'}'\"`"
    
    # Put the theozyme corresponding to the design first.
    bm_filt = OrderedDict(bm_filt.items())

    design_benchmark = df.iloc[0].benchmark
    bm_filt.move_to_end(design_benchmark, last=False)

    return df, bm_filt

def show_rfflow_trajectory_snapshots(show, struct='Xt', ts=[100, 30, 1], start_grid_slot=2, suffix='', diffused_sidechain_palette = show_tip_row.PymolPalette(cmd, 'Set1', start_val=0, stop_val=9), include_xt=False):

    all_entities = show_bench.show_df(
        show,
        # structs={},
        # structs={'Xt'},
        structs={struct, 'unidealized'},
        # structs={'unidealized'},
        # structs={'unidealized'},
        des=False,
        af2=False,
        mpnn_packed=False,
        return_entities=True,
    )
    if suffix == 'test':
        return all_entities


    cmd.unbond('hetatm', 'not hetatm')
    entities = all_entities[0]

    for i, k in enumerate([struct, 'unidealized'], start=1):
        e = entities[k]
        cmd.set('grid_slot', i, e.name)

    cmd.set('grid_mode', 1)
    paper_vis.visualize_design_entities(all_entities, pretty=False)
    xt = entities[struct]

    # cmd.unbond(f'{xt.name} and not hetatm', 'not hetatm')
    xt_keep_bonded_selection = f"{xt['sidechains_diffused']} OR {xt['sidechains_motif']} OR {xt['lig']}"
    xt_backbone_selection = f"{xt.name} and NOT ({xt_keep_bonded_selection})"
    # cmd.unbond(f"NOT ({keep_bonded_selection})", f"NOT ({keep_bonded_selection})")

    if struct == 'Xt':
        cmd.unbond(xt_backbone_selection, xt_backbone_selection)
        for i in range(1, 181):
            cmd.bond(f'resi {i} AND name CA', f'resi {i} AND (name C OR name N)')
    # paper_vis.visualize_design_entities(all_entities)
    # cmd.show('licorice', f"{xt.name}")
    cmd.show('licorice', f"({xt_backbone_selection}) AND (name C or name CA or name N)")
    # cmd.hide('licorice', f"{}")
    cmd.show('spheres', f"({xt_backbone_selection}) AND (name CA)")
    cmd.show('licorice', xt['sidechains_diffused'])
    cmd.show('spheres', f"({xt['sidechains_diffused']}) AND (name CA)")
    cmd.show('licorice', xt['sidechains_diffused'])
    xt['sidechains_diffused']
    
    def hacky_get_guidepost_resi(sidechains_diffused):
        resi = re.findall(r'resi\s(\d+)', sidechains_diffused)
        # return f"resi {'+'.join(resi)}"
        return [f"resi {i}" for i in resi]

    hacky_get_guidepost_resi(xt['sidechains_diffused'])
    for i, resi in enumerate(hacky_get_guidepost_resi(xt['sidechains_diffused'])):
        # cmd.color(diffused_sidechain_palette.name(i), f"({resi}) AND ({xt['sidechains_diffused']})")
        cmd.color(diffused_sidechain_palette.name(i), f"(({resi}) AND ({xt['sidechains_diffused']})) extend 1")
        xt.selectors[f'resi_{i}'] = f"({resi})"


    # # To redo states, uncomment
    # for s in states:
    #     cmd.delete(s)
    states = []
    for i in ts:
        state = f'{struct}{suffix}_state_{i}'
        cmd.create(state, xt.name, i, 1)
        states.append(state)
    # for i, name in enumerate(states + [entities['unidealized'].name], start=start_grid_slot):
    #     cmd.set('grid_slot', i, name)
    
    out_entities = []
    if include_xt:
        out_entities.append(xt)
    else:
        print(f'deleting {xt.name=}')
        cmd.delete(xt.name)
    for s in states:
        out_entities.append(show_tip_pa.PymolObj(s, xt.selectors))
    out_entities.append(entities['unidealized'])

    # for e in out_entities:
    #     paper_vis.visualize_design(e)
    
    obj_by_panel = {}
    for i, e in enumerate(out_entities, start=start_grid_slot):
        cmd.set('grid_slot', i, e.name)
        obj_by_panel[i] = [e]


    for entities in obj_by_panel.values():
        for e in entities:
            cmd.set('cartoon_side_chain_helper', 0, e.name)

    cmd.disable(xt.name)

    return obj_by_panel

def load_rfflow():
    df = analyze.combine(
        '/net/scratch/ahern/se3_diffusion/benchmarks/2024-08-15_19-17-14_demo_cfg_center_all_finetune_ec_4/compiled_metrics.csv',
    )

    enzyme_class = 'enzyme_class'
    def get_enzyme_class(row):
        for i in range(1, 6):
            if f'ec{i}' in row['benchmark'] or f'EC{i}' in row['benchmark']:
                return f'EC{i}'
        if row['benchmark'] == 'retroaldolase':
            return 'retroaldolase'
        raise Exception(f"{row['benchmark']=} not recognized as one of EC1-5 or retroaldolase")

    df[enzyme_class] = df.apply(get_enzyme_class, axis=1)
    df = df[df[enzyme_class] == 'EC2']
    successes = df[
        (df['backbone_aligned_allatom_rmsd_af2_unideal_all_sym_resolved'] < 1.5) &
        (df['rmsd_af2_des'] < 2.5)
    ]
    successes = successes.sort_values('backbone_aligned_allatom_rmsd_af2_unideal_all_sym_resolved')
    show = successes
    show = show.drop_duplicates('design_id')

    show_examples = df[
        df['name'] == 'run_ec2_M0151_cond2_656-atomized-bb-False'
    ]
    show = show_examples.iloc[7:8]

    return show


## Pymol visualizations for submotif fitting comparison across conditioning types

import matplotlib as mpl
def subsample_colormap(cmap, I):
    return [cmap(i) for i in I]


def register_subsampled_colormap(cmap_name, I):

    subsampled_pastels = subsample_colormap(mpl.colormaps[cmap_name], I)
    cmap_name = f'{cmap_name}_{"-".join(map(str, I))}'
    subsampled_colormap = mpl.colors.ListedColormap(subsampled_pastels, name=cmap_name)
    mpl.colormaps.register(subsampled_colormap, force=True)
    return cmap_name

def get_sidechain_palette():
    cmap_name = register_subsampled_colormap('Paired', [2,6,4,0])
    # mpl.colormaps[cmap_name]
    cmap = mpl.colormaps[cmap_name]
    return show_tip_row.PymolPalette(cmd, cmap_name, 0, cmap.N)

# Entry point
def show_submotif_fitting_comparison(show, ts=[100, 95, 90], motif_i=[0,3], grid_slot_iterator=iter_grid_slot(), hide_style='hide', display_x0_style='cartoon'):

    # grid_slot_iterator = iter_grid_slot()
    # ts = [100, 95, 90]
    # ts = [100, 95, 0]
    # ts = [95]
    grouped_entities = show_trajectory_with_input(show, ts, grid_slot_iterator)
    cmd.set('grid_mode', 1)

    for e in get_x0_entities(grouped_entities):
        hide_X0_cartoon(e)
    for e in get_x0_entities(grouped_entities):
        hide_not_m1m2(e, show.iloc[0], motif_i=motif_i, hide_style=hide_style, display_x0_style=display_x0_style)
    
    for e in get_x0_entities(grouped_entities):
        cmd.cartoon('tube', e.name)

    is_bb = 'backbone' in show.iloc[0]['sweep']
    if is_bb:
        show_motif_bb(grouped_entities)
        for e in get_x0_entities(grouped_entities):
            hide_not_m1m2(e, show.iloc[0], motif_i=motif_i, hide_style=hide_style)

    return grouped_entities

def show_trajectory_with_input(show, ts, grid_slot_iterator,diffused_sidechain_palette = None):
    if diffused_sidechain_palette is None:
        diffused_sidechain_palette = get_sidechain_palette()

    entity_by_grid_slot = show_rfflow_trajectory_snapshots(show, ts=ts, struct='X0',
        diffused_sidechain_palette=diffused_sidechain_palette, suffix=show.iloc[0]['sweep'])
    # entity_by_grid_slot = show_rfflow_trajectory_snapshots(show, ts=ts, struct='X0',
    #     diffused_sidechain_palette=diffused_sidechain_palette)
    
    # input_entity = list(entity_by_grid_slot.values())[0][0]

    # Input:
    all_entities = show_bench.show_df(show, structs={'unidealized'}, return_entities=True)
    paper_vis.visualize_design_entities(all_entities, pretty=False)

    e_input = all_entities[0]['unidealized']
    # input_sel = f"({e_input.selectors['sidechains_motif']} or het)"
    input_sel = f"({e_input.selectors['sidechains_motif']} or {e_input.selectors['residue_motif']} or {e_input.selectors['residue_gp_motif']} or het)"

    e_input.selectors['sidechains_motif'] = input_sel
    # input_sel = f"({e.selectors['sidechains_motif']})"
    # cmd.color('red', input_sel)
    cmd.hide('everything', f"{e_input.name} and not {input_sel}")

    
    # entity_dict_list = []
    # entity_dict_list['input'] = entity_by_grid_slot['input']
    grouped_entities = list(entity_by_grid_slot.values())
    grouped_entities.insert(0, [e_input])
    # return list(entity_by_grid_slot.values())
    for ee in grouped_entities:
        slot = next(grid_slot_iterator)
        for e in ee:
            cmd.set('grid_slot', slot, e.name)
    return grouped_entities

def hide_X0_cartoon(e):
    cmd.hide('cartoon', e.name)

def get_x0_entities(grouped_entities):
    o = []
    for ee in grouped_entities:
        for e in ee:
            if '_state' in e.name:
                o.append(e)
    return o

def hide_not_m1m2(e, row, motif_i=[0,3], hide_style='hide', display_x0_style='cartoon'):
    trb = analyze.get_trb(row)
    traj_motif_idx = trb['con_hal_pdb_idx']
    _, res_start = traj_motif_idx[motif_i[0]]
    _, res_end = traj_motif_idx[motif_i[1]]
    if res_start > res_end:
        res_start, res_end = res_end, res_start
    sel_total = f"resi {res_start}-{res_end}"
    sel = f"resi {res_start+1}-{res_end-1}"
    # sel = sel_total


    atom_names_by_res_idx = show_tip_pa.get_motif_spec(row, traj=True)
    # motif_atom_names_by_res_idx = tuple(atom_names_by_res_idx)
    # motif_atom_names_by_res_idx = motif_atom_names_by_res_idx[e]
    motif_atom_names_by_res_idx = {}
    for i, (k, v) in enumerate(atom_names_by_res_idx.items()):
        if i in motif_i:
            motif_atom_names_by_res_idx[k] = v
    # motif_selectors = show_tip_row.get_selectors_2(motif_atom_names_by_res_idx)
    submotif_selector = ' OR '.join(f"resi {k}" for k in motif_atom_names_by_res_idx.keys())
    print(f'{submotif_selector=}')

    shown = f"{e.name} and ({sel_total} or {submotif_selector})"
    # notshown = f"{e.name} and ({sel} or {submotif_selector})"
    not_shown = f"{e.name} and not ({shown})"
    print(f'{not_shown=}')

    # sel_bb = f"{e.name} and ({sel}) AND (name CA or name N or name C)"
    sel_bb = f'{e.name} and {sel}'
    if display_x0_style == 'cartoon':
        # cmd.show_as('cartoon', f"{e.name} and {sel}")
        cmd.show_as('cartoon', f"{sel_bb}")
        print(f'showing cartoon for f"{e.name} and {sel_total}"')
        cmd.show('cartoon', f"{e.name} and {sel_total}")
    elif display_x0_style == 'cartoon_spheres':
        cmd.show_as('cartoon', f"{sel_bb}")
        cmd.show('spheres', f"{e.name} and {sel} and name CA")
    else:
        raise ValueError(f"{display_x0_style=} not recognized")


    if hide_style == 'hide':
        cmd.hide('everything', not_shown)
    elif hide_style == 'transparent':
        transparency_not_shown = 0.95
        cmd.set('stick_transparency', transparency_not_shown, not_shown)
        cmd.set('sphere_transparency', transparency_not_shown, not_shown)
    else:
        raise ValueError(f"{hide_style=} not one of 'hide' or 'transparent'")


def hacky_get_guidepost_resi(selector):
    resi = re.findall(r'resi\s(\d+)', selector)
    placeholder_resi = '9999'
    return [f"resi {i}" for i in resi if i != placeholder_resi]

def show_motif_comparison(e, coarse_grained=True):
    print(f"{e.selectors.keys()=}")
    print(f"{e.selectors['sidechains_motif']=}")
    print(f"{e.selectors['residue_motif']=}")
    print(f"{e.selectors['residue_gp_motif']=}")

    motif_selector = ' OR '.join(e.selectors[s] for s in ['residue_motif', 'residue_gp_motif', 'sidechains_motif'])
    motif_selector = f"{e.name} AND ({motif_selector})"

    if coarse_grained:

        cmd.hide('everything', motif_selector)

        motif_backbone_selector = f"{e.name} AND ({motif_selector}) AND (name CA or name N or name C or name CB)"
        cmd.show('licorice', motif_backbone_selector)
        cmd.color('paper_teal', motif_backbone_selector)

        motif_cb_selector = f"{e.name} AND ({motif_selector}) AND (name CB)"
        cmd.do(f"alter ({motif_cb_selector}), vdw=3.5")
        cmd.show('spheres', motif_cb_selector)

        motif_ca_selector = f"{e.name} AND ({motif_selector}) AND (name CA)"
        cmd.show('spheres', motif_ca_selector)

        individual_residues = hacky_get_guidepost_resi(motif_selector)
        for resi, one_letter in zip(individual_residues, ['R', 'D', 'R', 'H']):
            cb_sel = f"({e.name} and {resi} and name CB)"
            cmd.do(f'label {cb_sel}, "{one_letter}"')
        cmd.set('label_position', [0.0,0.0,5])
        cmd.set('label_size', 10)

def show_motif_bb(grouped_entities):
    for ee in grouped_entities:
        for e in ee:
            show_motif_comparison(e)

    final_e = grouped_entities[-1][0]
    cmd.show('cartoon', f"{final_e.name}")
    cmd.show_as('cartoon', f"{final_e.name} AND (name N or name C)")
    cmd.hide('spheres', f"{final_e.name} AND name CA")
