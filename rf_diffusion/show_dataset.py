#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'
'''Shows the dataset used for training.  Uses a training config

Example usage: python show_dataset.py --config-name=prod_1024 zero_weights=True debug=True wandb=False show_dataset.n=5 
'''
from icecream import ic
import hydra
import numpy as np
import torch
from omegaconf import DictConfig


from rf_diffusion.dev import pymol
pymol.init(pymol_url='http://chesaw.dhcp.ipd:9123')

cmd = pymol.cmd

from rf_diffusion import show
from rf_diffusion import noisers
from rf_diffusion.dev import show_tip_pa
from rf_diffusion import aa_model
from rf_diffusion.conditions.ss_adj.sec_struct_adjacency import SS_HELIX, SS_STRAND, SS_LOOP, SS_SM, ADJ_STRAND_PAIR
from rf_diffusion.conditions import hbond_satisfaction

import rf_diffusion.dev.show_tip_row
# from rf_diffusion.dev.show_tip_row import OR, AND, NOT

def AND(*i):
    i = [f'({e})' for e in i]
    return '('+ ' and '.join(i) + ')'

def OR(*i):
    i = [f'({e})' for e in i]
    return '(' + ' or '.join(i) +')'

def NOT(e):
    return f'not ({e})'

def no_batch_collate_fn(data):
    assert len(data) == 1
    return data[0]


# def show_data(data, stack=False):
#     analyze.sak.clear(cmd)
#     cmd.do('@~/.pymolrc')
#     counter = 1
#     for i, (_, row) in enumerate(data.iterrows()):
#         pdbs = get_pdbs(row)
#         # print(pdbs)
#         pymol_prefix = []
#         for k in ['epoch', 't', 'dataset']:
#             pymol_prefix.append(f"{k}-{row[k]}")
#         pymol_prefix = '_'.join(pymol_prefix) + str(i)
#         pymol_objects = [f"{pymol_prefix}_{typ}" for typ in ['input', 'pred', 'true']]
            
#         pymol_objects = load_pdbs(pdbs, pymol_objects)
#         # print(pymol_objects)
#         atom_names_by_res_idx = get_atom_names_by_res_idx(row)
#         selectors = show_tip_row.get_selectors_2(atom_names_by_res_idx)
#         # ic(selectors)

#         # if atom_names_by_res_idx:
#         shown = selectors.pop('shown')
#         # else:
#         #     shown = '((name C or name N or name CA) or hetatm)'
#         # print(f'{shown=}')
#         cmd.show_as('licorice', shown)
#         # print(f'{selectors=}')
#         for i, obj in enumerate(pymol_objects, start=1):
#             sels = combine_selectors([obj], selectors)
#             palette = show_tip_row.color_selectors(sels)
#             cmd.set('grid_slot', counter, obj)
#             counter += 1
            
#             print(f'{sels=}')
        
#             sidechains = f"{sels['sidechains_diffused']} or {sels['sidechains_motif']}"
#             cmd.alter(sidechains, 'vdw=3.0')
#             cmd.show('sphere', sidechains)
        

    # cmd.set('grid_mode', 1)
    # cmd.unbond('chain A', 'chain B')

    # cmd.alter('name CA', 'vdw=2.0')
    # cmd.set('sphere_transparency', 0.1)
    # cmd.show('spheres', 'name CA')
from rf_diffusion.data_loader import get_fallback_dataset_and_dataloader

@hydra.main(version_base=None, config_path="config/training", config_name="base")
def run(conf: DictConfig) -> None:

    if conf.debug:
        ic.configureOutput(includeContext=True)
    noiser = noisers.get(conf.diffuser)
    
    # mp.cpu_count()-1
    LOAD_PARAM = {
        'shuffle': False,
        # 'num_workers': test_utils.available_cpu_count() - 3,
        'num_workers': 0,
        'pin_memory': True
    }

    # Get the fallback dataset and dataloader
    train_set, train_loader = get_fallback_dataset_and_dataloader(
        conf=conf,
        diffuser=noiser,
        num_example_per_epoch=conf.epoch_size,
        world_size=1,
        rank=0,
        LOAD_PARAM=LOAD_PARAM,
    )

    counter = -1

    show_tip_pa.clear()
    cmd.set('grid_mode', 1)
    for epoch in range(0, conf.n_epoch):
        train_loader.sampler.set_epoch(epoch)
        train_loader.dataset.fallback_sampler.set_epoch(epoch)
        for i, loader_out in enumerate(train_loader):
            ic(epoch, i)
            counter += 1
            indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
            # # For testing deatomization
            # if atomizer:
            #     _ = atomize.deatomize(atomizer, indep)
            item_context = eval(item_context)
            chosen_dataset, _index = item_context['chosen_dataset'], item_context['index']
            for xyz_label, xyz in [
                    ('true', indep.xyz),
                    ('input', rfi.xyz[0,:,:14])
            ]:
                indep.xyz = xyz
                bonds = indep.metadata['covale_bonds']
                # ic(torch.nonzero(indep.bond_feats == 6))
                # ic(torch.nonzero(indep.bond_feats == 7))
                name = f'{xyz_label}_dataset-{chosen_dataset}_mask-{masks_1d["mask_name"]}_gp-{masks_1d["use_guideposts"]}_bonds_{len(bonds)}_{show.get_counter()}'
                print(name)
                mask_by_name = {}
                if conf.show_dataset.show_diffused:
                    show.color_diffused(indep, is_diffused, name=name)
                if conf.show_dataset.show:
                    _, pymol_1d = show.one(indep, None, name=name)
                    show.cmd.do(f'util.cbc {name}')
                    show.cmd.color('orange', f'{name} and hetatm and elem C')

                    point_types = aa_model.get_point_types(indep, atomizer)
                    for point_category, point_mask in {
                        'residue': point_types == aa_model.POINT_RESIDUE,
                        'atomized': np.isin(point_types, [aa_model.POINT_ATOMIZED_BACKBONE, aa_model.POINT_ATOMIZED_SIDECHAIN]),
                        'ligand': point_types == aa_model.POINT_LIGAND,
                    }.items():
                        for diffused_category, diffused_mask in {
                            'diffused': is_diffused,
                            'motif': ~is_diffused,
                        }.items():
                            mask_by_name[f'{point_category}_{diffused_category}'] = torch.tensor(point_mask)*diffused_mask

                    if conf.show_dataset.show_only_backbone:
                        show.show_backbone_spheres(f'{name} and not hetatm')

                if conf.show_dataset.show_ss_cond:
                    ss_t1d_offset = conf.show_dataset.ss_t1d_offset
                    helix = indep.extra_t1d[:,ss_t1d_offset+SS_HELIX].bool()
                    strand = indep.extra_t1d[:,ss_t1d_offset+SS_STRAND].bool()
                    loop = indep.extra_t1d[:,ss_t1d_offset+SS_LOOP].bool()
                    sm = indep.extra_t1d[:,ss_t1d_offset+SS_SM].bool()

                    mask_by_name['ss_helix'] = helix
                    mask_by_name['ss_strand'] = strand
                    mask_by_name['ss_loop'] = loop
                    mask_by_name['ss_sm'] = sm

                if conf.show_dataset.show_adj_strand_pairs:
                    adj_t2d_offset = conf.show_dataset.adj_t2d_offset
                    is_pair = indep.extra_t2d[:,:,adj_t2d_offset+ADJ_STRAND_PAIR].bool()
                    is_pair_1d = is_pair.any(axis=-1)

                    mask_by_name['strand_pair'] = is_pair_1d

                if conf.show_dataset.show_hotspots:
                    hotspot_t1d_offset = conf.show_dataset.hotspot_t1d_offset
                    hotspots = indep.extra_t1d[:,hotspot_t1d_offset].bool() | (indep.extra_t1d[:,hotspot_t1d_offset+1] != 0)
                    antihotspots = indep.extra_t1d[:,hotspot_t1d_offset+2].bool() | (indep.extra_t1d[:,hotspot_t1d_offset+3] != 0)

                    mask_by_name['hotspots'] = hotspots
                    mask_by_name['antihotspots'] = antihotspots

                if conf.show_dataset.get('show_target_hbond_satisfaction', False):
                    satisfaction_t1d_offset = conf.show_dataset.target_hbond_satisfaction_t1d_offset
                    keys = hbond_satisfaction.get_target_hbond_satisfaction_keys_for_t1d()

                    for key in conf.show_dataset.target_hbond_satisfaction_shown_keys:
                        assert key in keys, f'{key} not in target_hbond_satisfaction keys: {keys}'

                        value_index = satisfaction_t1d_offset + keys.index(key) * 2 + 1
                        mask_by_name[key] = indep.extra_t1d[:,value_index].bool()


                if len(mask_by_name) > 0:

                    selectors = {}
                    for mask_name, mask in mask_by_name.items():
                        selectors[mask_name] = AND(name, OR('id 99999', *pymol_1d[mask]))
                    palette = rf_diffusion.dev.show_tip_row.color_selectors(selectors, palette_name='Paired', palette_n_colors=12)


            if conf.show_dataset.n == counter+1:
                break
        if conf.show_dataset.n == counter+1:
            break

    def label_selectors(selectors, palette):
        '''
        Creates labels with the names 'selectors' in pymol around the origin in the colors of the input 'palette'.
        Params:
            selectors: iteratable of string labels
            palette: rf_diffusion.dev.show_tip_row.PymolPalette
        Returns:
            list of pseudoatom names
        '''
        label_pos_top = [20,0,0]
        for i,s in enumerate(selectors):
            cmd.set('label_size', -3)
            label_pos = label_pos_top
            label_pos[1] -= 4
            cmd.pseudoatom(s,'', 'PS1','PSD', '1', 'P',
                    'PSDO', 'PS', -1.0, 1, 0.0, 0.0, '',
                    '', label_pos)
            cmd.set('grid_slot', 0, s)
            cmd.do(f'label {s}, "{s}"')
            color = palette.name(i)
            cmd.set('label_color', color, s)
        return list(selectors.keys())
    
    if conf.show_dataset.show:
        pseudoatoms = label_selectors(selectors, palette)
        for pseudoatom_name in pseudoatoms:
            cmd.set('grid_slot', show.counter+1, pseudoatom_name)


if __name__ == "__main__":
    run()
