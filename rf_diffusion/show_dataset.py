from icecream import ic
import hydra
from omegaconf import DictConfig
from torch.utils import data
from tqdm import tqdm
from rf_diffusion import show

from rf_diffusion import atomize
from rf_diffusion.dev import analyze, show_tip_pa
from rf_diffusion.data_loader import (
    get_train_valid_set, loader_pdb, loader_fb, loader_complex, loader_pdb_fixbb, loader_fb_fixbb, loader_complex_fixbb, loader_cn_fixbb, default_dataset_configs,
    #Dataset, DatasetComplex, 
    DistilledDataset, DistributedWeightedSampler
)
from rf_se3_diffusion.data import se3_diffuser

cmd = analyze.cmd

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

@hydra.main(version_base=None, config_path="config/training", config_name="base")
def run(conf: DictConfig) -> None:

    if conf.debug:
        ic.configureOutput(includeContext=True)
    diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)
    diffuser.T = conf.diffuser.T
    dataset_configs, homo = default_dataset_configs(conf.dataloader, debug=conf.debug)

    for k, dataset_conf in dataset_configs.items():
        print(f'{k:<20}: {len(dataset_conf.ids)}')

    print('Making train sets')
    train_set = DistilledDataset(dataset_configs,
                                    conf.dataloader, diffuser,
                                    conf.preprocess, conf, homo)
    
    
    train_sampler = DistributedWeightedSampler(dataset_configs,
                                                dataset_options=conf.dataloader['DATASETS'],
                                                dataset_prob=conf.dataloader['DATASET_PROB'],
                                                num_example_per_epoch=conf.epoch_size,
                                                num_replicas=1, rank=0, replacement=True)
    set_epoch = train_sampler.set_epoch
    ic(conf.show_dataset)
    if conf.use_nonechucks:
        import nonechucks as nc
        train_set = nc.SafeDataset(train_set)
        train_sampler = nc.SafeSampler(train_set, train_sampler)
    
    # mp.cpu_count()-1
    LOAD_PARAM = {
        'shuffle': False,
        # 'num_workers': test_utils.available_cpu_count() - 3,
        'num_workers': 0,
        'pin_memory': True
    }
    n_validate=conf.show_dataset.n
    train_loader = data.DataLoader(train_set, sampler=train_sampler, batch_size=conf.batch_size, collate_fn=no_batch_collate_fn, **LOAD_PARAM)
    counter = 0

    show_tip_pa.clear()
    cmd.set('grid_mode', 1)
    for epoch in range(0, conf.n_epoch):
        set_epoch(epoch)
        for i, loader_out in enumerate(train_loader):
            ic(epoch, i)
            counter += 1
            indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_out
            item_context = eval(item_context)
            chosen_dataset, index = item_context['chosen_dataset'], item_context['index']
            ic('loader out', chosen_dataset, index)
            bonds = indep.metadata['covale_bonds']
            name = f'{chosen_dataset}_true_bonds_{len(bonds)}_{show.get_counter()}'
            print(name)
            if conf.show_dataset.show_diffused:
                show.color_diffused(indep, is_diffused, name=name)
            if conf.show_dataset.show:
                show.one(indep, None, name=name)
                show.cmd.do(f'util.cbc {name}')
                show.cmd.color('orange', f'{name} and hetatm and elem C')
                show.cmd.show('licorice')

            if atomizer:
                _ = atomize.deatomize(atomizer, indep)

            # print('-------------------------------------------------------------------------------')

if __name__ == "__main__":
    run()
