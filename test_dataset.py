import itertools
import os
import sys
from functools import partial
import unittest

from icecream import ic
import copy
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
from torch.utils import data
from tqdm import tqdm
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RF2-allatom'))
import atomize
from dev import analyze, show_tip_pa
from data_loader import (
    get_train_valid_set, loader_pdb, loader_fb, loader_complex, loader_pdb_fixbb, loader_fb_fixbb, loader_complex_fixbb, loader_cn_fixbb, default_dataset_configs,
    #Dataset, DatasetComplex, 
    DistilledDataset, DistributedWeightedSampler
)
from data import se3_diffuser
import test_utils
from rf2aa import tensor_util
import run_inference
import aa_model

cmd = analyze.cmd

def no_batch_collate_fn(data):
    assert len(data) == 1
    return data[0]


def construct_conf(overrides, config_name='debug'):
    # overrides = overrides + ['inference.cautious=False', 'inference.design_startnum=0']
    initialize(version_base=None, config_path="config/training", job_name="test_app")
    conf = compose(config_name=f'{config_name}.yaml', overrides=overrides, return_hydra_config=True)
    # This is necessary so that when the model_runner is picking up the overrides, it finds them set on HydraConfig.
    # HydraConfig.instance().set_config(conf)
    # conf = compose(config_name='aa_small.yaml', overrides=overrides)
    return conf

def get_dataloader(conf: DictConfig, epoch=0) -> None:

    if conf.debug:
        ic.configureOutput(includeContext=True)
    diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)
    diffuser.T = conf.diffuser.T
    dataset_configs, homo = default_dataset_configs(conf.dataloader, debug=conf.debug)

    print('Making train sets')
    train_set = DistilledDataset(dataset_configs,
                                    conf.dataloader, diffuser,
                                    conf.preprocess, conf, homo)
    
    train_sampler = DistributedWeightedSampler(dataset_configs,
                                                dataset_options=conf.dataloader['DATASETS'],
                                                dataset_prob=conf.dataloader['DATASET_PROB'],
                                                num_example_per_epoch=conf.epoch_size,
                                                num_replicas=1, rank=0, replacement=True)
    train_sampler.epoch = epoch
    
    # mp.cpu_count()-1
    LOAD_PARAM = {
        'shuffle': False,
        'num_workers': 0,
        'pin_memory': True
    }
    n_validate=1
    train_loader = data.DataLoader(train_set, sampler=train_sampler, batch_size=conf.batch_size, collate_fn=no_batch_collate_fn, **LOAD_PARAM)
    return train_loader

REWRITE=False
# REWRITE=True
class Dataloader(unittest.TestCase):

    def tearDown(self) -> None:
        hydra.core.global_hydra.GlobalHydra().clear()
        return super().tearDown()

    def indep_for_dataset(self, dataset, mask, epoch=0, overrides=[]):

        show_tip_pa.clear()
        cmd.set('grid_mode', 1)
        conf = construct_conf([
            'dataloader.DATAPKL_AA=aa_dataset_256_subsampled_10.pkl',
            'dataloader.CROP=256',
            f'dataloader.DATASETS={dataset}',
            f'dataloader.DATASET_PROB=[1.0]',
            f'dataloader.DIFF_MASK_PROBS=null',
            f'dataloader.DIFF_MASK_PROBS={{{mask}:1.0}}',
            'spoof_item=null',
        ] + overrides, config_name='debug')
        dataloader = get_dataloader(conf, epoch)
        for loader_out in dataloader:
            indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_out
            indep.metadata = None
            return loader_out
        
    
    def test_uncond_sm(self):
        dataset = 'sm_complex'
        mask = 'get_unconditional_diffusion_mask'
        
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['dataloader.CROP=60'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_outindep = loader_out


        
        golden_name = f'indep_{dataset}-{mask}'
        cmp = partial(tensor_util.cmp, atol=1e-20, rtol=1e-5)
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=cmp)

    def test_atomized_sm(self):
        dataset = 'sm_complex'
        mask = 'get_closest_tip_atoms'
        
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['dataloader.CROP=60'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_outindep = loader_out


        
        golden_name = f'indep_{dataset}-{mask}'
        cmp = partial(tensor_util.cmp, atol=1e-20, rtol=1e-5)
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=cmp)

    
    # @pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
    def test_sm_compl_covale_uncond(self):
        dataset = 'sm_compl_covale'
        mask = 'get_unconditional_diffusion_mask'
        loader_out = self.indep_for_dataset(dataset, mask)
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_out
        indep.metadata = None

        print(item_context)
        ic(indep.is_sm)
        ic(indep.is_sm.sum())
        sm_bond_feats = indep.bond_feats[indep.is_sm]
        inter_bond_feats = sm_bond_feats[:, ~indep.is_sm]
        ic(inter_bond_feats)
        ic(inter_bond_feats.nonzero())
        for i in inter_bond_feats.nonzero():
            print(i, inter_bond_feats[i[0], i[1]])
        ic(inter_bond_feats.any())
        ic(~is_diffused)
        
        golden_name = f'indep_{dataset}-{mask}'
        cmp_ = partial(tensor_util.cmp, atol=1e-3, rtol=1e-5)
        def cmp(got, want):
            # i = np.arange(want.length())
            # i = np.concatenate((i[:-13], i[-7:], i[-13:-7]))
            # aa_model.rearrange_indep(want, i)
            # want.bond_feats *= want.bond_feats != 6
            # got.xyz = got.xyz[:,:14]
            # want.xyz[want.is_sm, 0] = 0
            # want.xyz[want.is_sm, 2:] = 0
            # got.xyz[want.is_sm, 0] = 0
            # got.xyz[want.is_sm, 2:] = 0
            # ic(
            #     list(zip(
            #         np.arange(13),
            #         np.arange(got.length())[-13:],
            #         aa_model.human_readable_seq(got.seq[-13:]),
            #     )),
            #     # list(zip(got.atom_frames, want.atom_frames)),
            #     list(zip(
            #         got.bond_feats[-13:,-13:].nonzero(),
            #         want.bond_feats[-13:,-13:].nonzero(),
            #     )),
            # )

            # def standardize_frames(atom_frames):
            #     o = atom_frames.clone()
            #     for i, f in enumerate(atom_frames):
            #         if f[0,0] < f[2,0]:
            #             continue
            #         o[i, 0, 0] = atom_frames[i, 2, 0]
            #         o[i, 2, 0] = atom_frames[i, 0, 0]
            #     return o
            
            # got.atom_frames = standardize_frames(got.atom_frames)
            # want.atom_frames = standardize_frames(want.atom_frames)

            # for got_f, want_f in zip(
            #         got.human_readable_atom_frames(),
            #         want.human_readable_atom_frames(),
            #     ):
            #     if got_f == want_f:
            #         print('MATCH')
            #     else:
            #         print(f'got:  ', got_f)
            #         print(f'want: ', want_f)

            # got.atom_frames = None
            # want.atom_frames = None
            # got.idx[-13:] = -1
            # want.idx[-13:] = -1

            # ic(
            #     list(zip(
            #         want.human_readable_seq(),
            #         want.idx,
            #         got.idx,
            #     ))
            # )

            return cmp_(got, want)
        show(indep, atomizer)
        cmd.show('licorice', 'resi 491')
        
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=cmp)

    def test_simple_mask(self):
        dataset = 'pdb_aa'
        mask = 'get_diffusion_mask_simple'
        
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['dataloader.CROP=60'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_outindep = loader_out
        ic(indep.length())


        
        golden_name = f'indep_{dataset}-{mask}'
        cmp = partial(tensor_util.cmp, atol=1e-20, rtol=1e-5)
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=cmp)
    
    # def test_covale_simple_mask_0(self):
    #     dataset = 'sm_compl_covale'
    #     mask = 'get_diffusion_mask_simple'
    #     loader_out = self.indep_for_dataset(dataset, mask, epoch=0)
    #     indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_out

    #     print(item_context)
    #     ic(indep.chains())
    #     ic(indep.is_sm)
    #     ic(indep.is_sm.sum())
    #     sm_bond_feats = indep.bond_feats[indep.is_sm]
    #     inter_bond_feats = sm_bond_feats[:, ~indep.is_sm]
    #     ic(inter_bond_feats)
    #     ic(inter_bond_feats.nonzero())
    #     for i in inter_bond_feats.nonzero():
    #         # print(i)
    #         # print(inter_bond_feats.shape)
    #         print(i, inter_bond_feats[i[0], i[1]])
    #     ic(inter_bond_feats.any())
    #     ic(~is_diffused)
        
    #     golden_name = f'indep_{dataset}-{mask}'
    #     cmp = partial(tensor_util.cmp, atol=1e-20, rtol=1e-5)
    #     show(indep, atomizer)
    #     test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=cmp)

    # def test_covale_simple_mask_1(self):
    #     dataset = 'sm_compl_covale'
    #     mask = 'get_diffusion_mask_simple'
    #     loader_out = self.indep_for_dataset(dataset, mask, epoch=5)
    #     indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_out

    #     print(item_context)
    #     ic(indep.chains())
    #     ic(indep.is_sm)
    #     ic(indep.is_sm.sum())
    #     sm_bond_feats = indep.bond_feats[indep.is_sm]
    #     inter_bond_feats = sm_bond_feats[:, ~indep.is_sm]
    #     ic(inter_bond_feats)
    #     ic(inter_bond_feats.nonzero())
    #     for i in inter_bond_feats.nonzero():
    #         # print(i)
    #         # print(inter_bond_feats.shape)
    #         print(i, inter_bond_feats[i[0], i[1]])
    #     ic(inter_bond_feats.any())
    #     ic(~is_diffused)
        
    #     golden_name = f'indep_{dataset}-{mask}'
    #     cmp = partial(tensor_util.cmp, atol=1e-20, rtol=1e-5)
    #     show(indep, atomizer)
    #     show_diffused(indep, is_diffused, 'true')
    #     test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=cmp)

from dev import analyze, show_tip_pa
cmd = analyze.cmd
def show(indep, atomizer):

    if atomizer:
        indep = atomize.deatomize(atomizer, indep)
    pdb = f'/tmp/true.pdb'
    indep.write_pdb(pdb)

    cmd.load(pdb)
    name = os.path.basename(pdb[:-4])
    cmd.show_as('cartoon', name)
    cmd.show('licorice', f'hetatm and {name}')
    cmd.color('orange', f'hetatm and elem c and {name}')

def show_diffused(indep, is_diffused, name):

    indep_motif = copy.deepcopy(indep)
    indep_diffused = copy.deepcopy(indep)
    aa_model.pop_mask(indep_motif, ~is_diffused)
    aa_model.pop_mask(indep_diffused, is_diffused)
    aa_model.show(indep_motif, None, f'{name}_motif')
    aa_model.show(indep_diffused, None, f'{name}_diffused')

if __name__ == '__main__':
        unittest.main()
