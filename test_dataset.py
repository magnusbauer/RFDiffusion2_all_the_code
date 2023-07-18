import itertools
import os
import sys
from functools import partial
import unittest

import assertpy
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
import show

cmd = analyze.cmd


weird_item = """{
    'chosen_dataset': 'sm_compl_covale',
    'mask_gen_seed': 79592293,
    'sel_item': {   'ASSEMBLY': 1,
                    'CHAINID': '6ndp_B',
                    'CLUSTER': 25998,
                    'COVALENT': [   (   ('B', '19', 'CYS', 'SG'),
                                        ('D', '900', 'BLA', 'CBC'))],
                    'DEPOSITION': '2018-12-14',
                    'HASH': '062428',
                    'LEN_EXIST': 602,
                    'LIGAND': [('D', '900', 'BLA')],
                    'LIGATOMS': 43,
                    'LIGATOMS_RESOLVED': 43,
                    'LIGXF': [('D', 3)],
                    'PARTNERS': [   (   'B',
                                        1,
                                        498,
                                        1.8329198360443115,
                                        'polypeptide(L)'),
                                    (   'A',
                                        0,
                                        0,
                                        13.859391212463379,
                                        'polypeptide(L)')],
                    'RESOLUTION': 3.89,
                    'SEQUENCE': 'MHHHHHHSTATNPLDLDVCAREPIHIPGLIQPYGVLLVIDPADGRIVQASTTAADLLGVPMAALLGMPYTQVLTLPEAQPFAVDDQPQHLMHAEVRFPQRATPPASAWVAAWHLYPQQWLVEMEPRDARLLDVTLREAMPLLRSVERDPGIAEAAVRVAKGLRSLIGFDRVMIYRFDEEWNGDIIAEARKPELEAYLGQHYPASDIPAQARALYLRNRVRQIADVGYQPSPIQPTVHPQLGTPVDLSDVSLRSVSPVHLEYLANMGVTATLVASIVVNDALWGLISCHHYSPHFTNHAMRDVTDAVARTLAGRIGALQAVARARLESVLLTVREKLITDFNDAEHMTVELLDDMAPDLMDVVDADGVAIFHGNDISRHGTTPDVAALRRIRDHIESEHHEALREDAVGALHVDAIGEVFPELADLAPLAAGFIFVPLMPQSRSALLWTRREQIQQIKWAGNPQLAKLEDIPNSRLSPRKSFDLWQQTVRGRARRWSPLHLESARSLRVLIELMERKRFQQDFTLLEASLSRLRDGVAIIERGTANAAHRLLFVNTAFADVCGSDVAELIGRELQTLYASDAPRANVELLQDALRNGRAAYVTLPLQVSDGAPVYRQFHLEPLPSPSGVTAHWLLQLRDPE',
                    'SUBSET': 'covale'},
    'task': 'diff',
}"""

bridge_item = """{
    'chosen_dataset': 'sm_compl_covale',
    'mask_gen_seed': 78231548,
    'sel_item': {   'ASSEMBLY': 2,
                    'CHAINID': '5fvb_P',
                    'CLUSTER': 2189,
                    'COVALENT': [   (   ('P', '48', 'CYS', 'SG'),
                                        ('SB', '203', 'PEB', 'CAA')),
                                    (   ('P', '59', 'CYS', 'SG'),
                                        ('SB', '203', 'PEB', 'CAD'))],
                    'DEPOSITION': '2016-02-05',
                    'HASH': '067414',
                    'LEN_EXIST': 183,
                    'LIGAND': [('SB', '203', 'PEB')],
                    'LIGATOMS': 43,
                    'LIGATOMS_RESOLVED': 43,
                    'LIGXF': [('SB', 40)],
                    'PARTNERS': [   (   'P',
                                        9,
                                        446,
                                        1.8050484657287598,
                                        'polypeptide(L)'),
                                    (   [('NA', '201', 'PEB')],
                                        [('NA', 27)],
                                        0,
                                        7.117906093597412,
                                        'nonpoly'),
                                    (   'F',
                                        5,
                                        0,
                                        7.5266852378845215,
                                        'polypeptide(L)'),
                                    (   'A',
                                        0,
                                        0,
                                        7.633793354034424,
                                        'polypeptide(L)'),
                                    (   [('RB', '202', 'PEB')],
                                        [('RB', 39)],
                                        0,
                                        12.504525184631348,
                                        'nonpoly'),
                                    (   'D',
                                        3,
                                        0,
                                        12.61732292175293,
                                        'polypeptide(L)'),
                                    (   [('QB', '201', 'PEB')],
                                        [('QB', 38)],
                                        0,
                                        12.873053550720215,
                                        'nonpoly'),
                                    (   [('Z', '202', 'PEB')],
                                        [('Z', 13)],
                                        0,
                                        15.891844749450684,
                                        'nonpoly'),
                                    (   'R',
                                        11,
                                        0,
                                        20.833248138427734,
                                        'polypeptide(L)')],
                    'RESOLUTION': 1.93,
                    'SEQUENCE': 'MLDAFSRAVVQADASTSVVADMGALKQFIAEGNRRLDAVNAIASNASCMVSDAVAGMICENQGLIQAGGXCYPNRRMAACLRDAEIILRYVTYALLAGDASVLDDRCLNGLKETYAALGVPTTSTVRAVQIMKAQAAAHIKDTPSEARAGGKLRKMGSPVVEDRCASLVAEASSYFDRVISALS',
                    'SUBSET': 'covale'},
    'task': 'diff',
}"""


multi_covale = """{
    'chosen_dataset': 'sm_compl_multi',             
    'mask_gen_seed': 82644955,            
    'sel_item': {   'ASSEMBLY': 1,                         
                    'CHAINID': '1btc_A',           
                    'CLUSTER': 18655,                         
                    'COVALENT': [],                
                    'DEPOSITION': '1993-02-18',
                    'HASH': '101639',                     
                    'LEN_EXIST': 491,               
                    'LIGAND': [   ('B', '1', 'GLC'),          
                                  ('B', '2', 'GLC'),
                                  ('B', '3', 'GLC'),
                                  ('B', '4', 'GLC'),       
                                  ('B', '5', 'GLC'), 
                                  ('B', '6', 'GLC')],         
                    'LIGATOMS': 66,                
                    'LIGATOMS_RESOLVED': 66,
                    'LIGXF': [('B', 1)],                   
                    'PARTNERS': [   (   'A',        
                                        0,  
                                        216,                                                                                                                                                                                                                                              
                                        2.691878318786621,
                                        'polypeptide(L)'),    
                                    (   [('F', '504', 'BME')],                                                                                                                                                                                                                            
                                        [('F', 5)],
                                        0,                                                                                                                                                                                                                                                
                                        5.804349899291992,
                                        'nonpoly'),                                                                                                                                                                                                                                       
                                    (   [('D', '502', 'BME')],
                                        [('D', 3)],                                                                                                                                                                                                                                       
                                        0,                
                                        6.248571395874023,                                                                                                                                                                                                                                
                                        'nonpoly')],
                    'RESOLUTION': 2.0,                                                                                                                                                                                                                                                    
                    'SEQUENCE': 'SNMLLNYVPVYVMLPLGVVNVDNVFEDPDGLKEQLLQLRAAGVDGVMVDVWWGIIELKGPKQYDWRAYRSLFQLVQECGLTLQAIMSFHQCGGNVGDIVNIPIPQWVLDIGESNHDIFYTNRSGTRNKEYLTVGVDNEPIFHGRTAIEIYSDYMKSFRENMSDFLESGLIIDIEVGLGPAGELRYPSYPQSQGWEFPRIGEFQCYDKYLKADFKAAVARAGHPEWELPDDAGKYNDVPESTGFFKSNGTYVTEKGKFFLTWYSNKLLNHGDQILDEANKAFLGCKVKLAIKVSGIHWWYKVENHAAELTAGYYNLNDRDGYRPIARMLSRHHAILNFTCLEMRDSEQPSDAKSGPQELVQQVLSGGWREDIRVAGENALPRYDATAYNQIILNAKPQGVNNNGPPKLSMFGVTYLRLSDDLLQKSNFNIFKKFVLKMHADQDYCANPQKYNHAITPLKPSAPKIPIEVLLEATKPTLPFPWLPETDMKVDG',                                      
                    'SUBSET': 'multi'},
    'task': 'diff',
}"""

def no_batch_collate_fn(data):
    assert len(data) == 1
    return data[0]

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

    def cmp(self, a, b):
        cmp = partial(tensor_util.cmp, atol=1e-20, rtol=1e-5)
        return cmp(a, b)

    def tearDown(self) -> None:
        hydra.core.global_hydra.GlobalHydra().clear()
        return super().tearDown()

    def indep_for_dataset(self, dataset, mask, epoch=0, overrides=[]):

        show_tip_pa.clear()
        cmd.set('grid_mode', 1)
        conf = test_utils.construct_conf([
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
            # indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_out
            # indep.metadata = None
            return loader_out
        
    
    def test_uncond_sm(self):
        dataset = 'sm_complex'
        mask = 'get_unconditional_diffusion_mask'
        
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['dataloader.CROP=60'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_outindep = loader_out
        indep.metadata = None

        golden_name = f'indep_{dataset}-{mask}'
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)

    def test_atomized_sm(self):
        dataset = 'sm_complex'
        mask = 'get_closest_tip_atoms'
        
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['dataloader.CROP=60'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_outindep = loader_out
        indep.metadata = None

        golden_name = f'indep_{dataset}-{mask}'
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)

    
    def test_covale_uncond(self):
        dataset = 'sm_compl_covale'
        mask = 'get_unconditional_diffusion_mask'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['guidepost_bonds=false'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_out
        indep.metadata = None
        
        golden_name = f'indep_{dataset}-{mask}'
        cmp_ = partial(tensor_util.cmp, atol=1e-3, rtol=1e-5)
        def cmp(got, want):
            got.idx[-13:] = -1
            want.idx[-13:] = -1
            return self.cmp(got, want)
        show.one(indep, atomizer)
        cmd.show('licorice', 'resi 491')
        
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=cmp)

    def test_simple_mask(self):
        dataset = 'pdb_aa'
        mask = 'get_diffusion_mask_simple'
        
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['dataloader.CROP=60'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_outindep = loader_out
        indep.metadata = None
        
        golden_name = f'indep_{dataset}-{mask}'
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)
    
    def test_covale_uncond_weird_covale_bonds(self):
        '''
        The atomized C and the residue C have a distance of 0.03A.
        Currently we are handling this by having the identity cutoff be <0.1A,
        But this begs further investigation, as this makes no sense: atomized coordinates should be exact copies.

        dist = torch.cdist...
        is_weird_distance = (dist > 1e-4) * (dist < 1e-1)
        '''
        dataset = 'sm_compl_covale'
        mask = 'get_unconditional_diffusion_mask'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['guidepost_bonds=false', f'spoof_item="{weird_item}"'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_out
        bonds = indep.metadata['covale_bonds']
        assertpy.assert_that(len(bonds)).is_equal_to(1)

        golden_name = f'indep_{dataset}-{mask}-weird'
        show.one(indep, None, golden_name)
        show.one(indep, atomizer, golden_name+'_deatomized')
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)
        
    def test_covale_uncond_bridge(self):
        '''
        Covale which is bonded to 2 residues.
        '''
        dataset = 'sm_compl_covale'
        mask = 'get_unconditional_diffusion_mask'
        
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['guidepost_bonds=false', f'spoof_item="{bridge_item}"'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_out
        bonds = indep.metadata['covale_bonds']
        assertpy.assert_that(len(bonds)).is_equal_to(2)

        golden_name = f'indep_{dataset}-{mask}-bridge'
        show.one(indep, None, golden_name)
        show.one(indep, atomizer, golden_name+'_deatomized')
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)

    def test_covale_uncond_bridge_gpbonds(self):
        '''
        Covale which is bonded to 2 residues.
        '''
        dataset = 'sm_compl_covale'
        mask = 'get_unconditional_diffusion_mask'
        
        loader_out = self.indep_for_dataset(dataset, mask, overrides=[f'spoof_item="{bridge_item}"'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_out
        bonds = indep.metadata['covale_bonds']
        assertpy.assert_that(len(bonds)).is_equal_to(2)
        assertpy.assert_that(indep.same_chain.all()).is_true()

        golden_name = f'indep_{dataset}-{mask}-bridge-gpbonds'
        show.one(indep, None, golden_name)
        show.one(indep, atomizer, golden_name+'_deatomized')
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)

    def test_covale_islands_partial_ligand(self):
        dataset = 'sm_compl_covale'
        mask = 'get_diffusion_mask_islands_partial_ligand'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['guidepost_bonds=false'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_out
        indep.metadata = None
        
        golden_name = f'indep_{dataset}-{mask}'
        cmp_ = partial(tensor_util.cmp, atol=1e-3, rtol=1e-5)
        def cmp(got, want):
            got.idx[-13:] = -1
            want.idx[-13:] = -1
            return self.cmp(got, want)
        ic((~is_diffused).nonzero()[:,0])
        ic(np.unique(indep.chains()))
        show.color_diffused(indep, is_diffused)
        
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=cmp)
    
    def test_compl_islands(self):
        dataset = 'compl'
        mask = 'get_diffusion_mask_islands'
        loader_out = self.indep_for_dataset(dataset, mask)
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_out
        indep.metadata = None
        
        golden_name = f'indep_{dataset}-{mask}'
        ic((~is_diffused).nonzero()[:,0])
        ic(indep.chains())
        unique_chains = np.unique(indep.chains())
        assertpy.assert_that(len(unique_chains)).is_equal_to(2)
        name, names = show.one(indep, None)
        show.cmd.do('util.cbc')
        show.diffused(indep, is_diffused, 'true')
        
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)

    # def test_multi_covale(self):
    #  WIP
    #     dataset = 'sm_compl_multi'
    #     mask = 'get_unconditional_diffusion_mask'
    #     loader_out = self.indep_for_dataset(dataset, mask, overrides=['dataloader.DATAPKL_AA=aa_dataset_256_subsampled_10_2.pkl'])
    #     indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_out
    #     indep.metadata = None
        
    #     golden_name = f'indep_{dataset}-{mask}'
    #     ic((~is_diffused).nonzero()[:,0])
    #     ic(indep.chains())
    #     unique_chains = np.unique(indep.chains())
    #     assertpy.assert_that(len(unique_chains)).is_equal_to(2)
    #     name, names = show.one(indep, None)
    #     show.cmd.do('util.cbc')
    #     show.diffused(indep, is_diffused, 'true')
        
    #     test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)

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

if __name__ == '__main__':
        unittest.main()
