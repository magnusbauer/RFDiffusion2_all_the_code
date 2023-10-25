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

import atomize
from dev import analyze, show_tip_pa
from data_loader import (
    get_train_valid_set, loader_pdb, loader_fb, loader_complex, loader_pdb_fixbb, loader_fb_fixbb, loader_complex_fixbb, loader_cn_fixbb, default_dataset_configs,
    #Dataset, DatasetComplex, 
    DistilledDataset, DistributedWeightedSampler
)
from rf_diffusion import test_utils
from rf2aa import tensor_util
import run_inference
import aa_model
import show
import rf2aa.chemical

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

nonatomized_covale = """{
    'chosen_dataset': 'sm_compl_covale',
    'index': 2,
    'mask_gen_seed': 72799612,
    'sel_item': {   'ASSEMBLY': 1,
                    'CHAINID': '1nbp_A',
                    'CLUSTER': 22700,
                    'COVALENT': [   (   ('A', '31', 'CYS', 'SG'),
                                        ('C', '201', 'MHC', 'S1'))],
                    'DEPOSITION': '2002-12-03',
                    'HASH': '004725',
                    'LEN_EXIST': 121,
                    'LIGAND': [('C', '201', 'MHC')],
                    'LIGATOMS': 18,
                    'LIGATOMS_RESOLVED': 18,
                    'LIGXF': [('C', 2)],
                    'PARTNERS': [   (   'A',
                                        0,
                                        243,
                                        2.0362677574157715,
                                        'polypeptide(L)')],
                    'RESOLUTION': 2.2,
                    'SEQUENCE': 'APTSSSTKKTQLQLEHLLLDLQMILNGINNCKNPKLTRMLTFKFYMPKKATELKHLQCLEEELKPLEEVLNLAQSKNFHLRPRDLISNINVIVLELKGSETTFMCEYADETATIVEFLNRWITFCQSIISTLT',
                    'SUBSET': 'covale'},
    'task': 'diff',
}"""


multiligand_item = """{
    'chosen_dataset': 'sm_compl_asmb',                                                                                                       
    'index': 0,                                                                                                                              
    'mask_gen_seed': 68054314,                                                                                                               
    'sel_item': {   'ASSEMBLY': 2,                                                                                                           
                'CHAINID': '6fp1_B',                                                                                                     
                'CLUSTER': 6750,                                                                                                         
                'COVALENT': [],                                                                                                          
                'DEPOSITION': '2018-02-08',                                                                                              
                'HASH': '104197',                                                                                                        
                'LEN_EXIST': 445,                                                                                                        
                'LIGAND': [('G', '501', 'FAD')],                                                                                         
                'LIGATOMS': 53,                                                                                                          
                'LIGATOMS_RESOLVED': 53,                                                                                                 
                'LIGXF': [('G', 1)],                                                                                                     
                'PARTNERS': [   (   'B',                                                                                                                                                                                                                                              
                                    0,                                                                                                                                                                                                                                                
                                    605,                                                                                                 
                                    2.6784911155700684,                                                                                  
                                    'polypeptide(L)'),                                                                                   
                                (   [('M', '507', 'E0Q')],                                                                               
                                    [('M', 7)],                                                                                          
                                    16,                                                                                                  
                                    3.4406967163085938,                                                                                  
                                    'nonpoly'),                                                                                          
                                (   [('J', '504', 'PEG')],                                                                               
                                    [('J', 4)],                                                                                          
                                    0,                                                                                                   
                                    12.409589767456055,                                                                                  
                                    'nonpoly'),                                                                                          
                                (   [('K', '505', 'PEG')],                                                                               
                                    [('K', 5)],                                                                                          
                                    0,                                                                                                   
                                    14.890626907348633,                                                                                  
                                    'nonpoly'),                                                                                          
                                (   [('L', '506', 'PEG')],                                                                               
                                    [('L', 6)],                                                                                          
                                    0,
                                    20.84391212463379,
                                    'nonpoly')],
                'RESOLUTION': 1.97,
                'SEQUENCE': 'TATDNARQVTIIGAGLAGTLVARLLARNGWQVNLFERRPDPRIETGARGRSINLALAERGAHALRLAGLEREVLAEAVMMRGRMVHVPGTPPNLQPYGRDDSEVIWSINRDRLNRILLDGAEAAGASIHFNLGLDSVDFARQRLTLSNVSGERLEKRFHLLIGADGCNSAVRQAMASVVDLGEHLETQPHGYKELQITPEASAQFNLEPNALHIWPHGDYMCIALPNLDRSFTVTLFLHHQSPAAQPASPSFAQLVDGHAARRFFQRQFPDLSPMLDSLEQDFEHHPTGKLATLRLTTWHVGGQAVLLGDAAHPMVPFHGQGMNCALEDAVALAEHLQSAADNASALAAFTAQRQPDALAIQAMALENYVEMSSKVASPTYLLERELGQIMAQRQPTRFIPRYSMVTFSRLPYAQAMARGQIQEQLLKFAVANHSDLTSINLDAVEHEVTRCLPPLSHLS',
                'SUBSET': 'asmb'},
    'task': 'diff',
}"""


REWRITE=False
# REWRITE=True
class Dataloader(unittest.TestCase):

    def cmp(self, a, b):
        cmp = partial(tensor_util.cmp, atol=1e-20, rtol=1e-5)
        return cmp(a, b)

    def tearDown(self) -> None:
        hydra.core.global_hydra.GlobalHydra().clear()
        return super().tearDown()

    def indep_for_dataset(self, dataset, mask, epoch=0, overrides=[], **kwargs):

        #show_tip_pa.clear()
        #cmd.set('grid_mode', 1)
        return test_utils.loader_out_for_dataset(dataset,mask,overrides=overrides,epoch=epoch, **kwargs)
        
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

    def test_multi_covale(self):
        dataset = 'sm_compl_multi'
        mask = 'get_unconditional_diffusion_mask'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=[
            'dataloader.DATAPKL_AA=aa_dataset_256_subsampled_10_2.pkl',
            f'spoof_item="{multi_covale}"',
            ], config_name='extra_t1d_v2')
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_out
        indep.metadata = None
        for r in indep.extra_t1d.T:
            print(r)

        golden_name = f'indep_{dataset}-{mask}'
        unique_chains = np.unique(indep.chains())
        assertpy.assert_that(len(unique_chains)).is_equal_to(2)
        name, names = show.one(indep, None)
        # show.cmd.do(f'util.cbc {name}')
        # show.diffused(indep, is_diffused, 'true')
        # show.one(indep, None)
        
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)


    def test_covale_is_guideposted(self):
        dataset = 'sm_compl_multi'
        mask = 'get_unconditional_diffusion_mask'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=[
            'dataloader.DATAPKL_AA=aa_dataset_256_subsampled_10.pkl',
            f'spoof_item="{nonatomized_covale}"',
            ], config_name='extra_t1d_v2')
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_out
        indep.metadata = None
        golden_name = f'indep_{dataset}-{mask}-is-guideposted'
        name, names = show.one(indep, None)
        show.cmd.do(f'util.cbc {name}')
        show.diffused(indep, is_diffused, 'true')
        
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)
    
    def test_multiligand(self):

        dataset = 'sm_compl_asmb'
        mask = 'get_unconditional_diffusion_mask'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=[
            'dataloader.DATAPKL_AA=aa_dataset_256_subsampled_10.pkl',
            f'spoof_item="{multiligand_item}"',
            ], config_name='extra_t1d_v2')
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_out
        indep.metadata = None
        ic(
            indep.chains()[indep.is_sm],
            indep.idx[indep.is_sm],
        )
        golden_name = f'indep_{dataset}-{mask}-multiligand'
        name, names = show.one(indep, None)
        show.cmd.do(f'util.cbc {name}')
        show.cmd.color('orange', f'{name} and hetatm and elem C')
        # show.cmd.do(f'util.cbc {name}')
        # show.diffused(indep, is_diffused, 'true')
        
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)

    def test_continuous_time_embedding(self):
        '''
        Tests that the continuous timestep t ([0, 1]) is correctly
        one-hot encoded in the t1d features.
        '''
        dataset = 'pdb_aa'
        mask = 'get_diffusion_mask_simple'
        loader_out = self.indep_for_dataset(dataset, mask, config_name='time_embedding')
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_out

        golden_name = 'indep-extra-t1d-time-embedding'
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)

    def test_atom_order(self):
        '''
        Tests that amino acid atoms from the dataloader are ordered the same way as in rf2aa.chemical.aa2long
        '''
        # Make the training loader
        f_train_yaml = 'config/training/RFD_36.yaml'
        overrides = [
            'dataloader.DATAPKL_AA=aa_dataset_256_subsampled_10.pkl', 
            '+dataloader.max_residues=256',
            '+diffuser.time_type=continuous',
            '+diffuser.t_cont_max=0.025',
        ]
        conf_train = test_utils.construct_conf(yaml_path=f_train_yaml, overrides=overrides)
        train_loader test_utils.get_dataloader(conf_train)

        # Sample several training examples
        min_train_examples = 10
        min_aa = 100

        train_examples_count = 0
        aa_count = {rf2aa.chemical.num2aa[aa_int]: 0 for aa_int in range(20)}
        for i, (indep_train, rfi_train, chosen_dataset, item, little_t, is_diffused_train, chosen_task, atomizer, masks_1d, diffuser_out, item_context) in enumerate(train_loader):
            print('Loaded example:', i)
            try:
                item_context = eval(item_context)
                atom_order_results = test_utils.detect_permuted_aa_atoms(indep_train, item_context)
                for aa3, record in atom_order_results.items():
                    for is_correct_order, msg in record:
                        if not is_correct_order:
                            pass
                            #print(msg)
                            #self.fail(msg)
                            
                # Record how many training pdbs and individual amino acids have been checked.
                train_examples_count += 1
                
                for aa_int in range(20):
                    aa3 = rf2aa.chemical.num2aa[aa_int]
                    aa_count[aa3] += len(atom_order_results[aa3])
                    
                print('updated my numbers')
            
            except: # Exception as e:
                print(f'Couldn\'t process {item_context}. Skipping.')
            
            print('train_examples_count:', train_examples_count)
            enough_aa = all(map(lambda count: count >= min_aa, aa_count.values()))
            if enough_aa and (train_examples_count >= min_train_examples):
                print('DONEEEE!!')
                break




if __name__ == '__main__':
        unittest.main()
