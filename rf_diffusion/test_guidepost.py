import copy
import unittest

import torch
from icecream import ic

from rf_diffusion import atomize
import rf_diffusion.guide_posts as gp
from rf_diffusion import aa_model
from rf_diffusion import test_utils
import rf_diffusion.inference.data_loader

class TestGuidepost(unittest.TestCase):

    def test_guidepost_appends_only(self):
        testcases = [
            (
                {'contigs': ['A510-515']},
                None,
            ),
            (
                {'contigs': ['A510-515']},
                'LG1',
            ),
        ]

        for contig_kwargs, ligand in testcases:
            test_pdb = 'benchmark/input/gaa.pdb'
            conf = test_utils.construct_conf_single(
                    inference=True,
                    config_name='base_training_base_inference',
                    overrides=['inference.contig_as_guidepost=False',
                               f'inference.input_pdb={test_pdb}',
                               '++transforms.names=["Center","AddConditionalInputs"]',
                               '++transforms.configs.Center={}',
                               '++transforms.configs.AddConditionalInputs.p_is_guidepost_example=${inference.contig_as_guidepost}',
                               '++transforms.configs.AddConditionalInputs.guidepost_bonds=${guidepost_bonds}',                               
                               ])
            dataset = rf_diffusion.inference.data_loader.InferenceDataset(conf)
            _, _, indep, _, is_diffused, atomizer, contig_map, t_step_input, _ = next(iter(dataset))
            ic(indep.chirals.shape)
            indep.xyz = atomize.set_nonexistant_atoms_to_nan(indep.xyz, indep.seq)
            indep_init = copy.deepcopy(indep)

            is_ptn = torch.zeros(indep.length()).bool()
            is_ptn[[1,3]] = True
            indep_gp, _, gp_to_ptn_idx0 = gp.make_guideposts(indep, is_ptn, placement='anywhere')
            
            indep_ungp = copy.deepcopy(indep_gp)
            aa_model.pop_mask(indep_ungp, ~indep_ungp.is_gp)
            ic(indep_ungp.length())


            diff = test_utils.cmp_pretty(indep_ungp, indep_init)
            if diff:
                print(diff)
                self.fail(f'{contig_kwargs=} {diff=}')

if __name__ == '__main__':
        unittest.main()
