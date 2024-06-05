import copy
import unittest

import torch
from icecream import ic

import atomize
import guide_posts as gp
import aa_model
import test_utils
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
                               f'inference.input_pdb={test_pdb}'])
            dataset = rf_diffusion.inference.data_loader.InferenceDataset(conf)
            _, _, indep, _, is_diffused, atomizer, contig_map, t_step_input = next(iter(dataset))
            ic(indep.chirals.shape)
            indep.xyz = atomize.set_nonexistant_atoms_to_nan(indep.xyz, indep.seq)
            indep_init = copy.deepcopy(indep)

            is_ptn = torch.zeros(indep.length()).bool()
            is_ptn[[1,3]] = True
            indep_gp, _, gp_to_ptn_idx0 = gp.make_guideposts(indep, is_ptn, placement='anywhere')
            is_gp = torch.zeros(indep_gp.length()).bool()
            for k in gp_to_ptn_idx0:
                is_gp[k] = True
            
            indep_ungp = copy.deepcopy(indep_gp)
            aa_model.pop_mask(indep_ungp, ~is_gp)
            ic(indep_ungp.length())
            

            diff = test_utils.cmp_pretty(indep_ungp, indep_init)
            if diff:
                print(diff)
                self.fail(f'{contig_kwargs=} {diff=}')

if __name__ == '__main__':
        unittest.main()
