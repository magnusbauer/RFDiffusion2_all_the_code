import unittest
import os
import torch

from functools import partial
from rf2aa import tensor_util

from rf_diffusion import aa_model
from rf_diffusion import silent_files
from rf_diffusion import import_pyrosetta
from rf_diffusion import test_utils
from rf_diffusion import calc_hbonds
from rf_diffusion.calc_hbonds import HBMAP_OTHER_IDX0, HBMAP_OTHER_IATOM, HBMAP_OUR_IATOM, HBMAP_WE_ARE_DONOR
from rf_diffusion.chemical import ChemicalData as ChemData

from rf_diffusion.import_pyrosetta import pyrosetta as pyro
from rf_diffusion.import_pyrosetta import rosetta as ros


class TestPyrosetta(unittest.TestCase):
    '''
    Tests that involve pyrosetta
    '''

    def setUp(self):
        conf = test_utils.construct_conf()
        import_pyrosetta.prepare_pyrosetta(conf)

    def test_silent_accuracy(self):
        '''
        Ensure that the silent file machinery generates a pdb that is equivalent
        '''
        input_pdb = './test_data/two_chain.pdb'
        ligand_name = ''
        silent_name = 'tmp/out.silent'
        pdb_name = 'tmp/out.pdb'
        tag = 'my_tag'

        indep = aa_model.make_indep(input_pdb, ligand_name)

        if os.path.exists(silent_name):
            os.remove(silent_name)

        silent_files.add_indep_to_silent(silent_name, tag, indep)
        pose, _ = silent_files.read_pose_from_silent(silent_name, tag)
        pose.dump_pdb(pdb_name)

        indep2 = aa_model.make_indep(pdb_name)
        
        cmp = partial(tensor_util.cmp, atol=0.0011, rtol=0)
        diff = cmp(indep, indep2)

        assert not diff


    def test_hbonds(self):
        '''
        This test makes sure that any h-bond found by rosetta is also found by diffusion

        The reverse test doesn't work great because this structure isn't relaxed. Additionally, since diffusion can't see hydrogens
         there are a few extra hydroxyl hbonds that it finds
        '''
        input_pdb = 'benchmark/input/4yl0.pdb'

        debug = None
        # debug = 'hbond_test_' # dump debug pdbs

        pose = pyro().pose_from_file(input_pdb)
        indep, metadata = aa_model.make_indep(input_pdb, ligand='GSH', return_metadata=True)

        hb_scorefxn = ros().core.scoring.ScoreFunctionFactory.create_score_function("none")
        hb_scorefxn.set_weight(ros().core.scoring.hbond_sc, 1)
        hb_scorefxn.set_weight(ros().core.scoring.hbond_sr_bb, 1)
        hb_scorefxn.set_weight(ros().core.scoring.hbond_lr_bb, 1)
        hb_scorefxn.set_weight(ros().core.scoring.hbond_bb_sc, 1)

        opts = hb_scorefxn.energy_method_options()
        opts.hbond_options().decompose_bb_hb_into_pair_energies(True)
        opts.hbond_options().bb_donor_acceptor_check(False)
        hb_scorefxn.set_energy_method_options(opts)

        ros().core.pack.optimizeH(pose, hb_scorefxn)

        hb_scorefxn(pose)
        hbset = ros().core.scoring.hbonds.HBondSet()
        ros().core.scoring.hbonds.fill_hbond_set(pose, False, hbset)
        hbset.hbond_options().bb_donor_acceptor_check(False)
        ros().core.scoring.hbonds.fill_hbond_set(pose, False, hbset)


        indep_chain_length = None
        range_size = 6
        test_range = indep.idx[:range_size]
        for ioff in range(1, indep.length()-range_size):
            if torch.allclose(test_range, indep.idx[ioff:ioff+range_size]):
                indep_chain_length = ioff
                break
        assert indep_chain_length is not None

        N_chains = 3
        assert (metadata['ligand_atom_names'][:indep_chain_length*N_chains] == '').all()
        assert (metadata['ligand_atom_names'][indep_chain_length*N_chains:] != '').all()


        idx0_to_seqpos_protein = {}
        seqpos_to_idx0_protein = {}
        for idx0 in range(indep_chain_length*N_chains):
            name1 = ChemData().one_letter[indep.seq[idx0]]

            chain_letter = 'ABC'[int(idx0 / indep_chain_length)]
            pdb_num = indep.idx[idx0]

            found_it = None
            for seqpos in range(1, pose.size()+1):
                if pose.pdb_info().number(seqpos) != pdb_num:
                    continue
                if pose.pdb_info().chain(seqpos) != chain_letter:
                    continue
                found_it = seqpos
                assert pose.residue(seqpos).name1() == name1
                break
            assert found_it is not None

            idx0_to_seqpos_protein[idx0] = seqpos
            seqpos_to_idx0_protein[seqpos] = idx0


        ligand_natoms = (indep.length() - indep_chain_length*N_chains) / N_chains
        assert ligand_natoms % 1 == 0
        ligand_natoms = int(ligand_natoms)

        ilig_to_seqpos = {}
        seqpos_to_ilig = {}
        for ilig in range(N_chains):
            found_seqpos = None
            count = -1
            for seqpos in range(1, pose.size()+1):
                if pose.residue(seqpos).name() == 'pdb_GSH':
                    count += 1
                    if count == ilig:
                        found_seqpos = seqpos
                        break
            assert found_seqpos is not None
            ilig_to_seqpos[ilig] = found_seqpos
            seqpos_to_ilig[found_seqpos] = ilig

        idx0_to_seqpos_iatom_sm = {}
        seqpos_iatom_to_idx0_sm = {}

        for ilig in range(N_chains):

            offset = indep_chain_length*N_chains + ilig * ligand_natoms
            for idx0 in range(offset, offset + ligand_natoms):
                atom_name = metadata['ligand_atom_names'][idx0]
                seqpos = ilig_to_seqpos[ilig]
                iatom = pose.residue(seqpos).atom_index(atom_name.strip())
                key = (seqpos, iatom)
                idx0_to_seqpos_iatom_sm[idx0] = key
                seqpos_iatom_to_idx0_sm[key] = idx0


        short_atom_names = [[x.strip() if x is not None else '' for x in ChemData().aa2long[aa][:ChemData().NHEAVY]] for aa in range(20)]

        def get_idx0_iatom(seqpos, ros_iatom):
            res = pose.residue(seqpos)

            if seqpos in seqpos_to_ilig:
                key = (int(seqpos), int(ros_iatom))
                assert key in seqpos_iatom_to_idx0_sm
                idx0 = seqpos_iatom_to_idx0_sm[key]
                iatom = 1
            else:
                seq0 = ChemData().one_letter.index(res.name1())
                idx0 = seqpos_to_idx0_protein[int(seqpos)]
                iatom = short_atom_names[seq0].index(res.atom_name(ros_iatom).strip())

            return idx0, iatom

        hbond_map, hbond_scores = calc_hbonds.calculate_hbond_map(indep, debug_pdb_prefix=debug)

        rosetta_hbond_map = torch.full(hbond_map.shape, -1, dtype=int)
        rosetta_hbond_scores = torch.full(hbond_scores.shape, torch.nan, dtype=float)

        for seqpos in range(1, pose.size()+1):
            res = pose.residue(seqpos)
            if int(seqpos) not in seqpos_to_ilig and int(seqpos) not in seqpos_to_idx0_protein:
                continue
            for hbond in hbset.residue_hbonds(seqpos):
                if hbond.don_res() != seqpos:
                    continue
                other_seqpos = hbond.acc_res()
                other_ros_iatom = hbond.acc_atm()
                our_ros_iatom = res.atom_base(hbond.don_hatm())

                if int(other_seqpos) not in seqpos_to_ilig and int(other_seqpos) not in seqpos_to_idx0_protein:
                    continue

                our_idx0, our_iatom = get_idx0_iatom(seqpos, our_ros_iatom)
                other_idx0, other_iatom = get_idx0_iatom(other_seqpos, other_ros_iatom)

                N_our_hbonds = (rosetta_hbond_map[our_idx0,:,HBMAP_OTHER_IDX0] != -1).sum()
                rosetta_hbond_map[our_idx0,N_our_hbonds,HBMAP_OTHER_IDX0] = other_idx0
                rosetta_hbond_map[our_idx0,N_our_hbonds,HBMAP_OTHER_IATOM] = other_iatom
                rosetta_hbond_map[our_idx0,N_our_hbonds,HBMAP_OUR_IATOM] = our_iatom
                rosetta_hbond_map[our_idx0,N_our_hbonds,HBMAP_WE_ARE_DONOR] = True
                rosetta_hbond_scores[our_idx0,N_our_hbonds] = hbond.energy()

                N_other_hbonds = (rosetta_hbond_map[other_idx0,:,HBMAP_OTHER_IDX0] != -1).sum()
                rosetta_hbond_map[other_idx0,N_other_hbonds,HBMAP_OTHER_IDX0] = our_idx0
                rosetta_hbond_map[other_idx0,N_other_hbonds,HBMAP_OTHER_IATOM] = our_iatom
                rosetta_hbond_map[other_idx0,N_other_hbonds,HBMAP_OUR_IATOM] = other_iatom
                rosetta_hbond_map[other_idx0,N_other_hbonds,HBMAP_WE_ARE_DONOR] = False
                rosetta_hbond_scores[other_idx0,N_other_hbonds] = hbond.energy()

        def idx0_iatom_to_pretty_name(idx0, iatom):
            if int(idx0) in idx0_to_seqpos_iatom_sm:
                seqpos, ros_iatom = idx0_to_seqpos_iatom_sm[int(idx0)]
                return f'{pose.pdb_info().chain(seqpos)} {pose.pdb_info().number(seqpos)} {pose.residue(seqpos).atom_name(ros_iatom)}'
            else:
                seqpos = idx0_to_seqpos_protein[int(idx0)]
                one_letter = pose.residue(seqpos).name1()
                return f'{pose.pdb_info().chain(seqpos)} {pose.pdb_info().number(seqpos)} {short_atom_names[ChemData().one_letter.index(one_letter)][iatom]}'


        def compare_hbond_map(ground_map, ground_scores, comp_map, comp_scores, ground_cutoff=-1, prefix=''):

            all_good = True
            for our_idx0 in range(len(ground_map)):
                ground_mask = ground_scores[our_idx0] < ground_cutoff

                for iground in torch.where(ground_mask)[0]:
                    other_idx0, other_iatom, our_iatom, we_are_donor = list(ground_map[our_idx0,iground])

                    other_mask = ( (comp_map[our_idx0,:,HBMAP_OTHER_IDX0] == other_idx0)
                                 & (comp_map[our_idx0,:,HBMAP_OTHER_IATOM] == other_iatom)
                                 & (comp_map[our_idx0,:,HBMAP_OUR_IATOM] == our_iatom)
                                 # & (comp_map[our_idx0,:,3] == we_are_donor) # we don't care about this because diffusion can't see the hydrogens
                                 )

                    
                    if other_mask.sum() == 0:

                        our_name = idx0_iatom_to_pretty_name(our_idx0, our_iatom)
                        other_name = idx0_iatom_to_pretty_name(other_idx0, other_iatom)

                        don_name, acc_name = (our_name, other_name) if we_are_donor else (other_name, our_name)

                        print(f'{prefix} Hbond not found. Donor: {don_name} Acceptor: {acc_name}', f'our_idx0={int(our_idx0)}; our_iatom={int(our_iatom)}; other_idx0={int(other_idx0)}; other_iatom={int(other_iatom)}; we_are_donor={bool(we_are_donor)}')

                        all_good = False


            return all_good


        good = compare_hbond_map(rosetta_hbond_map, rosetta_hbond_scores, hbond_map, hbond_scores, prefix='Ros exists in diff:')

        # Honestly, Rosetta misses a lot of good stuff. Probably because the structure isn't relaxed
        # good = compare_hbond_map(hbond_map, hbond_scores, rosetta_hbond_map, rosetta_hbond_scores, prefix='Diff exists in ros:') and good

        assert good



if __name__ == '__main__':
        unittest.main()
