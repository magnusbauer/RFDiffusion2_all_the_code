import copy
from datetime import datetime
import torch
from assertpy import assert_that
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import data_loader
from icecream import ic

import rf2aa.chemical
from rf2aa.chemical import NAATOKENS, MASKINDEX, NTOTAL, NHEAVYPROT
import rf2aa.util
import rf2aa.data_loader
from rf2aa.util_module import XYZConverter
from rf2aa.RoseTTAFoldModel import RoseTTAFoldModule
from rf2aa.kinematics import xyz_to_c6d, c6d_to_bins, xyz_to_t2d, get_chirals
import rf2aa.parsers
import rf2aa.tensor_util
import rf_diffusion.aa_model as aa_model
import dataclasses

from rf_diffusion.kinematics import get_init_xyz
from rf_diffusion.contigs import ContigMap
from rf_diffusion.inference import utils as iu
from rf_diffusion.potentials.manager import PotentialManager
from rf_diffusion.inference import symmetry
import logging
import torch.nn.functional as nn
import rf_diffusion.util as util
import hydra
from hydra.core.hydra_config import HydraConfig
from rf_se3_diffusion.data import all_atom
import rf_se3_diffusion.data.utils as du
from openfold.utils.rigid_utils import Rigid
from rf_se3_diffusion import rf_score
from rf_se3_diffusion.rf_score.model import RFScore
from rf_se3_diffusion.data import se3_diffuser
from rf_diffusion import features
import os
from rf_diffusion import noisers
from rf_diffusion.config import config_format

import sys

# When you import this it causes a circular import due to the changes made in apply masks for self conditioning
# This import is only used for SeqToStr Sampling though so can be fixed later - NRB
# import data_loader 
import rf_diffusion.model_input_logger as model_input_logger
from rf_diffusion.model_input_logger import pickle_function_call

def idealize_peptide_frames(indep, generator=None):
    indep = copy.deepcopy(indep)
    rigids = du.rigid_frames_from_atom_14(indep.xyz)
    atom37 = all_atom.atom37_from_rigid(rigids, generator=generator)
    # Not sure if this clone is necessary
    atom37 = torch.clone(atom37)
    indep.xyz = atom37[:,:14]
    return indep

def add_fake_peptide_frame(indep, generator=None):
    indep = copy.deepcopy(indep)
    indep.xyz = aa_model.add_fake_frame_legs(indep.xyz, indep.is_sm, generator=generator)
    return idealize_peptide_frames(indep, generator=generator)

def sample_init(
        conf,
        contig_map,
        target_feats,
        diffuser,
        insert_contig,
        diffuse_all,
        frame_legs_rng=None):
    """Initial features to start the sampling process.
    
    Modify signature and function body for different initialization
    based on the config.
    
    Returns:
        xt: Starting positions with a portion of them randomly sampled.
        seq_t: Starting sequence with a portion of them set to unknown.
    """

    L = len(target_feats['pdb_idx'])

    indep_orig, metadata = aa_model.make_indep(conf.inference.input_pdb, conf.inference.ligand, return_metadata=True)
    for_partial_diffusion = conf.diffuser.partial_T != None
    indep, is_diffused, is_seq_masked = insert_contig(
            indep_orig, 
            contig_map,
            metadata=metadata,
            for_partial_diffusion=for_partial_diffusion)
    
    #ic(
    #    aa_model.what_is_diffused(indep, self.is_diffused, self.model_adaptor.atomizer)
    #)
    t_step_input = conf.diffuser.T
    if for_partial_diffusion:
        mappings = contig_map.get_mappings()
        # This is due to the fact that when inserting a contig, the non-motif coordinates are reset.
        if conf.inference.safety.sidechain_partial_diffusion:
            print("You better know what you're doing when doing partial diffusion with sidechains")
        else:
            assert indep.xyz.shape[0] ==  L + torch.sum(indep.is_sm), f"there must be a coordinate in the input PDB for each residue implied by the contig string for partial diffusion.  length of input PDB != length of contig string: {indep.xyz.shape[0]} != {L+torch.sum(indep.is_sm)}"
            assert torch.all(is_diffused[indep.is_sm] == 0), f"all ligand atoms must be in the motif"
        assert (mappings['con_hal_idx0'] == mappings['con_ref_idx0']).all(), f"all positions in the input PDB must correspond to the same index in the output pdb: {list(zip(mappings['con_hal_idx0'], mappings['con_ref_idx0']))=}"
    indep.seq[is_seq_masked] = rf2aa.chemical.MASKINDEX
    # Diffuse the contig-mapped coordinates 
    if for_partial_diffusion:
        t_step_input = conf.diffuser.partial_T
        assert conf.diffuser.partial_T <= conf.diffuser.T

    indep_orig = copy.deepcopy(indep)
    aa_model.centre(indep_orig, is_diffused)
    indep_uncond, indep_cond = aa_model.diffuse_then_add_conditional(conf, diffuser, indep, is_diffused, t_step_input)

    # indep_orig is the starting structure with native C, N, O, CB, etc. positions.  This gets
    # # used for replacing implicit sidechains and O / CB positions.
    # # indep_cond is the starting structure, with fake frame legs added and wonky O, CB,
    # # and sidechain positions resulting from frame-idealization.  This is used to make
    # # an unconditional indep conditional in the ClassifierFreeGuidance sampler.
    return indep_uncond, indep_orig, indep_cond, is_diffused

class Sampler:

    def __init__(self, conf: DictConfig):
        """Initialize sampler.
        Args:
            conf: Configuration.
        """
        self.initialized = False
        self.initialize(conf)
    
    def initialize(self, conf: DictConfig):
        self._log = logging.getLogger(__name__)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        needs_model_reload = not self.initialized or conf.inference.ckpt_path != self._conf.inference.ckpt_path

        # Assign config to Sampler
        self._conf = conf

        # self.initialize_sampler(conf)
        self.initialized=True

        # Assemble config from the checkpoint
        ic(self._conf.inference.ckpt_path)
        weights_pkl = du.read_pkl(
            self._conf.inference.ckpt_path, use_torch=True,
                map_location=self.device)

        # WIP: if the conf must be read from a different checkpoint for backwards compatibility
        if hasattr( self._conf, 'score_model') and hasattr( self._conf.score_model, 'conf_pkl_path') and self._conf.score_model.conf_pkl_path:
            print(f'WARNING: READING CONF FROM NON-MODEL PICKLE: {self._conf.score_model.conf_pkl_path} THIS SHOULD ONLY BE DONE FOR DEBUGGING PURPOSES')
            weights_conf = du.read_pkl(
                self._conf.score_model.conf_pkl_path, use_torch=True,
                    map_location=self.device)['conf']
        else:
            weights_conf = weights_pkl['conf']

        # Merge base experiment config with checkpoint config.
        OmegaConf.set_struct(self._conf, False)
        OmegaConf.set_struct(weights_conf, False)
        self._conf = OmegaConf.merge(
            weights_conf, self._conf)
        config_format.alert_obsolete_options(self._conf)

        self.diffuser = noisers.get(self._conf.diffuser)
        self.model = RFScore(self._conf.rf.model, self.diffuser, self.device)
        
        ema = 'unknown'
        if self._conf.inference.state_dict_to_load == 'final_state_dict':
            ema = False
        elif self._conf.inference.state_dict_to_load == 'model_state_dict':
            ema = True

        if 'final_state_dict' in weights_pkl:
            ic(ema)
            model_weights = weights_pkl[self._conf.inference.state_dict_to_load] # model_state_dict | final_state_dict
        else:
            model_weights = weights_pkl['model']

        self.model.load_state_dict(model_weights)
        self.model.to(self.device)

        # Initialize helper objects
        self.inf_conf = self._conf.inference
        self.contig_conf = self._conf.contigmap
        self.denoiser_conf = self._conf.denoiser
        self.ppi_conf = self._conf.ppi
        self.potential_conf = self._conf.potentials
        self.diffuser_conf = self._conf.diffuser
        self.preprocess_conf = self._conf.preprocess
        self.model_adaptor = aa_model.Model(self._conf)

        # TODO: Add symmetrization RMSD check here
        if self._conf.seq_diffuser.seqdiff is None:
            self.seq_diffuser = None

            assert(self._conf.preprocess.seq_self_cond is False), 'AR decoding does not make sense with sequence self cond'
            self.seq_self_cond = self._conf.preprocess.seq_self_cond

        elif self._conf.seq_diffuser.seqdiff == 'continuous':
            ic('Doing Continuous Bit Diffusion')

            kwargs = {
                     'T': self._conf.diffuser.T,
                     's_b0': self._conf.seq_diffuser.s_b0,
                     's_bT': self._conf.seq_diffuser.s_bT,
                     'schedule_type': self._conf.seq_diffuser.schedule_type,
                     'loss_type': self._conf.seq_diffuser.loss_type
                     }
            self.seq_diffuser = seq_diffusion.ContinuousSeqDiffuser(**kwargs)

            self.seq_self_cond = self._conf.preprocess.seq_self_cond

        else:
            sys.exit(f'Seq Diffuser of type: {self._conf.seq_diffuser.seqdiff} is not known')

        if self.inf_conf.symmetry is not None:
            self.symmetry = symmetry.SymGen(
                self.inf_conf.symmetry,
                self.inf_conf.model_only_neighbors,
                self.inf_conf.recenter,
                self.inf_conf.radius, 
            )
        else:
            self.symmetry = None


        self.converter = XYZConverter()
        self.target_feats = iu.process_target(self.inf_conf.input_pdb, parse_hetatom=False, center=False)
        self.chain_idx = None

        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)
        if self.inf_conf.ppi_design and self.inf_conf.autogenerate_contigs:
            self.ppi_conf.binderlen = ''.join(chain_idx[0] for chain_idx in self.target_feats['pdb_idx']).index('B')

        self.potential_manager = PotentialManager(self.potential_conf, 
                                                  self.ppi_conf, 
                                                  self.diffuser_conf, 
                                                  self.inf_conf)
        
        # Get recycle schedule    
        recycle_schedule = str(self.inf_conf.recycle_schedule) if self.inf_conf.recycle_schedule is not None else None
        self.recycle_schedule = iu.recycle_schedule(self.T, recycle_schedule, self.inf_conf.num_recycles)


        

    def process_target(self, pdb_path):
        assert not (self.inf_conf.ppi_design and self.inf_conf.autogenerate_contigs), "target reprocessing not implemented yet for these configuration arguments"
        self.target_feats = iu.process_target(self.inf_conf.input_pdb)
        self.chain_idx = None

    @property
    def T(self):
        '''
            Return the maximum number of timesteps
            that this design protocol will perform.

            Output:
                T (int): The maximum number of timesteps to perform
        '''
        return self.diffuser_conf.T
    
    def load_checkpoint(self) -> None:
        """Loads RF checkpoint, from which config can be generated."""
        self._log.info(f'Reading checkpoint from {self.ckpt_path}')
        print(f'loading {self.ckpt_path}')
        self.ckpt  = torch.load(
            self.ckpt_path, map_location=self.device)
        print(f'loaded {self.ckpt_path}')


    def load_model(self):
        """Create RosettaFold model from preloaded checkpoint."""

        # for all-atom str loss
        self.ti_dev = rf2aa.util.torsion_indices
        self.ti_flip = rf2aa.util.torsion_can_flip
        self.ang_ref = rf2aa.util.reference_angles
        self.fi_dev = rf2aa.util.frame_indices
        self.l2a = rf2aa.util.long2alt
        self.aamask = rf2aa.util.allatom_mask
        self.num_bonds = rf2aa.util.num_bonds
        self.atom_type_index = rf2aa.util.atom_type_index
        self.ljlk_parameters = rf2aa.util.ljlk_parameters
        self.lj_correction_parameters = rf2aa.util.lj_correction_parameters
        self.hbtypes = rf2aa.util.hbtypes
        self.hbbaseatoms = rf2aa.util.hbbaseatoms
        self.hbpolys = rf2aa.util.hbpolys
        self.cb_len = rf2aa.util.cb_length_t
        self.cb_ang = rf2aa.util.cb_angle_t
        self.cb_tor = rf2aa.util.cb_torsion_t

        # model_param.
        self.ti_dev = self.ti_dev.to(self.device)
        self.ti_flip = self.ti_flip.to(self.device)
        self.ang_ref = self.ang_ref.to(self.device)
        self.fi_dev = self.fi_dev.to(self.device)
        self.l2a = self.l2a.to(self.device)
        self.aamask = self.aamask.to(self.device)
        self.num_bonds = self.num_bonds.to(self.device)
        self.atom_type_index = self.atom_type_index.to(self.device)
        self.ljlk_parameters = self.ljlk_parameters.to(self.device)
        self.lj_correction_parameters = self.lj_correction_parameters.to(self.device)
        self.hbtypes = self.hbtypes.to(self.device)
        self.hbbaseatoms = self.hbbaseatoms.to(self.device)
        self.hbpolys = self.hbpolys.to(self.device)
        self.cb_len = self.cb_len.to(self.device)
        self.cb_ang = self.cb_ang.to(self.device)
        self.cb_tor = self.cb_tor.to(self.device)

        # HACK: TODO: save this in the model config
        self.loss_param = {'lj_lin': 0.75}
        model = RoseTTAFoldModule(
            **self._conf.model,
            aamask=self.aamask,
            atom_type_index=self.atom_type_index,
            ljlk_parameters=self.ljlk_parameters,
            lj_correction_parameters=self.lj_correction_parameters,
            num_bonds=self.num_bonds,
            cb_len = self.cb_len,
            cb_ang = self.cb_ang,
            cb_tor = self.cb_tor,
            lj_lin=self.loss_param['lj_lin'],
            assert_single_sequence_input=True,
            ).to(self.device)
        
        if self._conf.logging.inputs:
            pickle_dir = pickle_function_call(model, 'forward', 'inference', minifier=aa_model.minifier)
            print(f'pickle_dir: {pickle_dir}')
        model = model.eval()
        self._log.info(f'Loading checkpoint.')
        if not self._conf.inference.zero_weights:
            model.load_state_dict(self.ckpt[self._conf.inference.state_dict_to_load], strict=True)
        return model

    def construct_contig(self, target_feats):
        """Create contig from target features."""
        if self.inf_conf.ppi_design and self.inf_conf.autogenerate_contigs:
            seq_len = target_feats['seq'].shape[0]
            self.contig_conf.contigs = [f'{self.ppi_conf.binderlen}',f'B{self.ppi_conf.binderlen+1}-{seq_len}']
        self._log.info(f'Using contig: {self.contig_conf.contigs}')
        # self.contig_conf.contigs = ['']
        if self.contig_conf.contigs == 'whole':
            L = len(target_feats["pdb_idx"])
            self.contig_conf.contigs = [f'{L}-{L}']
        return ContigMap(target_feats, **self.contig_conf)

    def sample_init(self, return_forward_trajectory=False):
        """Initial features to start the sampling process.
        
        Modify signature and function body for different initialization
        based on the config.
        
        Returns:
            xt: Starting positions with a portion of them randomly sampled.
            seq_t: Starting sequence with a portion of them set to unknown.
        """
        self.contig_map = ContigMap(self.target_feats, **self.contig_conf)
        self.frame_legs_rng = copy_rng(torch.default_generator)
        indep, self.indep_orig, self.indep_cond, self.is_diffused = sample_init(self._conf, self.contig_map, self.target_feats, self.diffuser, self.model_adaptor.insert_contig, diffuse_all=False,
                                                            frame_legs_rng=copy_rng(self.frame_legs_rng))

        indep = copy.deepcopy(self.indep_cond)
        return indep

    def symmetrise_prev_pred(self, px0, seq_in, alpha):
        """
        Method for symmetrising px0 output, either for recycling or for self-conditioning
        """
        _,px0_aa = self.converter.compute_all_atom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0_sym,_ = self.symmetry.apply_symmetry(px0_aa.to('cpu').squeeze()[:,:14], torch.argmax(seq_in, dim=-1).squeeze().to('cpu'))
        px0_sym = px0_sym[None].to(self.device)
        return px0_sym

def copy_rng(rng: torch.Generator) -> torch.Generator:
    current_state = torch.get_rng_state()
    rng = torch.Generator()
    rng.set_state(current_state)
    return rng

def peek_rng(rng):
    current_state = rng.get_state()
    # o = rng.random()
    o = torch.rand(1, generator=rng)
    rng.set_state(current_state)
    return o


class NRBStyleSelfCond(Sampler):
    """
    Model Runner for self conditioning in the style attempted by NRB.

    Works for diffusion and flow matching models.
    """

    def sample_step(self, t, indep, rfo, extra):
        '''
        Generate the next pose that the model should be supplied at timestep t-1.
        Args:
            t (int): The timestep that has just been predicted
            seq_t (torch.tensor): (L,22) The sequence at the beginning of this timestep
            x_t (torch.tensor): (L,14,3) The residue positions at the beginning of this timestep
            seq_init (torch.tensor): (L,22) The initialized sequence used in updating the sequence.
        Returns:
            px0: (L,14,3) The model's prediction of x0.
            x_t_1: (L,14,3) The updated positions of the next step.
            seq_t_1: (L) The updated sequence of the next step.
            tors_t_1: (L, ?) The updated torsion angles of the next  step.
            plddt: (L, 1) Predicted lDDT of x0.
        '''
        extra_t1d_names = getattr(self._conf, 'extra_t1d', [])
        t_cont = t/self._conf.diffuser.T
        indep.extra_t1d = features.get_extra_t1d_inference(indep, extra_t1d_names, self._conf.extra_t1d_params, self._conf.inference.conditions, is_gp=indep.is_gp, t_cont=t_cont)
        rfi = self.model_adaptor.prepro(indep, t, self.is_diffused)

        rf2aa.tensor_util.to_device(rfi, self.device)
        seq_init = torch.nn.functional.one_hot(
                indep.seq, num_classes=rf2aa.chemical.NAATOKENS).to(self.device).float()
        seq_t = torch.clone(seq_init)
        seq_in = torch.clone(seq_init)
        # B,N,L = xyz_t.shape[:3]

        ##################################
        ######## Str Self Cond ###########
        ##################################
        do_self_cond = ((t < self._conf.diffuser.T) and (t != self._conf.diffuser.partial_T)) and self._conf.inference.str_self_cond
        if do_self_cond:
            rfi = aa_model.self_cond(indep, rfi, rfo)

        if self.symmetry is not None:
            idx_pdb, self.chain_idx = self.symmetry.res_idx_procesing(res_idx=idx_pdb)

        with torch.no_grad():
            if self.recycle_schedule[t-1] > 1:
                raise Exception('not implemented')
            for rec in range(self.recycle_schedule[t-1]):
                # This is the assertion we should be able to use, but the
                # network's ComputeAllAtom requires even atoms to have N and C coords.
                # aa_model.assert_has_coords(rfi.xyz[0], indep)
                assert not rfi.xyz[0,:,:3,:].isnan().any(), f'{t}: {rfi.xyz[0,:,:3,:]}'
                model_out = self.model.forward_from_rfi(rfi, torch.tensor([t/self._conf.diffuser.T]).to(rfi.xyz.device), use_checkpoint=False)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        # self._log.info(
        #         f'{current_time}: Timestep {t}')

        rigids_t = du.rigid_frames_from_atom_14(rfi.xyz)
        # ic(self._conf.denoiser.noise_scale, do_self_cond)
        rigids_t = self.diffuser.reverse(
            rigid_t=rigids_t,
            rot_score=du.move_to_np(model_out['rot_score'][:,-1]),
            trans_score=du.move_to_np(model_out['trans_score'][:,-1]),
            diffuse_mask=du.move_to_np(self.is_diffused.float()[None,...]),
            t=t/self._conf.diffuser.T,
            dt=1/self._conf.diffuser.T,
            center=self._conf.denoiser.center,
            noise_scale=self._conf.denoiser.noise_scale,
            rigid_pred=model_out['rigids_raw'][:,-1]
        )
        # x_t_1 = all_atom.atom37_from_rigid(rigids_t)
        # x_t_1 = x_t_1[0,:,:14]
        # # Replace the xyzs of the motif
        # x_t_1[~self.is_diffused.bool(), :14] = indep.xyz[~self.is_diffused.bool(), :14]
        # seq_t_1 = seq_t
        # tors_t_1 = torch.ones((self.is_diffused.shape[-1], 10, 2))

        px0 = model_out['atom37'][0, -1]
        px0 = px0.cpu()
        # x_t_1 = x_t_1.cpu()
        # seq_t_1 = seq_t_1.cpu()

        # if self.symmetry is not None:
        #     x_t_1, seq_t_1 = self.symmetry.apply_symmetry(x_t_1, seq_t_1)

        # return px0, x_t_1, seq_t_1, model_out['rfo'], {}
    
        return px0, get_x_t_1(rigids_t, indep.xyz, self.is_diffused), get_seq_one_hot(indep.seq), model_out['rfo'], {}

def get_x_t_1(rigids_t, xyz, is_diffused):
    x_t_1 = all_atom.atom37_from_rigid(rigids_t)
    x_t_1 = x_t_1[0,:,:14]
    # Replace the xyzs of the motif
    x_t_1[~is_diffused.bool(), :14] = xyz[~is_diffused.bool(), :14]
    x_t_1 = x_t_1.cpu()
    return x_t_1

def get_seq_one_hot(seq):
    seq_init = torch.nn.functional.one_hot(
            seq, num_classes=rf2aa.chemical.NAATOKENS).float()
    return seq_init.cpu()
    # seq_t = torch.clone(seq_init)
    # seq_t_1 = seq_t
    # seq_t_1 = seq_t_1.cpu()
    # return seq_t_1

class FlowMatching(Sampler):
    """
    Model Runner for flow matching.
    """

    def get_grads(self, t, indep, rfo, is_diffused, is_diffused_grad):
        '''
        Generate the next pose that the model should be supplied at timestep t-1.
        Args:
            t (int): The timestep that has just been predicted
            seq_t (torch.tensor): (L,22) The sequence at the beginning of this timestep
            x_t (torch.tensor): (L,14,3) The residue positions at the beginning of this timestep
            seq_init (torch.tensor): (L,22) The initialized sequence used in updating the sequence.
        Returns:
            px0: (L,14,3) The model's prediction of x0.
            x_t_1: (L,14,3) The updated positions of the next step.
            seq_t_1: (L) The updated sequence of the next step.
            tors_t_1: (L, ?) The updated torsion angles of the next  step.
            plddt: (L, 1) Predicted lDDT of x0.
        '''
        extra_t1d_names = getattr(self._conf, 'extra_t1d', [])
        t_cont = t/self._conf.diffuser.T
        indep.extra_t1d = features.get_extra_t1d_inference(indep, extra_t1d_names, self._conf.extra_t1d_params, self._conf.inference.conditions, is_gp=indep.is_gp, t_cont=t_cont)
        rfi = self.model_adaptor.prepro(indep, t, is_diffused)
        rf2aa.tensor_util.to_device(rfi, self.device)

        ##################################
        ######## Str Self Cond ###########
        ##################################
        do_self_cond = ((t < self._conf.diffuser.T) and (t != self._conf.diffuser.partial_T)) and self._conf.inference.str_self_cond
        if do_self_cond:
            rfi = aa_model.self_cond(indep, rfi, rfo)

        if self.symmetry is not None:
            idx_pdb, self.chain_idx = self.symmetry.res_idx_procesing(res_idx=idx_pdb)

        with torch.no_grad():
            # assert not rfi.xyz[0,:,:3,:].isnan().any(), f'{t}: {rfi.xyz[0,:,:3,:]}'
            model_out = self.model.forward_from_rfi(rfi, torch.tensor([t/self._conf.diffuser.T]).to(rfi.xyz.device), use_checkpoint=False)

        rigids_t = du.rigid_frames_from_atom_14(rfi.xyz)
        # ic(self._conf.denoiser.noise_scale, do_self_cond)
        trans_grad, rots_grad = self.diffuser.get_grads(
            rigid_t=rigids_t,
            rot_score=du.move_to_np(model_out['rot_score'][:,-1]),
            trans_score=du.move_to_np(model_out['trans_score'][:,-1]),
            diffuse_mask=du.move_to_np(is_diffused_grad.float()[None,...]),
            t=t/self._conf.diffuser.T,
            dt=1/self._conf.diffuser.T,
            center=self._conf.denoiser.center,
            noise_scale=self._conf.denoiser.noise_scale,
            rigid_pred=model_out['rigids_raw'][:,-1]
        )

        px0 = model_out['atom37'][0, -1]
        px0 = px0.cpu()

        return trans_grad, rots_grad, px0, model_out
    
    def get_rigids(indep):
        rigids = du.rigid_frames_from_atom_14(indep.xyz)
        return rigids

    def sample_step(self, t, indep, rfo, extra):
        '''
        Generate the next pose that the model should be supplied at timestep t-1.
        Args:
            t (int): The timestep that has just been predicted
            seq_t (torch.tensor): (L,22) The sequence at the beginning of this timestep
            x_t (torch.tensor): (L,14,3) The residue positions at the beginning of this timestep
            seq_init (torch.tensor): (L,22) The initialized sequence used in updating the sequence.
        Returns:
            px0: (L,14,3) The model's prediction of x0.
            x_t_1: (L,14,3) The updated positions of the next step.
            seq_t_1: (L) The updated sequence of the next step.
            tors_t_1: (L, ?) The updated torsion angles of the next  step.
            plddt: (L, 1) Predicted lDDT of x0.
        '''
        trans_grad, rots_grad, px0, model_out = self.get_grads(t, indep, rfo, self.is_diffused, is_diffused_grad=self.is_diffused)
        trans_dt, rots_dt = self.diffuser.get_dt(t/self._conf.diffuser.T, 1/self._conf.diffuser.T)
        rigids_t = du.rigid_frames_from_atom_14(indep.xyz)[None,...]
        rigids_t = self.diffuser.apply_grads(rigids_t, trans_grad, rots_grad, trans_dt, rots_dt)
    
        return px0, get_x_t_1(rigids_t, indep.xyz, self.is_diffused), get_seq_one_hot(indep.seq), model_out['rfo'], {}

def sampler_selector(conf: DictConfig):
    if conf.inference.model_runner == 'default':
        sampler = Sampler(conf)
    elif conf.inference.model_runner == 'NRBStyleSelfCond':
        sampler = NRBStyleSelfCond(conf)
    elif conf.inference.model_runner == 'FlowMatching':
        sampler = FlowMatching(conf)
    elif conf.inference.model_runner == 'FlowMatching_make_conditional':
        sampler = FlowMatching_make_conditional(conf)
    elif conf.inference.model_runner == 'NRBStyleSelfCond_debug':
        sampler = NRBStyleSelfCond_debug(conf)
    elif conf.inference.model_runner == 'ClassifierFreeGuidance':
        sampler = ClassifierFreeGuidance(conf)
    elif conf.inference.model_runner in globals():
        sampler = globals()[conf.inference.model_runner](conf)
    else:
        raise ValueError(f'Unrecognized sampler {conf.inference.model_runner}')
    return sampler


def assemble_config_from_chk(conf, ckpt) -> None:
    """
    Function for loading model config from checkpoint directly.

    Takes:
        - config file

    Actions:
        - Replaces all -model and -diffuser items
        - Throws a warning if there are items in -model and -diffuser that aren't in the checkpoint
    
    This throws an error if there is a flag in the checkpoint 'config_dict' that isn't in the inference config.
    This should ensure that whenever a feature is added in the training setup, it is accounted for in the inference script.

    JW
    """
    
    # get overrides to re-apply after building the config from the checkpoint
    overrides = []
    if HydraConfig.initialized():
        overrides = HydraConfig.get().overrides.task
        ic(overrides)
    if 'config_dict' in ckpt.keys():
        print("Assembling -model, -diffuser and -preprocess configs from checkpoint")

        # First, check all flags in the checkpoint config dict are in the config file
        for cat in ['model','diffuser','seq_diffuser','preprocess']:
            #assert all([i in self._conf[cat].keys() for i in self.ckpt['config_dict'][cat].keys()]), f"There are keys in the checkpoint config_dict {cat} params not in the config file"
            for key in conf[cat]:
                if key == 'chi_type' and ckpt['config_dict'][cat][key] == 'circular':
                    ic('---------------------------------------------SKIPPPING CIRCULAR CHI TYPE')
                    continue
                try:
                    print(f"USING MODEL CONFIG: self._conf[{cat}][{key}] = {ckpt['config_dict'][cat][key]}")
                    conf[cat][key] = ckpt['config_dict'][cat][key]
                except:
                    print(f'WARNING: config {cat}.{key} is not saved in the checkpoint. Check that conf.{cat}.{key} = {conf[cat][key]} is correct')
        # add back in overrides again
        for override in overrides:
            if override.split(".")[0] in ['model','diffuser','seq_diffuser','preprocess']:
                print(f'WARNING: You are changing {override.split("=")[0]} from the value this model was trained with. Are you sure you know what you are doing?') 
                mytype = type(conf[override.split(".")[0]][override.split(".")[1].split("=")[0]])
                conf[override.split(".")[0]][override.split(".")[1].split("=")[0]] = mytype(override.split("=")[1])
    else:
        print('WARNING: Model, Diffuser and Preprocess parameters are not saved in this checkpoint. Check carefully that the values specified in the config are correct for this checkpoint')     

    print('self._conf:')
    ic(conf)

class FlowMatching_make_conditional(FlowMatching):
    
    def sample_step(self, t, indep, *args, **kwargs):
        indep = aa_model.make_conditional_indep(indep, self.indep_cond, self.is_diffused)
        return super().sample_step(t, indep, *args, **kwargs)

class FlowMatching_make_conditional_diffuse_all(FlowMatching_make_conditional):

    def sample_init(self):
        self.contig_map = ContigMap(self.target_feats, **self.contig_conf)
        self.frame_legs_rng = copy_rng(torch.default_generator)
        indep, self.indep_orig, self.indep_cond, self.is_diffused = sample_init(self._conf, self.contig_map, self.target_feats, self.diffuser, self.model_adaptor.insert_contig, diffuse_all=True,
                                                            frame_legs_rng=copy_rng(self.frame_legs_rng))
        return indep

class ClassifierFreeGuidance(FlowMatching):
    # WIP
    def sample_init(self):
        self.contig_map = ContigMap(self.target_feats, **self.contig_conf)
        self.frame_legs_rng = copy_rng(torch.default_generator)
        indep, self.indep_orig, self.indep_cond, self.is_diffused = sample_init(self._conf, self.contig_map, self.target_feats, self.diffuser, self.model_adaptor.insert_contig, diffuse_all=True,
                                                            frame_legs_rng=copy_rng(self.frame_legs_rng))
        return indep
    
    def sample_step(self, t, indep, rfo, extra):
        extra_out = {}
        uncond_is_diffused = torch.ones_like(self.is_diffused).bool()
        indep_cond = aa_model.make_conditional_indep(indep, self.indep_cond, self.is_diffused)
        with torch.random.fork_rng():
            trans_grad_cond, rots_grad_cond, px0_cond, model_out_cond = self.get_grads(t, indep_cond, extra['rfo_cond'], self.is_diffused, is_diffused_grad=uncond_is_diffused)

        extra_out['rfo_cond'] = model_out_cond['rfo']
        trans_grad, rots_grad, px0_uncond, model_out_uncond = self.get_grads(t, indep, extra['rfo_uncond'], uncond_is_diffused, is_diffused_grad=uncond_is_diffused)
        extra_out['rfo_uncond'] = model_out_uncond['rfo']
        w = self._conf.inference.classifier_free_guidance_scale
        trans_grad = (1-w) * trans_grad + w * trans_grad_cond
        rots_grad = (1-w) * rots_grad + w * rots_grad_cond
        ic(trans_grad_cond[0, ~self.is_diffused])
        trans_dt, rots_dt = self.diffuser.get_dt(t/self._conf.diffuser.T, 1/self._conf.diffuser.T)
        rigids_t = du.rigid_frames_from_atom_14(indep.xyz)
        rigids_t = self.diffuser.apply_grads(rigids_t, trans_grad, rots_grad, trans_dt, rots_dt)

        # TODO: write both px0 trajectories
        px0 = px0_cond
        if w == 0:
            px0 = px0_uncond
        
        x_t_1 = get_x_t_1(rigids_t, indep.xyz, uncond_is_diffused)
        return px0, x_t_1, get_seq_one_hot(indep.seq), extra_out['rfo_cond'], extra_out
