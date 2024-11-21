#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'
"""
Inference script.

To run with base.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name symmetry

where symmetry can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""
import os

# # Hack for autobenching
# PKG_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# SE3_DIR = os.path.join(PKG_DIR, 'lib/se3_flow_matching')
# sys.path.append(SE3_DIR)

import re
from collections import defaultdict
import time
import pickle
import dataclasses
import torch 
from omegaconf import OmegaConf
import hydra
import logging
from icecream import ic
from hydra.core.hydra_config import HydraConfig
import numpy as np
import random
import glob
from rf_diffusion.inference import model_runners
import rf2aa.tensor_util
import rf2aa.util
from rf_diffusion.chemical import ChemicalData as ChemData
from rf_diffusion.chemical import reinitialize_chemical_data
from rf_diffusion import aa_model
from rf_diffusion import guide_posts as gp
import copy
from rf_diffusion import atomize
from rf_diffusion.dev import idealize_backbone
from rf_diffusion.idealize import idealize_pose
import rf_diffusion.features as features
from rf_diffusion.import_pyrosetta import prepare_pyrosetta
from rf_diffusion import silent_files
import rf_diffusion.inference.utils as iu
import tqdm
import rf_diffusion.atomization_primitives
ic.configureOutput(includeContext=True)

import rf_diffusion.nucleic_compatibility_utils as nucl_utils

logger = logging.getLogger(__name__)

def make_deterministic(seed=0, ignore_if_cuda=False):
    # if not (ignore_if_cuda and torch.cuda.device_count() > 0):
    #     torch.use_deterministic_algorithms(True)
    torch.use_deterministic_algorithms(True)
    seed_all(seed)

def seed_all(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_seeds():
    return {
        'torch': torch.get_rng_state(),
        'np': np.random.get_state(),
        'python': random.getstate(),
    }

@hydra.main(version_base=None, config_path='config/inference', config_name='base')
def main(conf: HydraConfig) -> None:
    if 'custom_chemical_config' in conf:
        reinitialize_chemical_data(**conf.custom_chemical_config)
    prepare_pyrosetta(conf)
        
    sampler = get_sampler(conf)
    sample(sampler)

def get_sampler(conf):
    if conf.inference.deterministic:
        seed_all()

    # Loop over number of designs to sample.
    design_startnum = conf.inference.design_startnum
    if conf.inference.design_startnum == -1:
        existing = glob.glob(conf.inference.output_prefix + '*.pdb')
        indices = [-1]
        for e in existing:
            m = re.match(fr'{conf.inference.output_prefix}_(\d+).*\.pdb$', e)
            if m:
                m = m.groups()[0]
                indices.append(int(m))
        design_startnum = max(indices) + 1   

    conf.inference.design_startnum = design_startnum
    # Initialize sampler and target/contig.
    sampler = model_runners.sampler_selector(conf)
    return sampler

def expand_config(conf):
    confs = {}
    if conf.inference.guidepost_xyz_as_design:
        sub_conf = copy.deepcopy(conf)
        for val in conf.inference.guidepost_xyz_as_design_bb:
            sub_conf.inference.guidepost_xyz_as_design_bb = val
            suffix = f'atomized-bb-{val}'
            confs[suffix] = copy.deepcopy(sub_conf)
    else:
        confs = {'': conf}
    return confs


def sampler_i_des_bounds(sampler):
    '''
    Get i_des_start and i_des_end for a given sampler

    Args:
        sampler (Sampler): sampler

    Returns:
        i_des_start (int): The first i_des to sample
        i_des_end (int): One past the last i_des to sample
    '''
    i_des_start = sampler._conf.inference.design_startnum
    i_des_end = i_des_start + sampler.inf_conf.num_designs
    return i_des_start, i_des_end

def sampler_out_prefix(sampler, i_des=0):
    '''
    Get the output prefix for a sampler run

    Args:
        sampler (Sampler): sampler
        i_des (int): Which design

    Returns:
        run_prefix (str): A prefix that is general for all outputs from this run
        individual_prefix (str): A prefix for this individual i_des
    '''
    run_prefix = sampler.inf_conf.output_prefix
    individual_prefix = f'{run_prefix}_{i_des}'

    return run_prefix, individual_prefix

def load_checkpoint_done(sampler):
    '''
    Load a dict of which designs have already been finished

    Args:
        sampler (Sampler): sampler

    Returns:
        checkpoint_set (set[str,str]): A list of all of the individual prefixes that have already run and a message as to why they are done
    '''
    run_prefix, _ = sampler_out_prefix(sampler)

    if sampler._conf.inference.silent_out:
        # Someday this might switch to using a .silent.idx file
        checkpoint_done = silent_files.load_silent_checkpoint(run_prefix)
    else:
        # Run glob exactly once since it's not good for the file system
        files = glob.glob(run_prefix + '*')

        # Loop through what the names will be and look for the outputs
        checkpoint_done = dict()
        i_des_start, i_des_end = sampler_i_des_bounds(sampler)
        for i_des in range(i_des_start, i_des_end):
            _, individual_prefix = sampler_out_prefix(sampler, i_des=i_des)

            # Check for 4 output patterns that might exist
            for pattern in ['[.]trb', '-.*[.]trb', '[.]pdb', '-.*[.]pdb']:
                re_comp = re.compile(individual_prefix + pattern)

                for file in files:
                    match = re_comp.match(file)
                    if match:
                        message = f'{match.group(0)} already exists.'
                        checkpoint_done[individual_prefix] = message

    return checkpoint_done

def checkpoint_i_des(sampler, i_des):
    '''
    Note that a design has finished

    Args:
        sampler (Sampler): sampler
        i_des (int): Number of design
    '''
    run_prefix, individual_prefix = sampler_out_prefix(sampler, i_des)
    if sampler._conf.inference.silent_out:
        silent_files.silent_checkpoint_design(run_prefix, individual_prefix)

def sample(sampler):
    log = logging.getLogger(__name__)

    # Load a dictionary of finished designs with their finished messages
    checkpoint_done = load_checkpoint_done(sampler)

    # Sample each of the designs
    i_des_start, i_des_end = sampler_i_des_bounds(sampler)
    for i_des in range(i_des_start, i_des_end):
        if sampler._conf.inference.deterministic:
            seed_all(i_des + sampler._conf.inference.seed_offset)

        start_time = time.time()
        _, out_prefix = sampler_out_prefix(sampler, i_des=i_des)
        log.info(f'Making design {out_prefix}')
        if sampler.inf_conf.cautious and out_prefix in checkpoint_done:
             log.info(f'(cautious mode) Skipping this design because {checkpoint_done[out_prefix]}')
             continue
        print(f'making design {i_des} of {i_des_start}:{i_des_end}', flush=True)
        sampler_out = sample_one(sampler, i_des)
        log.info(f'Finished design in {(time.time()-start_time)/60:.2f} minutes')
        original_conf = copy.deepcopy(sampler._conf)
        confs = expand_config(sampler._conf)
        for suffix, conf in confs.items():
            sampler._conf = conf
            out_prefix_suffixed = out_prefix
            if suffix:
                out_prefix_suffixed += f'-{suffix}'
            print(f'{out_prefix_suffixed=}, {conf.inference.guidepost_xyz_as_design_bb=}')
            # TODO: See what is being altered here, so we don't have to copy sampler_out
            save_outputs(sampler, out_prefix_suffixed, *(copy.deepcopy(o) for o in sampler_out))
            sampler._conf = original_conf
        checkpoint_i_des(sampler, i_des)

def sample_one(sampler, i_des=0, simple_logging=False):
    # For intermediate output logging
    indep, contig_map, atomizer, t_step_input = sampler.sample_init(i_des)
    log = logging.getLogger(__name__)

    traj_stack = defaultdict(list)
    denoised_xyz_stack = []
    px0_xyz_stack = []
    seq_stack = []

    rfo = None
    extra = {
        'rfo_uncond': None,
        'rfo_cond': None,
        'n_steps': None
    }

    # Initialize featurizers
    extra_tXd_names = getattr(sampler._conf, 'extra_tXd', [])
    features_cache = features.init_tXd_inference(indep, extra_tXd_names, sampler._conf.extra_tXd_params, sampler._conf.inference.conditions)

    ts = torch.arange(int(t_step_input), sampler.inf_conf.final_step-1, -1)
    n_steps = torch.ones(len(ts), dtype=int)
    partially_diffuse_before = torch.zeros(len(ts), dtype=bool)
    if sampler._conf.inference.custom_t_range:
        ts, n_steps, partially_diffuse_before = iu.get_custom_t_range(sampler._conf)

    # Loop over number of reverse diffusion time steps.
    for it, t in tqdm.tqdm(list(enumerate(ts))):
        sampler._log.info(f'Denoising {t=}')
        if simple_logging:
            e = '.'
            if t%10 == 0:
                e = t
            print(f'{e}', end='')
        if partially_diffuse_before[it]:
            indep.xyz[:,:3] = rfo.xyz[-1, 0, :]
            indep.xyz = indep.xyz.to('cpu')
            indep, _ = aa_model.diffuse(sampler._conf, sampler.diffuser, indep, sampler.is_diffused, int(t))
        if sampler._conf.preprocess.randomize_frames:
            print('randomizing frames')
            indep.xyz = aa_model.randomly_rotate_frames(indep.xyz)
        extra['n_steps'] = n_steps[it]
        px0, x_t, seq_t, rfo, extra = sampler.sample_step(
            t, indep, rfo, extra, features_cache)
        # assert_that(indep.xyz.shape).is_equal_to(x_t.shape)
        rf2aa.tensor_util.assert_same_shape(indep.xyz, x_t)
        # ic(t)
        # # @Altaeth EXPERIMENT: final step, replace x_t with px0
        # if sampler._conf.experiment.get('final_sc_pred', False) and t == sampler.inf_conf.final_step:
        #     x_t = px0[:,:ChemData().NHEAVY,:]
        # ic(x_t.shape)
        indep.xyz = x_t
        x_t = copy.deepcopy(x_t)
        # @Altaeth, this code below might not be necessary, but it blocks important coordinates
        #x_t[:,3:] = np.nan
            
        aa_model.assert_has_coords(indep.xyz, indep)
        # missing_backbone = torch.isnan(indep.xyz).any(dim=-1)[...,:3].any(dim=-1)
        # prot_missing_bb = missing_backbone[~indep.is_sm]
        # sm_missing_ca = torch.isnan(indep.xyz).any(dim=-1)[...,1]
        # try:
        #     assert not prot_missing_bb.any(), f'{t}:prot_missing_bb {prot_missing_bb}'
        #     assert not sm_missing_ca.any(), f'{t}:sm_missing_ca {sm_missing_ca}'
        # except Exception as e:
        #     print(e)
        #     import ipdb
        #     ipdb.set_trace()

        px0_xyz_stack.append(px0)
        denoised_xyz_stack.append(x_t)
        seq_stack.append(seq_t)
        for k, v in extra['traj'].items():
            traj_stack[k].append(v)

    if t_step_input == 0:
        # Null-case: no diffusion performed.
        px0_xyz_stack.append(sampler.indep_orig.xyz)
        denoised_xyz_stack.append(indep.xyz)
        alanine_one_hot = torch.nn.functional.one_hot(torch.tensor(torch.zeros((indep.length(),), dtype=int)), ChemData().NAATOKENS)
        seq_stack.append(alanine_one_hot)
    
    # Flip order for better visualization in pymol
    denoised_xyz_stack = torch.stack(denoised_xyz_stack)
    denoised_xyz_stack = torch.flip(denoised_xyz_stack, [0,])
    px0_xyz_stack = torch.stack(px0_xyz_stack)
    px0_xyz_stack = torch.flip(px0_xyz_stack, [0,])

    for k, v in traj_stack.items():
        traj_stack[k] = torch.flip(torch.stack(v), [0,])

    raw = (px0_xyz_stack, denoised_xyz_stack)

    # Add back any (implicit) side chain atoms from the motif
    denoised_xyz_stack = add_implicit_side_chain_atoms(
        seq=indep.seq,
        act_on_residue=~sampler.is_diffused,
        xyz=denoised_xyz_stack,
        xyz_with_sc=sampler.indep_orig.xyz,
    )
    px0_xyz_stack_filler = add_implicit_side_chain_atoms(
        seq=indep.seq,
        act_on_residue=~sampler.is_diffused,
        xyz=px0_xyz_stack[..., :ChemData().NHEAVY, :],
        # xyz_with_sc=sampler.indep_orig.xyz,
        xyz_with_sc=sampler.indep_orig.xyz[..., :ChemData().NHEAVY, :],
    )
    px0_xyz_stack[..., :ChemData().NHEAVY, :] = px0_xyz_stack_filler

    for k, v in traj_stack.items():
        traj_stack[k] = add_implicit_side_chain_atoms(
            seq=indep.seq,
            act_on_residue=~sampler.is_diffused,
            xyz=v[..., :ChemData().NHEAVY, :],
            xyz_with_sc=sampler.indep_orig.xyz[..., :ChemData().NHEAVY, :],
        )

    # Idealize protein backbone
    is_protein = rf2aa.util.is_protein(indep.seq)
    denoised_xyz_stack[:, is_protein] = idealize_backbone.idealize_bb_atoms(
        xyz=denoised_xyz_stack[:, is_protein],
        idx=indep.idx[is_protein]
    )
    px0_xyz_stack_idealized = torch.clone(px0_xyz_stack)
    px0_xyz_stack_idealized[:, is_protein] = idealize_backbone.idealize_bb_atoms(
        xyz=px0_xyz_stack[:, is_protein],
        idx=indep.idx[is_protein]
    )
    log = logging.getLogger(__name__)
    backbone_ideality_gap = idealize_backbone.backbone_ideality_gap(px0_xyz_stack[0], px0_xyz_stack_idealized[0])
    log.debug(backbone_ideality_gap)
    px0_xyz_stack = px0_xyz_stack_idealized


    for k, v in traj_stack.items():
        traj_stack[k][:, is_protein] = idealize_backbone.idealize_bb_atoms(
            xyz=v[:, is_protein],
            idx=indep.idx[is_protein]
        )

    is_diffused = sampler.is_diffused.clone()

    if atomizer is not None:
        indep_atomized = indep.clone()

        # deatomize `is_diffused`
        is_diffused = atomize.deatomize_mask(atomizer, indep_atomized, is_diffused)

        init_seq_stack = copy.deepcopy(seq_stack)
        indep, px0_xyz_stack, denoised_xyz_stack, seq_stack = \
            deatomize_sampler_outputs(atomizer, indep, px0_xyz_stack, denoised_xyz_stack, seq_stack)
        
        for k, v in traj_stack.items():
            xyz_stack_new = []
            for i in range(len(v)):
                xyz_i = aa_model.pad_dim(v[i], 1, ChemData().NTOTAL, torch.nan)
                indep_atomized.seq = init_seq_stack[i].argmax(-1)
                indep_atomized.xyz = xyz_i
                indep_deatomized = atomizer.deatomize(indep_atomized)
                xyz_stack_new.append(indep_deatomized.xyz)
            traj_stack[k] = torch.stack(xyz_stack_new)

    return indep, contig_map, atomizer, t_step_input, denoised_xyz_stack, px0_xyz_stack, seq_stack, is_diffused, raw, traj_stack, ts

def add_implicit_side_chain_atoms(seq, act_on_residue, xyz, xyz_with_sc):
    '''
    Copies the coordinates of side chain atoms (in residues marked "True" 
    in `act_on_residue`) in `xyz_with_sc` to `xyz`.

    Inputs
    ------------
        seq (L,)
        act_on_residue (L,): Only residues marked True will have side chain atoms added.
        xyz (..., L, n_atoms, 3)
        xyz_with_sc (L, n_atoms, 3)

    '''
    # ic(xyz.shape, xyz_with_sc.shape)
    # Shape checks
    L, n_atoms = xyz_with_sc.shape[:2]
    assert xyz.shape[-3:] == xyz_with_sc.shape, f'{xyz.shape[-3:]=} != {xyz_with_sc.shape=}'
    assert len(seq) == L
    assert len(act_on_residue) == L

    replace_sc_atom = ChemData().allatom_mask[seq][:, :n_atoms]
    is_prot = nucl_utils.get_resi_type_mask(seq, 'prot_and_mask')
    replace_sc_atom[is_prot, :5] = False  # Does not add cb, since that can be calculated from N, CA and C for proteins
    replace_sc_atom[~act_on_residue] = False

    xyz[..., replace_sc_atom, :] = xyz_with_sc[replace_sc_atom]

    return xyz

def deatomize_sampler_outputs(atomizer, indep, px0_xyz_stack, denoised_xyz_stack, seq_stack):
    """Converts atomized residues back to residue-as-residue representation in
    the outputs of a single design trajectory.

    NOTE: `indep` will have `idx`, `bond_features`, `same_chain` updated to
    de-atomized versions, but other features will remain unchanged (and
    therefore become inconsistent).
    """
    px0_xyz_stack_new = []
    denoised_xyz_stack_new = []
    seq_stack_new = []
    for i in range(len(px0_xyz_stack)):
        px0_xyz = aa_model.pad_dim(px0_xyz_stack[i], 1, ChemData().NTOTAL, torch.nan)
        denoised_xyz = aa_model.pad_dim(denoised_xyz_stack[i], 1, ChemData().NTOTAL, torch.nan)
        indep.seq = seq_stack[i].argmax(-1)

        indep.xyz = px0_xyz
        indep_deatomized = atomizer.deatomize(indep)
        px0_xyz_stack_new.append(indep_deatomized.xyz)

        indep.xyz = denoised_xyz
        indep_deatomized = atomizer.deatomize(indep)
        denoised_xyz_stack_new.append(indep_deatomized.xyz)

        seq_ = torch.nn.functional.one_hot(indep_deatomized.seq, ChemData().NAATOKENS)
        alanine_one_hot = torch.nn.functional.one_hot(torch.tensor([0]), ChemData().NAATOKENS)
        cond = ~indep_deatomized.is_sm[...,None] * (seq_ >= ChemData().UNKINDEX)
        seq_ = torch.where(cond, alanine_one_hot, seq_)
        seq_stack_new.append(seq_)
    denoised_xyz_stack_new = torch.stack(denoised_xyz_stack_new)
    px0_xyz_stack_new = torch.stack(px0_xyz_stack_new)

    return indep_deatomized, px0_xyz_stack_new, denoised_xyz_stack_new, seq_stack_new


def match_guideposts_and_generate_mappings(indep, is_diffused, contig_map, denoised_xyz):
    '''
    Determine which diffused residues map to the guidepost residues
        Return those paired lists as well as extra contig keys

    Args:
        indep (indep): Indep with guidepost residues
        is_diffused (torch.Tensor[bool]): Which residues are diffused [L]
        contig_map (ContigMap): The contig map
        denoised_xyz (torch.Tensor[float]): The xyz coordinates to do the matching on

    Returns:
        match_idx (np.array[int]): The idx of the residue on indep that was closest to the guidepost
        gp_idx (np.array[int]): The idx of the guidepost residue
        gp_contig_mappings (dict): Overwrites for some contig_map fields that need to change because of the guideposting
    '''

    # Only diffused residues that aren't small molecules are elgible to be guidepost residues
    could_be_gp_corr = is_diffused & ~indep.is_sm & ~indep.is_gp

    # Make where masks for our subsetted arrays
    could_be_gp_corr_idx = torch.nonzero(could_be_gp_corr)[:,0].numpy()
    idx_by_gp_sequential_idx = torch.nonzero(indep.is_gp)[:,0].numpy()

    # Generate xyz arrays to match
    diffused_xyz = denoised_xyz[could_be_gp_corr]
    gp_alone_xyz = denoised_xyz[indep.is_gp]

    # Do the matching
    gp_alone_to_diffused_idx0 = gp.greedy_guide_post_correspondence(diffused_xyz, gp_alone_xyz)

    # Translate the local-indexing of the matched dictionary to global indexing
    match_idx_by_gp_idx = {}
    for k, v in gp_alone_to_diffused_idx0.items():
        match_idx_by_gp_idx[idx_by_gp_sequential_idx[k]] = could_be_gp_corr_idx[v]

    # If there were any guidepost residues...
    if len(gp_alone_to_diffused_idx0) > 0:

        # Turn the dictionary into lists
        gp_idx, match_idx = zip(*match_idx_by_gp_idx.items())
        gp_idx = np.array(gp_idx)
        match_idx = np.array(match_idx)

        # Generate the contig_map overrides
        gp_contig_mappings = gp.get_infered_mappings(
            contig_map.gp_to_ptn_idx0,
            match_idx_by_gp_idx,
            contig_map.get_mappings()
        )
    else:
        gp_idx = np.array([], dtype=int)
        match_idx = np.array([], dtype=int)
        gp_contig_mappings = {}

    return match_idx, gp_idx, gp_contig_mappings


def save_outputs(sampler, out_prefix, indep, contig_map, atomizer, t_step_input, denoised_xyz_stack, px0_xyz_stack, seq_stack, is_diffused_in, raw, traj_stack, ts):
    log = logging.getLogger(__name__)

    # Make the output folder
    out_head, out_tail = os.path.split(out_prefix)
    os.makedirs(out_head, exist_ok=True)

    final_seq = seq_stack[-1]

    if sampler._conf.seq_diffuser.seqdiff is not None:
        # When doing sequence diffusion the model does not make predictions beyond category 19
        #final_seq = final_seq[:,:20] # [L,20]
        # Cannot do above code for NA, but instead get rid of mask tokens
        final_seq = final_seq[20:22] = 0 

    # All samplers now use a one-hot seq so they all need this step, get rid of non polymer residues
    final_seq[~indep.is_sm, ChemData().NNAPROTAAS:] = 0 
    final_seq = torch.argmax(final_seq, dim=-1)

    # replace mask and unknown tokens in the final seq with alanine
    final_seq = torch.where((final_seq == 20) | (final_seq==21), 0, final_seq)

    # determine lengths of protein and ligand for correct chain labeling in output pdb
    chain_Ls = rf2aa.util.Ls_from_same_chain_2d(indep.same_chain[~indep.is_gp][:,~indep.is_gp])

    # Figure out which timesteps we are going to output
    write_ts = []
    for extra_t in sampler._conf.inference.write_extra_ts:
        assert extra_t in ts, f'inference.write_extra_ts: t:{t} was not part of the ts: {ts}'
        write_ts.append(extra_t)
    write_ts.append(None) # Make sure the default is last so that the variables all end up correct after the loop

    for write_t in write_ts:

        t_suffix = '' if write_t is None else f'_t{write_t}'
        stack_idx = 0 if write_t is None else list(ts).index(write_t)
        # ic(seq_design)

        # Make copies of sampler outputs for final modifications
        xyz_design = px0_xyz_stack[stack_idx].clone()
        seq_design = final_seq.clone()
        gp_contig_mappings = {}
        is_atomized = torch.zeros(indep.length()).bool()
        is_diffused = is_diffused_in.clone()
        if atomizer is not None:
            is_atomized = copy.deepcopy(atomizer.residue_to_atomize)

        # If using guideposts, infer their placement from the final pX0 prediction.
        if sampler._conf.inference.contig_as_guidepost:
            # Use final denoised_xyz for inference unless we aren't at final_step
            match_xyz = denoised_xyz_stack[stack_idx] if write_t is None else px0_xyz_stack[stack_idx]
            match_idx, gp_idx, gp_contig_mappings = match_guideposts_and_generate_mappings(indep, is_diffused, contig_map, match_xyz)

            # Copy guidepost sequence and idx to output if desired
            if sampler._conf.inference.guidepost_xyz_as_design:
                seq_design[match_idx] = seq_design[gp_idx]
                if sampler._conf.inference.guidepost_xyz_as_design_bb:
                    xyz_design[match_idx] = xyz_design[gp_idx]
                else:
                    xyz_design[match_idx, 4:] = xyz_design[gp_idx, 4:]

            # Drop guidepost residues from output and transfer arrays
            xyz_design = xyz_design[~indep.is_gp]
            seq_design = seq_design[~indep.is_gp]
            is_diffused[match_idx] = is_diffused[gp_idx]
            is_diffused = is_diffused[~indep.is_gp]
            is_atomized[match_idx] = is_atomized[gp_idx]
            is_atomized = is_atomized[~indep.is_gp]


        # Save idealized pX0 last step
        xyz_design_idealized = xyz_design.clone()[None]

        idealization_rmsd = float('nan')
        if sampler._conf.inference.idealize_sidechain_outputs:
            log.info('Idealizing atomized sidechains for pX0 of the last step...')
            # Only idealize residues that are atomized.
            xyz_design_idealized[0, is_atomized], idealization_rmsd, _, _ = idealize_pose(
                xyz_design[None, is_atomized],
                seq_design[None, is_atomized]
            )

        # Create pdb, idealize the backbone, and rename ligand atoms
        idealized_pdb_stream = aa_model.write_traj(None, xyz_design_idealized, seq_design, indep.bond_feats, ligand_name_arr=contig_map.ligand_names, chain_Ls=chain_Ls, idx_pdb=indep.idx)
        idealized_pdb_stream = idealize_backbone.rewrite(None, None, pdb_stream=idealized_pdb_stream)
        idealized_pdb_stream = aa_model.rename_ligand_atoms(sampler._conf.inference.input_pdb, None, pdb_stream=idealized_pdb_stream)

        if sampler._conf.inference.silent_out:
            # Tag starts out the same as the pdb name as usual
            tag = out_tail + t_suffix

            # But then we add in the folder path so that people can cat everything together and not have duplicate names
            if bool(sampler._conf.inference.silent_folder_sep):
                tag = os.path.join(out_head, tag).replace('/', sampler._conf.inference.silent_folder_sep)

            run_prefix, _ = sampler_out_prefix(sampler)
            silent_name = run_prefix + '_out.silent'
            silent_files.add_pdb_stream_to_silent(silent_name, tag, idealized_pdb_stream)
            des_path = f'{silent_name}:{tag}'
        else:
            # Write pdb to disk
            out_idealized = f'{out_prefix}{t_suffix}.pdb'
            des_path = os.path.abspath(out_idealized)
            with open(des_path, 'w') as fh:
                fh.write(''.join(idealized_pdb_stream))

    # pX0 last step
    write_unidealized = not sampler._conf.inference.silent_out
    if write_unidealized:
        unidealized_dir = os.path.join(out_head, 'unidealized')
        os.makedirs(unidealized_dir, exist_ok=True)
        out_unidealized = os.path.join(unidealized_dir, f'{out_tail}.pdb')
        aa_model.write_traj(out_unidealized, xyz_design[None,...], seq_design, indep.bond_feats, ligand_name_arr=contig_map.ligand_names, chain_Ls=chain_Ls, idx_pdb=indep.idx)

    # Setup stack_mask for writing smaller trajectories
    t_int = np.arange(int(t_step_input), sampler.inf_conf.final_step-1, -1)[::-1]
    stack_mask = torch.ones(len(denoised_xyz_stack), dtype=bool)
    if len(sampler._conf.inference.write_trajectory_only_t) > 0:
        assert sampler._conf.inference.write_trajectory or sampler._conf.inference.write_trb_trajectory, ('If inference.write_trajectory_only_t is enabled'
                ' at least one of inference.write_trajectory or inference.write_trb_trajectory must be enabled')
        stack_mask[:] = False
        for write_t in sampler._conf.inference.write_trajectory_only_t:
            stack_mask[write_t == t_int] = True
        assert stack_mask.sum() > 0, ('Your inference.write_trajectory_only_t has led to no frames being selected for output. Something is specified wrong.'
            f' inference.write_trajectory_only_t={inference.write_trajectory_only_t}')

    # trajectory pdbs
    if sampler._conf.inference.write_trajectory:
        traj_prefix = os.path.dirname(out_prefix)+'/traj/'+os.path.basename(out_prefix)
        os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

        out = f'{traj_prefix}_Xt-1_traj.pdb'
        aa_model.write_traj(out, denoised_xyz_stack[stack_mask], final_seq, indep.bond_feats, ligand_name_arr=contig_map.ligand_names, chain_Ls=chain_Ls, idx_pdb=indep.idx)
        xt_traj_path = os.path.abspath(out)
        aa_model.rename_ligand_atoms(sampler._conf.inference.input_pdb, xt_traj_path)

        out=f'{traj_prefix}_pX0_traj.pdb'
        aa_model.write_traj(out, px0_xyz_stack[stack_mask], final_seq, indep.bond_feats, chain_Ls=chain_Ls, ligand_name_arr=contig_map.ligand_names, idx_pdb=indep.idx)
        x0_traj_path = os.path.abspath(out)
        aa_model.rename_ligand_atoms(sampler._conf.inference.input_pdb, x0_traj_path)

        for k, v in traj_stack.items():
            out=f'{traj_prefix}_{k}_traj.pdb'
            aa_model.write_traj(out, v[stack_mask], final_seq, indep.bond_feats, chain_Ls=chain_Ls, ligand_name_arr=contig_map.ligand_names, idx_pdb=indep.idx)
            traj_path = os.path.abspath(out)
            aa_model.rename_ligand_atoms(sampler._conf.inference.input_pdb, traj_path)

    # run metadata
    sampler._conf.inference.input_pdb = os.path.abspath(sampler._conf.inference.input_pdb)
    write_trb = (sampler._conf.inference.write_trb and not sampler._conf.inference.silent_out) or str(sampler._conf.inference.write_trb) == 'FORCE'
    if write_trb:
        trb = dict(
            config = OmegaConf.to_container(sampler._conf, resolve=True),
            device = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'CPU',
            t_int=t_int,
            t=np.arange(int(t_step_input), sampler.inf_conf.final_step-1, -1)[::-1] / sampler._conf.diffuser.T,
            is_diffused=sampler.is_diffused,
            point_types=aa_model.get_point_types(sampler.indep_orig, atomizer),
            atomizer_spec=None if atomizer is None else rf_diffusion.atomization_primitives.AtomizerSpec(atomizer.deatomized_state, atomizer.residue_to_atomize),
        )
        # The trajectory and the indep are big and contributed to the /net/scratch crisis of 2024
        if sampler._conf.inference.write_trb_trajectory:
            trb['px0_xyz_stack'] = raw[0].detach().cpu()[stack_mask].numpy()
            trb['denoised_xyz_stack'] = raw[1].detach().cpu()[stack_mask].numpy()
        if sampler._conf.inference.write_trb_indep:
            trb['indep'] = {k:v.detach().cpu().numpy() if hasattr(v, 'detach') else v for k,v in dataclasses.asdict(indep).items()}
            trb['indep_true'] = {k:v.detach().cpu().numpy() if hasattr(v, 'detach') else v for k,v in dataclasses.asdict(sampler.indep_orig).items()}
        if contig_map:
            for key, value in contig_map.get_mappings().items():
                trb[key] = value

        if atomizer:
            motif_deatomized = atomize.convert_atomized_mask(atomizer, ~sampler.is_diffused)
            trb['motif'] = motif_deatomized
        trb['idealization_rmsd'] = idealization_rmsd
        if sampler._conf.inference.contig_as_guidepost:
            # Store the literal location of the guide post residues
            for k in ['con_hal_pdb_idx', 'con_hal_idx0', 'sampled_mask']:
                trb[k+'_literal'] = copy.deepcopy(trb[k])

            # Saved infered guidepost locations. This is probably what downstream applications want.
            trb.update(gp_contig_mappings)        

        with open(f'{out_prefix}.trb','wb') as f_out:
            pickle.dump(trb, f_out)
    else:
        assert not sampler._conf.inference.write_trb_trajectory, 'If you want to write the trajectory to the trb you must enable inference.write_trb'
        assert not sampler._conf.inference.write_trb_indep, 'If you want to write the indep to the trb you must enable inference.write_trb'

    log.info(f'design : {des_path}')
    if sampler._conf.inference.write_trajectory:
        log.info(f'Xt traj: {xt_traj_path}')
        log.info(f'X0 traj: {x0_traj_path}')


if __name__ == '__main__':
    main()
