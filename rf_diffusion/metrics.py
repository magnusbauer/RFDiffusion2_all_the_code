import copy
from functools import wraps
from collections import OrderedDict
import inspect
import torch
from dataclasses import asdict
import numpy as np
import itertools
from rf_diffusion import bond_geometry
import sys
from rf_diffusion.aa_model import Indep
from rf_diffusion.chemical import ChemicalData as ChemData
from rf_diffusion import loss
from rf_diffusion.frame_diffusion.data import r3_diffuser
from abc import abstractmethod, ABC
from rf_diffusion.frame_diffusion.data import utils as du
from rf_diffusion import idealize
from rf_diffusion import aa_model
from rf_diffusion import guide_posts as gp
import rf2aa.util
from rf_diffusion.dev import idealize_backbone

import logging
logger = logging.getLogger(__name__)

def calc_displacement(pred, true):
    """
    Calculates the displacement between predicted and true CA 

    pred - (I,B,L,3, 3)
    true - (  B,L,27,3)
    """
    B = pred.shape[1]


    assert B == 1
    pred = pred.squeeze(1)
    true = true.squeeze(0)

    pred_ca = pred[:,:,1,...] # (I,L,3)
    true_ca = true[:,1,...]   # (L,3)

    return pred_ca - true_ca[None,...]
 
def contig_description(diffusion_mask):
    is_contig = diffusion_mask
    return [(k.item(),len(list(g))) for k,g in itertools.groupby(is_contig)]

def contig_description_simple(diffusion_mask):
    is_contig_l = contig_description(diffusion_mask)
    return ''.join([str(int(k)) for k, _ in is_contig_l])

def n_contigs(diffusion_mask):
    simple_description = contig_description_simple(diffusion_mask)
    return simple_description.count('1')

def n_contig_res(diffusion_mask):
    is_contig_l = contig_description(diffusion_mask)
    return sum(l for is_contig, l in is_contig_l if is_contig)

def contigs(logit_s, label_s,
              logit_aa_s, label_aa_s, mask_aa_s, logit_exp,
              pred, pred_tors, true, mask_crds, mask_BB, mask_2d, same_chain,
              pred_lddt, idx, dataset, chosen_task, diffusion_mask, t, unclamp=False, negative=False,
              w_dist=1.0, w_aa=1.0, w_str=1.0, w_all=0.5, w_exp=1.0,
              w_lddt=1.0, w_blen=1.0, w_bang=1.0, w_lj=0.0, w_hb=0.0,
              lj_lin=0.75, use_H=False, w_disp=0.0, eps=1e-6, **kwargs):
    if diffusion_mask is None:
        diffusion_mask = torch.full((L,), False)
    return {
        'contig_description_simple': contig_description_simple(diffusion_mask),
        'n_contigs': n_contigs(diffusion_mask),
        'n_contig_res': n_contig_res(diffusion_mask),
    }

def atom_bonds(indep, true_crds, pred_crds, is_diffused, point_types, **kwargs):
    return bond_geometry.calc_atom_bond_loss(indep, true_crds, pred_crds, is_diffused, point_types)

def permute_metric(metric):
    @wraps(metric)
    def permuted_metric(indep, pred_crds, true_crds, input_crds, **kwargs):
        metric_by_input_permutation = {}
        crds_by_name = OrderedDict({
            'pred': pred_crds,
            'true': true_crds,
            'input': input_crds,
        })
        for (a, a_crds), (b, b_crds) in itertools.combinations_with_replacement(crds_by_name.items(), 2):
            if a == b:
                continue
            permutation_label = f'{a}:{b}'
            metric_by_input_permutation[permutation_label] = metric(indep, a_crds, b_crds, **kwargs)

        return metric_by_input_permutation
    return permuted_metric

atom_bonds_permutations = permute_metric(atom_bonds)

def rigid_loss(indep, pred_crds, true_crds, is_diffused, point_types, **kwargs):
    return bond_geometry.calc_rigid_loss(indep, pred_crds, true_crds, is_diffused, point_types)

def rigid_loss_input(indep, input_crds, true_crds, is_diffused, point_types, **kwargs):
    return bond_geometry.calc_rigid_loss(indep, input_crds, true_crds, is_diffused, point_types)

###################################
# Metric class. Similar to Potentials class.
###################################
class Metric(ABC):
    @abstractmethod
    def __init__(self, conf=None):
        pass

    @abstractmethod
    def __call__(
        self,
        indep: Indep, 
        pred_crds: torch.Tensor, 
        true_crds: torch.Tensor, 
        input_crds: torch.Tensor, 
        t: float, 
        is_diffused: torch.Tensor,
        point_types: np.array,
        ):
        pass

class VarianceNormalizedTransMSE():
    '''
    Not intended to be called directly as a metric.
    Does not have the correct call signature.
    '''
    def __init__(self, conf):
        self.r3_diffuser = r3_diffuser.R3Diffuser(conf.diffuser.r3)

    def __call__(
        self,
        other_crds: torch.Tensor, 
        true_crds: torch.Tensor, 
        t: float, 
        is_diffused: torch.Tensor,
        ):

        # Raw mean squared error over diffused atoms
        true_crds = true_crds[..., is_diffused, 1, :] * self.r3_diffuser._r3_conf.coordinate_scaling
        other_crds = other_crds[..., is_diffused, 1, :] * self.r3_diffuser._r3_conf.coordinate_scaling
        mse = loss.mse(other_crds, true_crds)

        # Normalize MSE by the variance of the added noise
        noise_var = 1 - torch.exp(-self.r3_diffuser.marginal_b_t(torch.tensor(t)))
        mse_variance_normalized = mse / noise_var

        return mse_variance_normalized


class VarianceNormalizedPredTransMSE(Metric):
    def __init__(self, conf):
        self.get_variance_normalized_mse = VarianceNormalizedTransMSE(conf)

    def __call__(
        self,
        pred_crds: torch.Tensor, 
        true_crds: torch.Tensor, 
        t: float, 
        is_diffused: torch.Tensor,
        **kwargs
        ):
        return self.get_variance_normalized_mse(pred_crds, true_crds, t, is_diffused)

class VarianceNormalizedInputTransMSE(Metric):
    def __init__(self, conf):
        self.get_variance_normalized_mse = VarianceNormalizedTransMSE(conf)

    def __call__(
        self,
        true_crds: torch.Tensor, 
        input_crds: torch.Tensor, 
        t: float, 
        is_diffused: torch.Tensor,
        **kwargs,
        ):

        return self.get_variance_normalized_mse(input_crds, true_crds, t, is_diffused)

class IdealizedResidueRMSD(Metric):
    '''
    Adjusts torsion angles in the residues to minimize the rmsd
    with the predicted coordinates. Returns the mean rmsd over all
    atoms in atomized residues.

    Note: The torsion angle optimizing is a local search and
    is not guaranteed to retrun the global optimum.
    '''

    def __init__(self, conf):
        self.n_steps = conf.idealization_metric_n_steps

    def __call__(
        self,
        indep,
        pred_crds: torch.Tensor,
        atomizer_spec,
        contig_as_guidepost,
        **kwargs
        ):
        '''
        Inputs
            indep: Indep of the *atomized* protein
            pred_crds (L, n_atoms=3, 3)
            atomizer_spec: Info needed to instantiate an atomizer.

        Currently does not suppor batching
        '''
        device = pred_crds.device

        if atomizer_spec is None:
            return torch.tensor(torch.nan)

        # Shape check
        L, n_atoms = pred_crds.shape[:2]
        assert (3 <= n_atoms) and (n_atoms <= 36), f'{n_atoms=}'

        # Pad pred_crds to 36 atoms
        pred_crds_padded = torch.zeros(L, 36, 3, device=device)
        pred_crds_padded[:, :3] = pred_crds[:, :3]
        indep.xyz = pred_crds_padded.detach()

        # Make an atomizer
        atomizer = aa_model.AtomizeResidues(**asdict(atomizer_spec))

        # Deatomize
        is_protein = rf2aa.util.is_protein(indep.seq)
        indep.xyz[is_protein] = idealize_backbone.idealize_bb_atoms(
            xyz=indep.xyz[None, is_protein],
            idx=indep.idx[is_protein]
        )
        
        indep_deatomized = atomizer.deatomize(indep)

        to_idealize = atomizer_spec.residue_to_atomize.detach().cpu()
        if contig_as_guidepost:
            match_idx_by_gp_idx = gp.match_guideposts(indep_deatomized, atomizer_spec.residue_to_atomize)
            logger.debug(f'{match_idx_by_gp_idx=}')
            match_idx = list(match_idx_by_gp_idx.values())
            indep_deatomized = gp.place_guideposts(indep_deatomized, atomizer_spec.residue_to_atomize, use_guidepost_coordinates_for='sidechain')
            to_idealize = torch.zeros((indep_deatomized.length(),), dtype=bool)
            match_idx = list(match_idx_by_gp_idx.values())
            to_idealize[match_idx] = True

        # Idealize only atomized residues
        _, rmsd, per_residue_rmsd, _ = idealize.idealize_pose(
            xyz=indep_deatomized.xyz[None, to_idealize].detach(),
            seq=indep_deatomized.seq[None, to_idealize].detach(),
            steps=self.n_steps,
        )
        any_residues_to_idealize = to_idealize.any()
        return {
            'rmsd_constellation': rmsd.detach(),
            'rmsd_mean': per_residue_rmsd[0].detach().mean() if any_residues_to_idealize else float('nan'),
            'rmsd_max': per_residue_rmsd[0].detach().max() if any_residues_to_idealize else float('nan'),
            'rmsd_min': per_residue_rmsd[0].detach().min() if any_residues_to_idealize else float('nan'),
        }

def get_guidepost_corresponding_indices(indep, is_gp):
    # gp_to_contig_idx0 = sampler.contig_map.gp_to_ptn_idx0  # map from gp_idx0 to the ptn_idx0 in the contig string.
    # is_gp = torch.zeros_like(indep.seq, dtype=bool)
    # is_gp[list(gp_to_contig_idx0.keys())] = True

    # Infer which diffused residues ended up on top of the guide post residues
    diffused_xyz = indep.xyz[~is_gp * ~indep.is_sm]
    gp_alone_xyz = indep.xyz[is_gp]

    idx_by_gp_sequential_idx = torch.nonzero(is_gp)[:,0].numpy()
    gp_alone_to_diffused_idx0 = gp.greedy_guide_post_correspondence(diffused_xyz, gp_alone_xyz, permissive=True)
    match_idx_by_gp_idx = {}
    for k, v in gp_alone_to_diffused_idx0.items():
        match_idx_by_gp_idx[idx_by_gp_sequential_idx[k]] = v
    gp_idx, match_idx = zip(*match_idx_by_gp_idx.items())
    gp_idx = np.array(gp_idx)
    match_idx = np.array(match_idx)


    return gp_idx, match_idx


def displacement(indep, true_crds, pred_crds, is_diffused, point_types, **kwargs):
    
    true_crds = true_crds[..., is_diffused, 1, :]
    other_crds = pred_crds[..., is_diffused, 1, :]
    mse = loss.mse(other_crds, true_crds)
    return mse


def true_bond_lengths(indep, true_crds, **kwargs):
    '''
    Calculates the min, max, and mean bond lengths for each bond type.
    '''
    out = {}
    for bond_type in ChemData().num2btype[1:]:
        is_bonded = torch.triu(indep.bond_feats == bond_type)
        i, j = torch.where(is_bonded)
        true_dist = torch.norm(true_crds[i,1]-true_crds[j,1],dim=-1)
        d = {
            'mean': torch.mean(true_dist) if true_dist.numel() else torch.nan,
            'min': torch.min(true_dist) if true_dist.numel() else torch.nan,
            'max': torch.max(true_dist) if true_dist.numel() else torch.nan,
        }
        out[bond_type] = d
    return out

displacement_permutations = permute_metric(displacement)

def rotations_input(indep, input_crds, true_crds, is_diffused, **kwargs):
    return rotations(indep, input_crds, true_crds, is_diffused)

def rotations(indep, pred_crds, true_crds, is_diffused, **kwargs):
    '''
    Calculates the min, max, and mean angles between predicted/true frames.
    '''

    pred_crds = pred_crds[~indep.is_sm * is_diffused]
    true_crds = true_crds[~indep.is_sm * is_diffused]

    rigid_pred = get_rigids(pred_crds)
    rigid_true = get_rigids(true_crds)

    rot_pred = rigid_pred.get_rots()
    rot_true = rigid_true.get_rots()

    omega = rot_true.angle_between_rotations(rot_pred) # [I, L]

    o = {}
    omega_i = omega
    o['omega'] = {
        'mean': torch.mean(omega_i) if omega_i.numel() else torch.nan,
        'max': torch.max(omega_i) if omega_i.numel() else torch.nan,
        'min': torch.min(omega_i) if omega_i.numel() else torch.nan,
    }
    return o

def guidepost_positioning(indep, true_crds, pred_crds, atomizer_spec, pred_logits, true_aa, input_aa, contig_as_guidepost, **kwargs):
    if pred_logits is None or true_aa is None or input_aa is None:
        raise Exception('Cannot be called with legacy metrics info pickles')
    indep_pred = copy.deepcopy(indep)
    indep_pred.xyz = pred_crds
    indep.xyz = true_crds

    is_gp = torch.zeros_like(indep.is_sm).bool()
    if atomizer_spec is not None:
        atomizer = aa_model.AtomizeResidues(**asdict(atomizer_spec))
        indep_pred = atomizer.deatomize(indep_pred)
        indep = atomizer.deatomize(indep)
        is_gp = torch.zeros_like(indep.is_sm).bool()
        if contig_as_guidepost:
            is_gp[atomizer_spec.residue_to_atomize] = True
    
    if not is_gp.any():
        return {}

    true_gp_idx, true_match_idx = get_guidepost_corresponding_indices(indep, is_gp)
    pred_gp_idx, pred_match_idx = get_guidepost_corresponding_indices(indep_pred, is_gp)
    
    pred_gp_idx, pred_match_idx = map(np.array, zip(*sorted(zip(pred_gp_idx, pred_match_idx), key=lambda x: x[1])))
    true_gp_idx, true_match_idx = map(np.array, zip(*sorted(zip(true_gp_idx, true_match_idx), key=lambda x: x[1])))

    pred_aa = pred_logits.argmax(dim=-1)

    pred_match_seq = pred_aa[pred_match_idx].numpy()
    true_match_seq = true_aa[true_match_idx].numpy()
    input_match_seq = input_aa[pred_match_idx].numpy()

    should_change = input_match_seq != true_match_seq
    did_change = input_match_seq != pred_match_seq
    correct_seq = pred_match_seq == true_match_seq
    input_seq_correct = input_match_seq == true_match_seq

    did_change_whole_seq = (input_aa != pred_aa).numpy()
    input_masked = input_aa.numpy().astype(int) == ChemData().MASKINDEX

    return dict(
        fraction_placed_correctly = (pred_match_idx == true_match_idx).mean(),
        fraction_pred_sequence_agreement = correct_seq.mean(),
        fraction_input_sequence_agreement = input_seq_correct.mean(),
        fraction_change_in_correctness = (correct_seq.astype(float) - input_seq_correct.astype(float)).mean(),

        fraction_should_change_changed_incorrectly = (~correct_seq)[did_change * should_change].mean(),
        fraction_should_change_did_not_change = (~did_change)[should_change].mean(),
        fraction_shouldnt_change_changed_incorrectly = (~correct_seq)[~should_change].mean(),

        fraction_masked_changed = did_change_whole_seq[input_masked].mean(),
        fraction_unmasked_changed = did_change_whole_seq[~input_masked].mean(),

        fraction_input_correct = (input_aa == true_aa).numpy().mean(),
        fraction_pred_correct = (pred_aa == true_aa).numpy().mean(),
    )


def get_rigids(atom14):
    return du.rigid_frames_from_atom_14(atom14)

def n_atomized(atomizer_spec, **kwargs):
    if atomizer_spec is None:
        return 0
    return {'n': atomizer_spec.residue_to_atomize.sum().item()}

###################################
# Metric manager
###################################
class MetricManager:
    def __init__(self, conf):
        '''
        conf: Configuration object for training. Must have...
            metrics: Name of class (or function) in this module to be used as a metric.
        '''
        self.conf = conf

        # Initialize all metrics to be used
        thismodule = sys.modules[__name__]
        self.metric_callables = {}
        for name in conf.metrics:
            obj = getattr(thismodule, name)
            # Currently support metrics being Metric subclass or a ducktyped function with identical call signature.
            # Might change to only supporting Metric subclasses in the future.
            if inspect.isclass(obj) and issubclass(obj, Metric):
                self.metric_callables[name] = obj(conf)
            elif callable(obj):
                self.metric_callables[name] = obj
            else:
                raise TypeError(f'Tried to use {name} as a metric, but it is neither a Metric subclass nor callable.')

    def compute_all_metrics(
        self, 
        indep: Indep, 
        pred_crds: torch.Tensor, 
        true_crds: torch.Tensor, 
        input_crds: torch.Tensor, 
        t: float, 
        is_diffused: torch.Tensor,
        point_types: np.ndarray,
        pred_crds_stack: torch.Tensor = None,
        atomizer_spec: aa_model.AtomizerSpec = None,
        contig_as_guidepost: bool = False,
        pred_logits: torch.Tensor = None,
        true_aa: torch.Tensor = None,
        input_aa: torch.Tensor = None,
        **kwargs,
        ) -> dict:
        '''
        Inputs
            indep: Defines protein and/or molecule connections. 
            pred_crds (..., L, n_atoms, 3)
            true_crds (..., L, n_atoms, 3)
            input_crds (..., L, n_atoms, 3)
            t: Time in the diffusion process. Between 0 and 1.
            is_diffused (L,): True if the residue was diffused.
            point_types (L,): 'L': Ligand, 'R': Residue, 'AB': Atomized backbone, 'AS': Atomized sidechain
            atomizer_spec: Info needed to instantiate an atomizer.
        Returns
            Dictionary of the name of the metric and what it returned.
        '''
        # Basic class and shape checks
        L = true_crds.shape[-3]
        assert (0 <= t) and (t <= 1), f't must be between 0 and 1, but was {t:.3f}'
        assert pred_crds.shape[-3:] == true_crds.shape[-3:]
        assert input_crds.shape[-3:] == true_crds.shape[-3:]
        assert is_diffused.ndim == 1
        assert is_diffused.shape[0] == L
        assert point_types.shape[0] == L

        # Evaluate each metric
        metric_results = {}
        for name, callable in self.metric_callables.items():
            metric_output = callable(
                indep=indep,
                pred_crds=pred_crds.cpu().detach(),
                true_crds=true_crds.cpu().detach(),
                input_crds=input_crds.cpu().detach(),
                t=t,
                is_diffused=is_diffused,
                point_types=point_types,
                pred_crds_stack=pred_crds_stack,
                atomizer_spec=atomizer_spec,
                contig_as_guidepost=contig_as_guidepost,
                pred_logits=pred_logits,
                true_aa=true_aa,
                input_aa=input_aa,
            )
            metric_results[name] = metric_output

        return metric_results
