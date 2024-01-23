import copy
from functools import wraps
from collections import OrderedDict
import inspect
import torch
from dataclasses import asdict
import numpy as np
from icecream import ic
import itertools
from rf_diffusion import bond_geometry
import sys
from rf_diffusion.aa_model import Indep
import networkx as nx
from rf2aa import chemical
from rf_diffusion import loss
from rf_se3_diffusion.data import r3_diffuser
from abc import abstractmethod, ABC
from rf2aa import util_module
from data import utils as du
from rf_diffusion import idealize
from rf_diffusion import atomize
from rf_diffusion import aa_model

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



def displacement(logit_s, label_s,
                  logit_aa_s, label_aa_s, mask_aa_s, logit_exp,
                  pred, pred_tors, true, mask_crds, mask_BB, mask_2d, same_chain,
                  pred_lddt, idx, dataset, chosen_task, diffusion_mask, t, unclamp=False, negative=False,
                  w_dist=1.0, w_aa=1.0, w_str=1.0, w_all=0.5, w_exp=1.0,
                  w_lddt=1.0, w_blen=1.0, w_bang=1.0, w_lj=0.0, w_hb=0.0,
                  lj_lin=0.75, use_H=False, w_disp=0.0, eps=1e-6, **kwargs):
    d_clamp = None if unclamp else 10.0
    disp = calc_displacement(pred, true)
    dist = torch.norm(disp, dim=-1)

    I, L = dist.shape[0:2]
    if diffusion_mask is None:
        diffusion_mask = torch.full((L,), False)

    o = {}
    for region, mask in (
            ('given', diffusion_mask),
            ('masked', ~diffusion_mask),
            ('total', torch.full_like(diffusion_mask, True))):
        o[f'displacement_{region}'] = torch.mean(dist[:,mask])
        fraction_clamped = 0.0
        if d_clamp is not None:
            # set squared distance clamp to d_clamp**2
            d_clamp=torch.tensor(d_clamp)[None].to(dist.device)
            fraction_clamped = torch.mean((dist>d_clamp).float()).item()
        o[f'displacement_fraction_clamped_{region}'] = fraction_clamped
        for i in range(I):
            o[f'displacement_{region}_i{i}'] = torch.mean(dist[i,mask])
    
    return o
 
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
        pass

    def __call__(
        self,
        indep,
        pred_crds: torch.Tensor,
        atomizer_spec,
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
        pred_crds_padded[:, :3] = pred_crds
        indep.xyz = pred_crds_padded.detach()

        # Make an atomizer
        atomizer = aa_model.AtomizeResidues(**asdict(atomizer_spec))

        # Deatomize
        indep_deatomized = atomizer.deatomize(indep)

        # Idealize only atomized residues
        rmsd = idealize.idealize_pose(
            xyz=indep_deatomized.xyz[None, atomizer_spec.residue_to_atomize].detach(),
            seq=indep_deatomized.seq[None, atomizer_spec.residue_to_atomize].detach(),
            steps=kwargs.pop('steps', 100)
        )[1]
        
        return rmsd[0].detach()

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
    for bond_label, bond_type in zip(
            chemical.btype_labels[1:],
            chemical.num2btype[1:]
    ):
        is_bonded = torch.triu(indep.bond_feats == bond_type)
        i, j = torch.where(is_bonded)
        true_dist = torch.norm(true_crds[i,1]-true_crds[j,1],dim=-1)
        d = {
            'mean': torch.mean(true_dist) if true_dist.numel() else torch.nan,
            'min': torch.min(true_dist) if true_dist.numel() else torch.nan,
            'max': torch.max(true_dist) if true_dist.numel() else torch.nan,
        }
        out[bond_label] = d
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

def get_rigids(atom14):
    return du.rigid_frames_from_atom_14(atom14)

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
            )
            metric_results[name] = metric_output

        return metric_results
