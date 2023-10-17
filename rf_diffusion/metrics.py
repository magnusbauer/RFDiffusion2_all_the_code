import inspect
import torch
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
from abc import abstractmethod

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

def atom_bonds(indep, pred_crds, is_diffused, point_types, **kwargs):
    return bond_geometry.calc_atom_bond_loss(indep, pred_crds, is_diffused, point_types)

###################################
# Bond geometry metrics
###################################
def make_chemical_graph(indep, crds):
    G = nx.Graph()

    # Add atoms as nodes
    for i, (element_int, xyz) in enumerate(zip(indep.seq[indep.is_sm], crds[indep.is_sm, 1])):
        element = chemical.num2aa[element_int]
        G.add_node(i, element=element, xyz=xyz)

    # Add edges if covalently bonded
    bond_feats = indep.bond_feats[indep.is_sm][indep.is_sm]
    is_covalently_bonded = (1 <= bond_feats) & (bond_feats <= 4)
    src, dst = torch.where(is_covalently_bonded)
    G.add_edges_from(zip(src.tolist(), dst.tolist()))

    # Add degree of each atom
    for n in list(G):
        G.nodes[n]['degree'] = nx.degree(G, n)

    return G

def get_bond_length_MAE(indep, other_crds, true_crds):
    # Make chemical graph
    G_other = make_chemical_graph(indep, other_crds)
    G_true = make_chemical_graph(indep, true_crds)

    # Get all bond distances
    geo_other = bond_geometry.gather_aa_geometries(G_other, bond_geometry.get_bond_dists)
    geo_true = bond_geometry.gather_aa_geometries(G_true, bond_geometry.get_bond_dists)
    dist_other, dist_true = bond_geometry.collate_and_flatten_bond_geo(geo_other, geo_true)
    dist_other = torch.tensor(dist_other)
    dist_true = torch.tensor(dist_true)

    # Calc MAE
    mae = (dist_other - dist_true).abs().mean()

    return mae

def get_pred_bond_length_MAE(indep, pred_crds, true_crds, **kwargs):
    return get_bond_length_MAE(indep, pred_crds, true_crds)

def get_input_bond_length_MAE(indep, input_crds, true_crds, **kwargs):
    return get_bond_length_MAE(indep, input_crds, true_crds)


###################################
# Metric class. Similar to Potentials class.
###################################
class Metric:
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
        is_diffused: torch.Tensor
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
        **kwargs
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
        indep: Indep, 
        pred_crds: torch.Tensor, 
        true_crds: torch.Tensor, 
        input_crds: torch.Tensor, 
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
        indep: Indep, 
        pred_crds: torch.Tensor, 
        true_crds: torch.Tensor, 
        input_crds: torch.Tensor, 
        t: float, 
        is_diffused: torch.Tensor,
        **kwargs,
        ):

        return self.get_variance_normalized_mse(input_crds, true_crds, t, is_diffused)


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
        point_types: np.array
        ) -> dict:
        '''
        Inputs
            indep: Defines protein and/or molecule connections. 
            pred_crds (..., L, n_atoms, 3)
            true_crds (..., L, n_atoms, 3)
            input_crds (..., L, n_atoms, 3)
            t: Time in the diffusion process. Between 0 and 1.
            is_diffused (L,): True if the residue was diffused.

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

        # Evaluate each metric
        metric_results = {}
        for name, callable in self.metric_callables.items():
            metric_output = callable(
                indep=indep,
                pred_crds=pred_crds,
                true_crds=true_crds,
                input_crds=input_crds,
                t=t,
                is_diffused=is_diffused,
                point_types=point_types
            )
            metric_results[name] = metric_output

        return metric_results