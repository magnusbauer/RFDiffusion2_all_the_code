import torch

from rf_diffusion import aa_model

from rf_diffusion import sasa

import torch.nn.functional as F

def get_relative_sasa(indep, conf=None, **kwargs):
    rasa = sasa.get_relative_sasa(indep)
    if torch.rand(1) < 0.5:
        return {'t1d':torch.zeros((indep.length(), conf.n_bins + 1))}
    is_feature_applicable = indep.is_sm
    one_hot = one_hot_buckets(rasa, conf.low, conf.high, conf.n_bins)
    one_hot[~is_feature_applicable] = 0
    return {'t1d':torch.cat((~is_feature_applicable[:, None], one_hot), dim=1)}

def radius_of_gyration_xyz(xyz):
    L, _ = xyz.shape
    com = torch.mean(xyz, dim=0)
    dist = torch.cdist(xyz[None,...], com[None,...])[0]
    return torch.sqrt( torch.sum(torch.square(dist)) / L)

def get_radius_of_gyration(indep, conf=None, is_gp=None, **kwargs):
    if torch.rand(1) < 0.5:
        return {'t1d':torch.zeros((indep.length(), conf.n_bins + 1))}
    assert is_gp is not None
    rog = torch.zeros((indep.length(),))
    is_prot = ~indep.is_sm * ~is_gp
    indep_prot, _ = aa_model.slice_indep(indep, is_prot)
    rog_prot = torch.full((indep_prot.length(),), 0.0)
    for is_chain in indep_prot.chain_masks():
        rog_chain = radius_of_gyration_xyz(indep_prot.xyz[is_chain, 1])
        rog_prot[is_chain] = rog_chain
    rog[is_prot] = rog_prot
    is_feature_applicable = is_prot
    one_hot = one_hot_buckets(rog, conf.low, conf.high, conf.n_bins)
    one_hot[~is_feature_applicable] = 0
    return {'t1d':torch.cat((~is_feature_applicable[:, None], one_hot), dim=1)}


def one_hot_buckets(a, low, high, n, eps=1e-6):
    '''
    First category absorbs anything below low
    Last category absorbs anything above high
    '''
    step = (high-low) / n
    bins = torch.linspace(low+step, high-step, n-1)
    cat = torch.bucketize(a, bins).long()
    return F.one_hot(cat, num_classes=n)


def get_radius_of_gyration_inference(indep, feature_conf, feature_inference_conf, is_gp=None, **kwargs):
    if not feature_inference_conf.active:
        return {'t1d':torch.zeros((indep.length(), feature_conf.n_bins + 1))}
    rog = torch.zeros((indep.length(),))
    is_prot = ~indep.is_sm * ~is_gp
    indep_prot, _ = aa_model.slice_indep(indep, is_prot)
    rog_prot = torch.full((indep_prot.length(),), 0.0)
    for is_chain in indep_prot.chain_masks():
        rog_chain = feature_inference_conf.rog
        rog_prot[is_chain] = rog_chain
    rog[is_prot] = rog_prot
    is_feature_applicable = is_prot
    one_hot = one_hot_buckets(rog, feature_conf.low, feature_conf.high, feature_conf.n_bins)
    one_hot[~is_feature_applicable] = 0
    return {'t1d':torch.cat((~is_feature_applicable[:, None], one_hot), dim=1)}


def get_relative_sasa_inference(indep, feature_conf, feature_inference_conf, **kwargs):
    if not feature_inference_conf.active:
        return {'t1d':torch.zeros((indep.length(), feature_conf.n_bins + 1))}
    rasa = torch.full((indep.length(),), feature_inference_conf.rasa)
    one_hot = one_hot_buckets(rasa, feature_conf.low, feature_conf.high, feature_conf.n_bins)
    is_feature_applicable = indep.is_sm
    one_hot[~is_feature_applicable] = 0
    return {'t1d':torch.cat((~is_feature_applicable[:, None], one_hot), dim=1)}
