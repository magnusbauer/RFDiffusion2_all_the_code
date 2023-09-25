import torch
from icecream import ic
import rf_diffusion.conditions.v2 as v2

from rf_diffusion import aa_model

from rf_diffusion import sasa

def get_extra_t1d(indep, featurizer_names, **kwargs):
    if not featurizer_names:
        return torch.zeros((indep.length(),0))
    t1d = []
    for name in featurizer_names:
        feats_1d = featurizers[name](indep, kwargs[name], **kwargs)
        t1d.append(feats_1d)
    return torch.cat(t1d, dim=-1)

def get_radius_of_gyration(indep, is_gp=None, radius_of_gyration=None, **kwargs):
    assert is_gp is not None
    rog = torch.zeros((indep.length(),))
    rog_std = torch.zeros((indep.length(),))
    is_prot = ~indep.is_sm * ~is_gp
    indep_prot, _ = aa_model.slice_indep(indep, is_prot)
    rog_prot = torch.full((indep_prot.length(),), -1.0)
    rog_std_prot = torch.full((indep_prot.length(),), -1.0)
    for is_chain in indep_prot.chain_masks():
        std = torch.abs(torch.normal(0.0, radius_of_gyration.std_std, (1,)))
        rog_chain = radius_of_gyration_xyz(indep_prot.xyz[is_chain, 1])
        rog_chain = torch.normal(rog_chain, std)
        rog_prot[is_chain] = rog_chain
        rog_std_prot[is_chain] = std
    rog[is_prot] = rog_prot
    rog_std[is_prot] = rog_std_prot
    return (rog, rog_std)

def radius_of_gyration_xyz(xyz):
    L, _ = xyz.shape

    com = torch.mean(xyz, dim=0)
    dist = torch.cdist(xyz[None,...], com[None,...])[0]
    return torch.sqrt( torch.sum(torch.square(dist)) / L)

def get_relative_sasa(indep, relative_sasa=None, **kwargs):
    return sasa.noised_relative_sasa(indep, relative_sasa.std_std)

featurizers = {
    'radius_of_gyration': get_radius_of_gyration,
    'relative_sasa': get_relative_sasa,
    'radius_of_gyration_v2': v2.get_radius_of_gyration,
    'relative_sasa_v2': v2.get_relative_sasa,
}

def get_radius_of_gyration_inference(indep, feature_conf, is_gp=None):

    assert is_gp is not None

    rog = torch.zeros((indep.length(),))
    rog_std = torch.zeros((indep.length(),))

    is_prot = ~indep.is_sm * ~is_gp
    indep_prot, _ = aa_model.slice_indep(indep, is_prot)
    rog_prot = torch.full((indep_prot.length(),), -1.0)
    rog_std_prot = torch.full((indep_prot.length(),), -1.0)
    for is_chain in indep_prot.chain_masks():
        rog_prot[is_chain] = feature_conf.mean
        rog_std_prot[is_chain] = feature_conf.std
    
    rog[is_prot] = rog_prot
    rog_std[is_prot] = rog_std_prot
    return (rog, rog_std)

def get_relative_sasa_inference(indep, feature_conf, **kwargs):
    sasa = torch.full((indep.length(),), -10.0)
    sasa[indep.is_sm] = feature_conf.mean
    std = torch.full((indep.length(),), feature_conf.std)
    return (sasa, std)

inference_featurizers = {
    'radius_of_gyration': get_radius_of_gyration_inference,
    'relative_sasa': get_relative_sasa_inference,
    'radius_of_gyration_v2': v2.get_radius_of_gyration_inference,
    'relative_sasa_v2': v2.get_relative_sasa_inference,
}

def get_extra_t1d_inference(indep, featurizer_names, params_train, params_inference, **kwargs):
    if not featurizer_names:
        return torch.zeros((indep.length(),0))
    t1d = []
    for name in featurizer_names:
        assert name in params_train
        assert name in params_inference
        feats_1d = inference_featurizers[name](indep, params_train[name], params_inference[name], **kwargs)
        t1d.append(feats_1d)
    return torch.cat(t1d, dim=-1)