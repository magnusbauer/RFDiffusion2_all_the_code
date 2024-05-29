import math
import torch
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

def one_hot_bucket(x: torch.Tensor, boundaries: torch.Tensor):
    '''
    Return a one-hot encoding of the bucket x falls into.
    x must be in the interval (boundaries_low, boundaries_high).
    '''
    n_cat = len(boundaries) - 1
    cat_int = torch.bucketize(x, boundaries) - 1
    return torch.nn.functional.one_hot(cat_int, n_cat)

def get_boundary_values(style: str, T:int):
    '''
    Inputs
        style: Different ways of constructing the boundary values.
        T: Controls how finely the [0, 1] interval is binned.
    Returns
        Boundaries for little t embeddings. Spans [0, 1]
    '''
    if style == 'linear':
        return torch.linspace(0, 1, T + 1),
    elif style == 'low_t_heavy':
        return torch.cat([
            torch.arange(0,    0.05, 1 / (T * 16)),
            torch.arange(0.05, 0.10, 1 / (T * 8)),
            torch.arange(0.10, 0.20, 1 / (T * 4)),
            torch.arange(0.20, 0.40, 1 / (T * 2)),
            torch.arange(0.40, 1.00, 1 / (T * 1)),
            torch.tensor([1.]),
        ])

def get_little_t_embedding_inference(indep, feature_conf, feature_inference_conf, t_cont, **kwargs):
    return get_little_t_embedding(indep, feature_conf, t_cont, **kwargs)

def get_little_t_embedding(indep, feature_conf, t_cont: float=None, **kwargs):
    '''
    Args
        t_cont [0, 1]: "continuous" time little_t

        feature_conf:
            style: Different ways of constructing the time boundary values.
            T: Controls how finely the [0, 1] interval is binned. Higher is finer.

    Returns
        One-hot encoding of the selected time bin.
    '''
    boundary_values = get_boundary_values(feature_conf.boundary_style, feature_conf.T)
    oh = one_hot_bucket(t_cont, boundary_values)[None]
    return oh.tile(indep.length(), 1)

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

def get_sinusoidal_timestep_embedding_inference(indep, feature_conf, feature_inference_conf, t_cont, **kwargs):
    return get_sinusoidal_timestep_embedding_training(indep, feature_conf, t_cont)

def get_sinusoidal_timestep_embedding_training(indep, feature_conf, t_cont: float=None, **kwargs):
    emb = get_sinusoidal_timestep_embedding(torch.tensor([t_cont]), feature_conf.embedding_dim, feature_conf.max_positions)
    return emb.tile((indep.length(),1))

def get_sinusoidal_timestep_embedding(timesteps, embedding_dim, max_positions):
    # Adapted from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert (embedding_dim % 2 == 0)
    assert ((0 <= timesteps) * (1 >= timesteps)).all()
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

featurizers = {
    'radius_of_gyration': get_radius_of_gyration,
    'relative_sasa': get_relative_sasa,
    'radius_of_gyration_v2': v2.get_radius_of_gyration,
    'relative_sasa_v2': v2.get_relative_sasa,
    'little_t_embedding': get_little_t_embedding,
    'sinusoidal_timestep_embedding': get_sinusoidal_timestep_embedding_training,
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
    'little_t_embedding': get_little_t_embedding_inference,
    'sinusoidal_timestep_embedding': get_sinusoidal_timestep_embedding_inference,
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
