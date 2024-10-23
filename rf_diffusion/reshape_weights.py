#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'
import os

from icecream import ic
import torch
import hydra
from omegaconf import DictConfig
from rf_diffusion.frame_diffusion.rf_score.model import RFScore

def get_templ_emb_updates(model_d, weight_d, n, m, weight_prefix):
    """
    Inputs:
        model_d:  newly initialized weight with target shape
        weight_d: old weights to extract relevant features from
        n:        how many new t1d feature dimensions we are adding
        m:        how many new t2d feature dimensions we are adding

    returns:
        updates: weight dict with new target shaped weights, with input weights inserted.
    """
    d_t1d_old = weight_d[weight_prefix+'model.templ_emb.templ_stack.proj_t1d.weight'].shape[1]
    updates = {}

    for p, new_idx in [
            ('model.templ_emb.templ_stack.proj_t1d.weight', torch.arange(d_t1d_old, d_t1d_old+n)),
            ('model.templ_emb.emb_t1d.weight', torch.arange(d_t1d_old, d_t1d_old+n)),
            ('model.templ_emb.emb.weight', torch.cat((
                                                torch.arange(-n-d_t1d_old-n-d_t1d_old-m,-n-d_t1d_old-n-d_t1d_old), # new t2d indices
                                                torch.arange(-n-d_t1d_old-n,-d_t1d_old-n), # first new t2d indices
                                                torch.arange(-n, 0)) # second new t2d indices
                                                )),
            ]:
        new_weight = model_d[p].clone() # Get tensor with expanded dims from model. Should have full final target shape
        i = torch.ones(model_d[p].shape[1]).bool() # get full set of bools of shape of new model dims
        i[new_idx] = False # Set the new dims to false, so that "i" only indexes the old features
        new_weight[:, i] = weight_d[weight_prefix+p] # Set the new_weight to the old_weight features at the old feature indices, "i".
        updates[p] = new_weight.clone()
    
    return updates

def changed_dimensions(model_d, weight_d, weight_prefix):
    changed = {}
    for param in model_d:
        if weight_prefix+param not in weight_d:
            raise Exception(f'missing {param}')
        if (weight_d[weight_prefix+param].shape != model_d[param].shape):
            changed[param] = (model_d[param], weight_d[weight_prefix+param])
    return changed

class FakeRFScore():
    pass
FakeRFScore.name = 'RFScore'

@hydra.main(version_base=None, config_path="config/training", config_name="base")
def run(conf: DictConfig) -> None:
    diffuser = None
    device = 'cpu'
    model = RFScore(conf.rf.model, diffuser, device).to(device)

    map_location = {"cuda:%d"%0: "cpu"}
    checkpoint = torch.load(conf.ckpt_load_path, map_location=map_location, weights_only=False)

    # Add extra 'model.' prefix if user specifies reshape.legacy_input_weights as True:
    weight_prefix = 'model.' * conf.reshape.legacy_input_weights

    # Handle loading from str pred weights
    model_name = getattr(checkpoint.get('model'), '__name__', '')
    is_str_pred = model_name != 'RFScore'
    if is_str_pred:
        checkpoint['model_name'] = 'RFScore'
        for wk in ['final_state_dict', 'model_state_dict']:
            checkpoint[wk] = {f'{weight_prefix}{k}':v for k,v in checkpoint[wk].items()}
    
    model_d = model.state_dict()
    weight_d = checkpoint['final_state_dict']

    ic(weight_d[weight_prefix+'model.templ_emb.emb.weight'].shape)
    changed = changed_dimensions(model_d, weight_d, weight_prefix)

    for param, (model_tensor, weight_tensor) in changed.items():
        print (f'wrong size: {param}\n\tmodel   :{model_tensor.shape}\n\tweights: {weight_tensor.shape}')
    
    d_emb_new   = model_d['model.templ_emb.emb.weight'].shape[1]
    d_emb_old   = weight_d[weight_prefix+'model.templ_emb.emb.weight'].shape[1]
    
    d_t1d_new   = model_d['model.templ_emb.templ_stack.proj_t1d.weight'].shape[1]
    d_t1d_old   = weight_d[weight_prefix+'model.templ_emb.templ_stack.proj_t1d.weight'].shape[1]

    new_emb_dim = d_emb_new - d_emb_old
    new_t1d_dim = d_t1d_new - d_t1d_old
    new_t2d_dim = new_emb_dim - 2*new_t1d_dim

    print(f'Increasing d_t1d by {new_t1d_dim}')
    print(f'Increasing d_t2d by {new_t2d_dim}')

    for k in ['final_state_dict', 'model_state_dict']:
        weight_d = checkpoint[k]
        updates = get_templ_emb_updates(model_d, weight_d, new_t1d_dim, new_t2d_dim, weight_prefix)
        weight_d.update(updates)
    
    assert not os.path.exists(conf.reshape.output_path)
    torch.save(checkpoint, conf.reshape.output_path)

if __name__ == "__main__":
    run()
