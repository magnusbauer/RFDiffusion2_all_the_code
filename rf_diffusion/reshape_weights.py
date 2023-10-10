import os
import sys

from icecream import ic
import torch
import hydra
from omegaconf import DictConfig
from rf_score.model import RFScore

def get_t1d_updates(model_d, weight_d, n):
    d_t1d_old = weight_d['model.templ_emb.templ_stack.proj_t1d.weight'].shape[1]
    updates = {}
    
    for p, new_idx in [
            ('model.templ_emb.emb.weight', torch.cat((torch.arange(-n, 0), torch.arange(-n-d_t1d_old-n, -d_t1d_old-n)))),
            ('model.templ_emb.templ_stack.proj_t1d.weight', torch.arange(d_t1d_old, d_t1d_old+n)),
            ('model.templ_emb.emb_t1d.weight', torch.arange(d_t1d_old, d_t1d_old+n))]:
        new_weight = model_d[p].clone()
        i = torch.ones(model_d[p].shape[1]).bool()
        i[new_idx] = False
        new_weight[:, i] = weight_d[p]
        updates[p] = new_weight.clone()
    
    return updates

def changed_dimensions(model_d, weight_d):
    changed = {}
    for param in model_d:
        if param not in weight_d:
            raise Exception(f'missing {param}')
        if (weight_d[param].shape != model_d[param].shape):
            changed[param] = (model_d[param], weight_d[param])
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
    checkpoint = torch.load(conf.rf.ckpt_path, map_location=map_location)

    # Handle loading from str pred weights
    model_name = getattr(checkpoint.get('model'), '__name__', '')
    is_str_pred = model_name != 'RFScore'
    if is_str_pred:
        checkpoint['model_name'] = 'RFScore'
        for wk in ['final_state_dict', 'model_state_dict']:
            checkpoint[wk] = {f'model.{k}':v for k,v in checkpoint[wk].items()}
    
    model_d = model.state_dict()
    weight_d = checkpoint['final_state_dict']


    ic(weight_d['model.templ_emb.emb.weight'].shape)
    changed = changed_dimensions(model_d, weight_d)

    for param, (model_tensor, weight_tensor) in changed.items():
        print (f'wrong size: {param}\n\tmodel   :{model_tensor.shape}\n\tweights: {weight_tensor.shape}')
    
    
    d_t1d_old = weight_d['model.templ_emb.templ_stack.proj_t1d.weight'].shape[1]
    d_t1d_new = model_d['model.templ_emb.templ_stack.proj_t1d.weight'].shape[1]
    new_t1d_dim = d_t1d_new - d_t1d_old

    for k in ['final_state_dict', 'model_state_dict']:
        weight_d = checkpoint[k]
        updates = get_t1d_updates(model_d, weight_d, new_t1d_dim)
        weight_d.update(updates)
    
    assert not os.path.exists(conf.reshape.output_path)
    torch.save(checkpoint, conf.reshape.output_path)

if __name__ == "__main__":
    run()