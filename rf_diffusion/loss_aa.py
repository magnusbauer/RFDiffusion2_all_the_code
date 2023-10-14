import torch
from icecream import ic
from rf_diffusion.aa_model import Indep
import loss

def frame_distance_loss(R_pred, R_true, is_sm):
    return loss.frame_distance_loss(R_pred[:,:,~is_sm], R_true[~is_sm])
