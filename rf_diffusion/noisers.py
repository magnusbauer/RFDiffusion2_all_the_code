import logging
import torch
import copy

from openfold.utils import rigid_utils as ru

from rf_diffusion.frame_diffusion.data import se3_diffuser
from se3_flow_matching.data import interpolant
from se3_flow_matching.data import so3_utils

logger = logging.getLogger(__name__)

def get(noiser_conf):
    if 'type' not in noiser_conf or noiser_conf.type == 'diffusion':
        return se3_diffuser.SE3Diffuser(noiser_conf)
    elif noiser_conf.type == 'flow_matching':
        return NormalizingFlow(cfg=noiser_conf)
    else:
        raise Exception(f'noiser type: {noiser_conf.type} not recognized')

class NormalizingFlow(interpolant.Interpolant):

    def __init__(self, *, noise_translations=True, **kwargs):
        super().__init__(**kwargs)
        self.noise_translations = noise_translations
        self._device = 'cpu'
        self._r3_diffuser = FakeR3Diffuser

    def forward_multi_t(self, rigids_1, T):
        assert T.ndim == 1
        rigids = []
        for t in T:
            rigids.append(self.forward(rigids_1, t)[None])
        return ru.Rigid.cat(rigids, dim=0)
    

    def forward(self, rigids_1, t):
        '''
        Parameters:
            rigids_1: Rigid [L]
            t: float in [0, 1]
            is_diffused: bool [L]
        Returns:
            rigids_t: Rigid [L]
        '''
        assert t.ndim == 0, t
        t = t[None, None]
        trans_1 = rigids_1.get_trans()[None]
        B, N, _ = trans_1.shape
        res_mask = torch.ones((B, N))
        trans_t = self._corrupt_trans(trans_1, t, res_mask)
        rotmats_1 = rigids_1.get_rots().get_rot_mats()[None]
        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)
        return ru.Rigid(trans=trans_t[0], rots=ru.Rotation(rotmats_t[0]))

    def forward_same_traj(self, rigids_1, t):
        t = t[None, ...]
        self.set_device(t.device)
        trans_1 = rigids_1.get_trans()[None]
        B, N, _ = trans_1.shape
        res_mask = torch.ones((B, N))
        trans_t = self._corrupt_trans_multi_t(trans_1, t, res_mask)
        rotmats_1 = rigids_1.get_rots().get_rot_mats()[None]
        rotmats_t = self._corrupt_rotmats_multi_t(rotmats_1, t, res_mask)
        return ru.Rigid(trans=trans_t[0], rots=ru.Rotation(rotmats_t[0]))
    
    def forward_marginal(
            self,
            rigids_0,
            t,
            diffuse_mask,
            **kwargs,
    ):
        is_diffused = diffuse_mask.bool()
        assert isinstance(rigids_0, ru.Rigid)
        ti = torch.tensor(1 - t, dtype=torch.float32)
        rigids_t = copy.deepcopy(rigids_0)
        rigids_t_diffused = self.forward(rigids_0[is_diffused], ti)
        rigids_t_diffused = ru.Rigid(trans=rigids_t_diffused.get_trans(), rots=rigids_t_diffused.get_rots())
        rigids_t[is_diffused] = rigids_t_diffused
        L, _ = rigids_0.get_trans().shape
        # May need to re-insert batch dimension?
        return {
            'rigids_t': rigids_t,
            # Placeholders to make calc_loss happy
            'rot_score': torch.zeros((L, 3), dtype=torch.float32, device=self._device),
            'trans_score': torch.zeros((L, 3), dtype=torch.float32, device=self._device),
            'rot_score_scaling': 1.0,
            'trans_score_scaling': 1.0,
        }

    def calc_rot_score(self, rots_t, rots_0, t):
        B, L = rots_t.shape
        return torch.zeros(B, 1, L, 3, dtype=rots_t.dtype, device=rots_t.device)
        # return torch.zeros_like(rots_t)

    
    def calc_trans_score(self, trans_t, trans_0, t, use_torch=False, scale=True):
        return torch.zeros_like(trans_0)
    
    def reverse(
            self,
            rigid_t,
            rigid_pred,
            t,
            dt,
            diffuse_mask,
            **kwargs
    ):
        is_diffused = torch.tensor(diffuse_mask[0]).bool() # [L]
        rigid_t_2_diffused = self.reverse_all(
            rigid_t[:, is_diffused],
            rigid_pred[:, is_diffused],
            t,
            dt,
        )
        rigid_t_2 = copy.deepcopy(rigid_t)
        rigid_t_2[:, is_diffused] = rigid_t_2_diffused
        return rigid_t_2
    
    def get_grads(
            self,
            rigid_t,
            rigid_pred,
            t,
            dt,
            **kwargs
    ):
        trans_t_1 = rigid_t.get_trans()
        rotmats_t_1 = rigid_t.get_rots().get_rot_mats()
        pred_trans_1 = rigid_pred.get_trans()
        pred_rotmats_1 = rigid_pred.get_rots().get_rot_mats()

        trans_t_1 = trans_t_1.to(pred_trans_1.device)
        rotmats_t_1 = rotmats_t_1.to(pred_rotmats_1.device)

        # Take reverse step
        trans_grad = (pred_trans_1 - trans_t_1)  #* trans_schedule_scaling

        rots_grad = so3_utils.calc_rot_vf(rotmats_t_1, pred_rotmats_1)

        return trans_grad, rots_grad

    def get_scale(self, t, schedule, rate):
        if schedule == 'linear':
            return 1 / (1-t)
        elif schedule == 'exp':
            return rate
        elif schedule == 'normed_exp':
            c = torch.tensor(rate)
            return c * torch.exp(-c*t) / (torch.exp(-c*t) - torch.exp(-c))
        raise ValueError(
                f'Unknown sample schedule {schedule}')
    
    def get_dt(self, t, dt):
        t = 1 - t
        return (
            self.get_scale(t, self._trans_cfg.sample_schedule, self._trans_cfg.exp_rate) * dt,
            self.get_scale(t, self._rots_cfg.sample_schedule, self._rots_cfg.exp_rate) * dt
        )
    
    def apply_grads(self, rigid_t, trans_grad, rots_grad, trans_dt, rots_dt):

        trans_t_1 = rigid_t.get_trans()
        rotmats_t_1 = rigid_t.get_rots().get_rot_mats()
        trans_t_1 = trans_t_1.to(trans_grad.device)
        trans_t_2 = trans_t_1 + trans_grad * trans_dt
        rots_vf = rots_dt * rots_grad

        rotmats_t_1 = rotmats_t_1.to(rots_vf.device)
        rotmats_t_2 = torch.einsum("...ij,...jk->...ik", rotmats_t_1, so3_utils.rotvec_to_rotmat(rots_vf))

        rigid_t_2 = ru.Rigid(trans=trans_t_2, rots=ru.Rotation(rot_mats=rotmats_t_2))
        return rigid_t_2

    def reverse_all(
            self,
            rigid_t,
            rigid_pred,
            t,
            dt,
            **kwargs
    ):
        #INVERT:
        t_1 = 1 - t

        trans_t_1 = rigid_t.get_trans()
        rotmats_t_1 = rigid_t.get_rots().get_rot_mats()
        pred_trans_1 = rigid_pred.get_trans()
        pred_rotmats_1 = rigid_pred.get_rots().get_rot_mats()

        # Take reverse step
        trans_t_2 = self._trans_euler_step(
            dt, t_1, pred_trans_1, trans_t_1)
        rotmats_t_2 = self._rots_euler_step(
            dt, t_1, pred_rotmats_1, rotmats_t_1)
        
        rigid_t_2 = ru.Rigid(trans=trans_t_2, rots=ru.Rotation(rot_mats=rotmats_t_2))
        return rigid_t_2

class FakeR3Diffuser:
    def marginal_b_t(*args, **kwargs):
        return torch.tensor(1.0)
