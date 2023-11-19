import torch
from icecream import ic

# from openfold.utils import rigid_utils as ru
from se3_flow_matching.openfold.utils import rigid_utils as ru

from rf_se3_diffusion.data import se3_diffuser
from se3_flow_matching.data import interpolant

def get(noiser_conf):
    if noiser_conf.type == 'diffusion':
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
        assert t.ndim == 0, t
        t = t[None, None]
        # self.set_device(t.device)
        # self.set_device(t.device)
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
            **kwargs,
    ):
        ti = torch.tensor(1 - t, dtype=torch.float32)
        rigids_t = self.forward(rigids_0, ti)
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