import collections
import random
import string

import torch as th

import ipd
import rf2aa as rf
import rf_diffusion as rfd

SymContig = collections.namedtuple('SymContig', ['contig', 'atoms', 'has_termini'])

class HighTSymmetryManager(rfd.sym.RFDSymmetryManager):
    """docstring for RFDSymManager, the High T edition"""
    kind = 'rf_diffusion_highT'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        assert 'idx' in dir(self)
        self._captured_indep = None

    # def update_px0(self, indep, px0):
    # if self.opt.motif_copy_position_from_px0:
    # mask = th.logical_or(self.idx.kind == 1, self.idx.kind == 12)

    def apply_symmetry(self, xyz, pair, opts, update_symmsub=False, disable_all_fitting=False, **kw):
        '''Apply symmetry to an object or xyz/pair'''
        kw = ipd.Bunch(kw)
        # kw.disable_all_fitting = disable_all_fitting
        kw.disable_all_fitting = True
        # print(kw.disable_all_fitting)
        # xyz = rf.sym.asu_to_best_frame_if_necessary(self, xyz, **kw)
        # xyz = rf.sym.set_particle_radius_if_necessary(self, xyz, **kw)
        # xyz = rf.sym.asu_to_canon_if_necessary(self, xyz, **kw)

        if opts.asu_sym:
            # log.info('Symmetrizing ASU')
            xyz = self.apply_asu_sym(xyz, opts)

        if update_symmsub and not disable_all_fitting:
            xyz, pair, self.symmsub = rf.sym.update_symm_subs_track_module(xyz[None], pair[None], self.symmatrix,
                                                                           self.symmsub, self.allsymmRs, self.metasymm,
                                                                           opts)
            return xyz, pair
        else:
            oshape = xyz.shape
            xyz = xyz.reshape(oshape[0], -1, 3)
            xyz = rf.sym.update_symm_Rs(xyz[None], kw.Lasu, self.symmsub, self.allsymmRs, opts)
            return xyz.reshape(oshape)

    def apply_asu_sym(self, xyz, opts, **kw):
        Lasu = xyz.shape[0] // opts.nsub
        xyz_asu = xyz[:Lasu].cpu()
        COM_asu = xyz_asu.mean(dim=0)
        Lasu_sub = Lasu // opts.high_t_number
        xyz_sub = xyz_asu[:Lasu_sub] - COM_asu
        # COM_sub = xyz_sub.mean(dim=0)
        for n in range(opts.high_t_number):
            R = opts.T_xforms[n][:3, :3].float()  # applying only rotations
            T = opts.T_xforms[n][:, -1][:3] / th.norm(opts.T_xforms[n][:, -1][:3], p=2)
            # T = opts.T_xforms[n][:,-1][:3]
            # t_xyz = xyz_sub - COM_sub
            t_xyz = xyz_sub
            t_xyz = t_xyz.reshape(-1, 3)
            new_coords = t_xyz @ R.T
            # new_coords = new_coords.reshape(xyz_sub.shape) + COM_sub + COM_asu + T
            new_coords = new_coords.reshape(xyz_sub.shape) + COM_asu + T*10
            xyz_asu[n * Lasu_sub:(n+1) * Lasu_sub] = new_coords
        xyz[:Lasu] = xyz_asu.to(device=xyz.device)
        return xyz

    def symmetrize_contigs(self, contigs, contig_atoms, has_termini, _length=None):
        '''
        Symmetrize a contigs string. if a range a-b is specified, choose the random value
        here, to ensure all are same for symmetrical structure
        This is specifically to high T cages where we have an ASU of multiple chains
        '''
        if not self or self.opt.contig_is_symmetric:
            return contigs, contig_atoms, has_termini
        assert len(contigs) == 1
        contigs = contigs[0].split('_')
        chains = [c.split(',') for c in contigs]
        chains = [map(_picksize, c) for c in chains]
        chains = [str.join(',', c) for c in chains]
        contigs = [str.join('_', chains)]
        contigs = contigs * self.nsub * self.opt.high_t_number
        catoms = list(contig_atoms.keys()) if contig_atoms else []
        contig_atoms = contig_atoms.copy() if contig_atoms else None
        for i in range(1, self.nsub * self.opt.high_t_number):
            old, new = 'A', string.ascii_uppercase[i]
            contigs[i] = contigs[i].replace(old, new)
            for res in catoms:
                assert res.startswith('A')
                contig_atoms[res.replace(old, new)] = contig_atoms[res]
        # ic(string.ascii_uppercase)
        # ic(contigs)
        contigs = [str.join('_', contigs)]
        # ic(contig_atoms)
        # assert 0
        has_termini = list(has_termini) * self.nsub * self.opt.high_t_number
        return SymContig(contigs, contig_atoms, has_termini)

    def get_approx_stubs(ca):
        assert ca.ndim == 3 and ca.shape[-1] == 3
        ca0centered = ca[0] - ca[0].mean(0)
        _, _, pc = th.pca_lowrank(ca0centered)
        stub = h.frame(*pc, cen=ca[0].mean(0))
        a_to_others = th.stack([th.eye(4)] + [h.rmsfit(ca[i], ca[0])[2] for i in range(1, len(ca))])
        stubs = h.xform(a_to_others, stub)
        return stubs

    def get_high_t_frames_from_file(fname):
        ca = th.as_tensor(ipd.pdb.readpdb(fname).ca(splitchains=True))
        stubs = get_approx_stubs(ca)
        symframes = ipd.sym.frames('icos', torch=True)
        asymframes = stubs @ h.inv(stubs[0])
        frames = h.xform(symframes, asymframes)
        return frames

def _picksize(segment):
    if '-' in segment and not any(map(str.isalpha, segment)):
        segment = str(random.randint(*[int(i) for i in segment.split('-')]))
    return segment
