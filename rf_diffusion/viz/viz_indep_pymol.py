import numpy as np
import torch as th

import ipd
import rf_diffusion as rfd
from ipd import h

pymol = ipd.lazyimport('pymol')

def get_atom_colors(indep, idx):
    coldict = dict(N=(0, 0, 1), O=(1, 0, 0), C=(0, 1, 0))
    seq = np.array(indep.human_readable_seq())[idx]
    return [coldict[s] if s in coldict else (1, 1, 1) for s in seq]

_count = 0

def show_indep_fancy(indep, name='Indep', sym=None, split=True, show='bb gp atom bond lig chirals', **kw):
    '''
   Show an Indep object in pymol
   '''
    # ic(split, show, delprev, sym, kw)
    # import ipd
    # wu.save((sym, indep), 'tmp/qcp_scan_test_indep.pickle')
    # assert 0
    # ic(name, split, show)

    show = show.split() if isinstance(show, str) else show
    cgo = None if split else []
    # nobj = len(pymol.cmd.get_object_list())
    # ic(pymol.cmd.get_object_list())

    nsub = sym.nsub if sym else 1
    xyz = h.point(th.where(th.isnan(indep.xyz), 0, indep.xyz))
    isprot = (indep.type() == 0)
    if sym and sym.idx: isprot &= (~sym.idx.isfakeprot)
    is_gp = th.zeros(len(xyz), dtype=bool)

    if hasattr(indep, 'is_gp'):
        is_gp = indep.is_gp
        gp = xyz[is_gp]
        isprot = isprot & ~is_gp
        gpcol = get_atom_colors(indep, is_gp)
        if gp.shape[0] and 'gp' in show:
            ic(gp.shape, show, split, cgo)
            ipd.viz.show_ndarray_point_or_vec(gp[:, 1],
                                              name=f'{name}{"_gp" if split else ""}',
                                              sphere=0.2,
                                              col=gpcol,
                                              addtocgo=cgo,
                                              **kw)

    ligxyz = h.point(xyz[(indep.type() == 1) & ~is_gp, 1])
    ligcol = get_atom_colors(indep, indep.type() == 1 * ~is_gp)
    if ligxyz.shape[1] and 'lig' in show:
        ipd.viz.show_ndarray_point_or_vec(ligxyz,
                                          name=f'{name}{"_lig" if split else ""}',
                                          sphere=0.3,
                                          col=ligcol,
                                          addtocgo=cgo,
                                          **kw)
        ipd.viz.show_ndarray_point_or_vec(ligxyz, name=f'{name}{"_lig" if split else ""}', sphere=0.3, col=ligcol, **kw)

    atomxyz = h.point(xyz[(indep.type() == 2) & ~is_gp, 1])
    if atomxyz.shape[1] and 'atom' in show:
        ipd.viz.show_ndarray_point_or_vec(atomxyz,
                                          name=f'{name}{"_atom" if split else ""}',
                                          sphere=0.3,
                                          col=(0, 1, 0),
                                          addtocgo=cgo,
                                          **kw)

    Lprot = th.sum(indep.type() == 0)
    ipd.viz.show_bonds(xyz[Lprot:, 1],
                       indep.bond_feats[Lprot:, Lprot:].cpu(),
                       name=f'{name}{"_bond" if split else ""}',
                       addtocgo=cgo,
                       **kw)

    if len(indep.chirals) and 'chirals' in show:
        ichiral = indep.chirals[:, 0].to(int).unique().cpu()
        ipd.viz.show_ndarray_point_or_vec(xyz[ichiral, 1],
                                          name=f'{name}{"_chirals" if split else ""}',
                                          sphere=0.4,
                                          col=(1, 0.5, 0.5),
                                          addtocgo=cgo,
                                          **kw)

    if not split:
        # ic(pymol.cmd.get_object_list())
        pymol.cmd.load_cgo(cgo, name)
        if name.startswith('function'):
            raise ValueError(f'bad name {name}')

    protxyz = xyz[isprot]
    # ic(protxyz.shape)
    # assert 0
    if protxyz.shape[1] and 'bb' in show:
        protbb = protxyz[:, :3].reshape(nsub, -1, 3, protxyz.shape[-1])
        contigbb = h.norm(protbb[:, 1:].reshape(nsub, -1, 4) - protbb[:, :-1].reshape(nsub, -1, 4)).max(1).values
        # ic(protbb[:,:, 1].mean(0,1))
        if th.sum(contigbb > 5) < 5:
            ipd.viz.show_ndarray_n_ca_c(protbb, name=f'{name}{"_bbone" if split else ""}', **kw)
            pymol.util.chainbow(f'{name}*')
        else:
            ipd.viz.show_ndarray_point_or_vec(protbb.reshape(sym.nsub, -1, 4),
                                              name=f'{name}{"_bbone" if split else ""}',
                                              sphere=0.3,
                                              **kw)

    # global _count
    # _count += 1
    # if _count > 30:
    # assert 0
    if sym: sym.assert_symmetry_correct(indep.xyz)

@ipd.viz.pymol_load.register(rfd.aa_model.Indep)
@ipd.viz.pymol_scene
def pymol_load_Indep(indep, name='Indep', sym=None, **kw):
    return show_indep_fancy(indep=indep, name=name, sym=sym, **kw)
    # try:
    # except Exception:  #IndexError:
    #      pymol.cmd.delete(f'{name}*')
    #     ipd.viz.show_ndarray_point_or_vec(indep.xyz[:, 1], name=name + '_ca', sphere=0.3, **kw)
