'''
utilties for symmetrizing an Indep
'''

import torch as th

import ipd
import rf_diffusion as rfd

def set_indep_slices(sym, indep, isasym):
    '''set the slices for the symmetrization of an Indep object'''
    nfakeprot = indep.is_gp[indep.chirals[:, 0].to(int).unique()].sum()
    # if th.any(indep.type() != 0):
    # firstnonprot = th.where(indep.type() != 0)[0].min()
    # else:
    # firstnonprot = len(indep.type())
    # idx = (firstnonprot - nfakeprot) // sym.nsub
    isfakeprot = th.zeros(len(indep.seq), dtype=bool)
    # ic(idx, nfakeprot)
    isfakeprot[:nfakeprot] = True
    colors = indep.type() + indep.is_gp * 10
    if not sym.opt.ligand_is_symmetric:
        colors[colors == 1] = -100
    if not sym.opt.guideposts_are_symmetric:
        colors[indep.is_gp] = -100
    # colors[isfakeprot] = -100
    if sym.opt.motif_position == 'fixed':
        onaxis = sym.is_on_symaxis(indep.xyz[indep.type() > 0, 1])
        if onaxis is not None:
            colors[th.sum(indep.type() == 0) + th.where(onaxis)[0]] = -100
    # colors[indep.is_gp] += 10
    # ic(indep.type())
    # ic(colors)
    slices = ipd.sym.symslices_from_colors(sym.nsub, colors, isasym)
    idx = ipd.sym.SymIndex(sym.nsub, slices)
    sym.idx = idx
    sym.idx.set_kind(sym(indep.type() + indep.is_gp * 10))
    isfakeprot = sym(isfakeprot)
    sym.idx.isfakeprot = isfakeprot
    sym.idx.kind[isfakeprot] = -1
    sym.idx.gpca = indep.chirals[:, 0].to(int).unique()
    ipd.hub.new_symmetry(sym)
    # ic(sym.idx.isfakeprot)
    # ic(sym)

class SymAdaptIndep(ipd.sym.SymAdaptDataClass):
    '''
    This class is a wrapper around the Indep class from rf_diffusion. It is used to
    symmetrize the Indep object.
    '''
    adapts = rfd.aa_model.Indep

    def __init__(self, indep, sym, isasym):
        super().__init__(indep, sym, isasym)
        if not sym.idx or not sym.idx.match_indep(indep):
            set_indep_slices(self.sym, indep, isasym)
        self.chirals = self.adapted['chirals']
        del self.adapted['chirals']

    def reconstruct(self, symparts, **kw):
        '''returns: the unholy Indep'''
        symparts['chirals'] = self.chirals
        new = super().reconstruct(symparts)
        for field in set(dir(self.orig)) - set(dir(new)):
            setattr(new, field, getattr(self.orig, field))
        new.idx = th.arange(len(new.idx))
        if self.sym.pseudo_cycle:
            new.terminus_type[:] = 0
            new.same_chain[:] = True
        new.idx = renumber_idx(new.same_chain)
        self.sym.idx.set_indep(new)
        # S = sym.idx
        # C = self.orig.chirals
        #  new.chirals = th.cat([C] * sym.nsub)
        #  for i in range(sym.nsub):
        # symC = S.idx_sub_to_sym[i, S.idx_asym_to_asu[C[:, :4].to(int)]]
        # new.chirals[i * len(C):(i + 1) * len(C), :4] = symC
        # ic(C[:,:4].to(int))
        # ic(new.chirals[:,:4].to(int))
        return new

def renumber_idx(same_chain, spacing=200):
    '''
    generate idx such that spacing beween chains is increased
    '''
    L = len(same_chain)
    # ic(same_chain.to(int))
    # ic(~same_chain[th.arange(L - 1), th.arange(1, L)])
    breaks = [0] + list(th.where(~same_chain[th.arange(L - 1), th.arange(1, L)])[0] + 1) + [L]
    idx = th.cat([th.arange(lb, ub) + spacing*i for i, (lb, ub) in enumerate(zip(breaks[:-1], breaks[1:]))])
    # ic(breaks)
    # ic(idx)
    return idx
