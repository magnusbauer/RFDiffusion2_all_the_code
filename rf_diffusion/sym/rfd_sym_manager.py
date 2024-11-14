import collections
import logging
import random
import string

import torch as th

import ipd
import rf2aa as rf
import rf_diffusion as rfd

log = logging.getLogger(__name__)

SymContig = collections.namedtuple('SymContig', ['contig', 'atoms', 'has_termini'])

class RFDSymmetryManager(rf.sym.RF2SymmetryManager):
    """Symmetry manager for rf_diffusion"""
    kind = 'rf_diffusion'

    def init(self, *a, **kw):
        super().init(*a, **kw)
        assert 'idx' in dir(self)
        self._captured_indep = None

    def symmetrize_contigs(self, contigs, contig_atoms, has_termini, length=None):
        '''symmetrize a contigs string. if a range a-b is specified, choose the random value
        here, to ensure all are same for symmetrical structure'''
        if not self or self.opt.contig_is_symmetric:
            return contigs, contig_atoms, has_termini
        assert len(contigs) == 1
        contigs = contigs[0].split('_')
        chains = [c.split(',') for c in contigs]
        chains = [map(_picksize, c) for c in chains]
        chains = [str.join(',', c) for c in chains]
        contigs = [str.join('_', chains)]
        contigs = contigs * self.nsub
        print(contigs)
        contig_atoms = contig_atoms or {}
        catoms = list(contig_atoms.keys())
        contig_atoms = contig_atoms.copy()
        catoms = list(contig_atoms.keys()) if contig_atoms else []
        contig_atoms = contig_atoms.copy() if contig_atoms else None
        for i in range(1, self.nsub):
            old, new = 'AA'
            if self.opt.contig_relabel_chains: new = string.ascii_uppercase[i]
            contigs[i] = contigs[i].replace(old, new)
            for res in catoms:
                assert res.startswith('A')
                contig_atoms[res.replace(old, new)] = contig_atoms[res]
        # ic(string.ascii_uppercase)
        # ic(contigs)
        contigs = [str.join('_', contigs)]
        # ic(contig_atoms)
        # assert 0
        has_termini = list(has_termini) * self.nsub
        return SymContig(contigs, contig_atoms, has_termini)

    def setup_for_symmetry(self, thing):
        if isinstance(thing, rfd.aa_model.Indep):
            thing.xyz = self.apply_initial_offset(thing.xyz)
            thing = self(thing)
        return thing

    def write_sym_pdb(self,
                      out_idealized,
                      xyz_design_idealized,
                      seq_design,
                      bond_feats,
                      ligand_name_arr=None,
                      chain_Ls=None,
                      idx_pdb=None):
        # symmetrize pdb and dump out full design
        Lasu = chain_Ls[0]  # this only works for homooligomers
        O = len(self.allsymmRs)
        B = xyz_design_idealized.shape[0]
        xyz_full = th.zeros((B, O * Lasu, 36, 3), device=xyz_design_idealized.device)
        chain_Ls = [chain_Ls[0]] * O
        seq_design = seq_design[:Lasu].repeat(O)
        idx_pdb = th.tensor([i for i in range(1, chain_Ls[0] * O + 1)])
        if ligand_name_arr is not None:
            ligand_name_arr = ligand_name_arr[:Lasu].repeat(O)

        for i in range(1, len(self.allsymmRs)):
            xyz_full[:, (i * Lasu):((i+1) * Lasu), :4, :] = th.einsum('ij,braj->brai',
                                                                      self.allsymmRs.cpu()[i],
                                                                      xyz_design_idealized.cpu()[:, :Lasu])[:, :Lasu, :4]

        rfd.aa_model.write_traj(out_idealized,
                                xyz_full,
                                seq_design,
                                None,
                                ligand_name_arr=ligand_name_arr,
                                chain_Ls=chain_Ls,
                                idx_pdb=idx_pdb)
        # rfd.dev.idealize_backbone.rewrite(out_idealized, out_idealized)

ipd.sym.set_default_sym_manager('rf_diffusion')

def _picksize(segment):
    if '-' in segment and segment[0].isdigit() and not any(map(str.isalpha, segment)):
        raise NotImplementedError('size picking not working for symmetry yet')
        segment = str(random.randint(*[int(i) for i in segment.split('-')]))
    return segment
