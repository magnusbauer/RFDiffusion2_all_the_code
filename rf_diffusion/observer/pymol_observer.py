import copy
import logging
import sys

import torch as th

import ipd
import rf_diffusion as rfd

log = logging.getLogger(__name__)

try:
    pymol = ipd.lazyimport('pymol')

    class PymolObserver(ipd.observer.Observer):
        '''
        Observer that visualizes an indep or xyz in PyMol.

        Automatically registers all events viz.whatever from the config
        '''
        def __init__(self):
            self.indep = None
            self.settings = {}
            self.viz_on_methods = set()

        def set_config(self, conf):
            '''automoatically register all events viz.whatever from the config'''
            if 'viz' not in conf or not conf.viz.settings.enabled: return
            for k, v in conf.viz.items():
                if k == 'settings':
                    self.settings = dict(v)
                elif v is True:
                    if hasattr(self, k):
                        self.viz_on_methods.add(k)
                    setattr(self, k, self.show_xyz_in_pymol)
                    log.debug(f'VIZ ON FOR {k}')
            if 'showinput' in self.settings and self.settings['showinput'] and conf.inference.input_pdb:
                ipd.showme(conf.inference.input_pdb, 'ref')

        def new_indep(self, indep, **kw):
            self._new_indep(indep)
            if 'new_indep' in self.viz_on_methods:
                self.show_xyz_in_pymol(indep, methodname='new_indep', **kw)

        def new_sym_indep(self, indep, **kw):
            self._new_indep(indep)
            if 'new_sym_indep' in self.viz_on_methods:
                self.show_xyz_in_pymol(indep, methodname='new_sym_indep', **kw)

        def _new_indep(self, indep):
            '''keep a copy of the indep for later use in show_xyz_in_pymol'''
            self.indep = copy.deepcopy(indep)
            if ipd.symmetrize.is_symmetrical(indep):
                self.sym_indep = indep
                self.asym_indep = ipd.symmetrize.asym(indep)
            else:
                self.asym_indep = indep

        def show_xyz_in_pymol(self, xyz, methodname, **kw):
            '''Show the indep or xyz in PyMol'''
            kw = dict(name=methodname) | self.settings | kw
            if xyz is None: sys.exit()
            if th.rand(1) > kw['showfrac']: return
            # pymol.cmd.do('axes')
            if isinstance(xyz, rfd.aa_model.Indep):
                self._new_indep(xyz)
                xyz = self.indep.xyz
            elif xyz.shape[0] == 1:
                xyz = xyz[0]
            if self.indep:
                if hasattr(self, 'sym_indep'):
                    indep = self.sym_indep if ipd.symmetrize.is_symmetrical(xyz) else self.asym_indep
                else:
                    indep = self.indep
                assert indep.xyz.shape[0] == xyz.shape[0]
                indep.xyz = xyz.cpu()
                ipd.showme(indep, sym=ipd.symmetrize, **kw)
            else:
                ic('!!!!!!!!!!!!!!!!!!!!')
                assert 0
                ipd.showme(xyz, sym=ipd.symmetrize, **kw)
            if methodname == 'debug_transforms':
                indep.print1d(compact=True)

except ImportError:
    pass
