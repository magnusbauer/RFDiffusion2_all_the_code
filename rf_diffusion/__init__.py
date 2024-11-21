import os
import sys

from icecream import ic

import rf2aa as _  # noqa needed for registration
from ipd.dev import lazyimport, install_ipd_pre_commit_hook
# lazyimport helps with import time and eliminates many circular import issues
aa_model = lazyimport('rf_diffusion.aa_model')
atomize = lazyimport('rf_diffusion.atomize')
bond_geometry = lazyimport('rf_diffusion.bond_geometry')
contigs = lazyimport('rf_diffusion.contigs')
inference = lazyimport('rf_diffusion.inference')
metrics = lazyimport('rf_diffusion.metrics')
model_runners = lazyimport('rf_diffusion.model_runners')
noisers = lazyimport('rf_diffusion.noisers')
perturbations = lazyimport('rf_diffusion.perturbations')
rotation_conversions = lazyimport('rf_diffusion.rotation_conversions')
run_inference = lazyimport('rf_diffusion.run_inference')
sym = lazyimport('rf_diffusion.sym')
test_utils = lazyimport('rf_diffusion.test_utils')
viz = lazyimport('rf_diffusion.viz')

from rf_diffusion import observer  # noqa needed for registration
import rf_diffusion.sym.rfd_sym_manager  # noqa needed for registration

ic.configureOutput(includeContext=True)
projdir = os.path.dirname(__file__)

install_ipd_pre_commit_hook(projdir, '..')

__all__ = [
    'aa_model',
    'atomize',
    'bond_geometry',
    'contigs',
    'inference',
    'metrics',
    'noisers',
    'perturbations',
    'projdir',
    'rotation_conversions',
    'run_inference',
    'sym',
    'test_utils',
    'viz',
]

# This is necessary for backward compatibility with goldens...
 # ... pickles are a very fragile way to store data
rfd_in_path = False
for p in list(sys.path):
    rfd_in_path |= p.endswith('/rf_diffusion')
if not rfd_in_path:
    count = 0
    for p in set(sys.path):
        if os.path.exists(p + '/rf_diffusion/__init__.py'):
            if not count:
                sys.path.append(p + '/rf_diffusion')
            count += 1
    if count != 1:
        print(sys.path)
        assert count == 1
