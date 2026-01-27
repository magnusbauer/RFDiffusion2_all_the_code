import contextlib
import os
import sys


def _ensure_atomworks_in_path() -> None:
    """Allow in-repo atomworks sources to be imported without manual PYTHONPATH tweaks."""
    repo_root = os.path.dirname(os.path.dirname(__file__))
    atomworks_src = os.path.realpath(os.path.join(repo_root, 'lib', 'atomworks', 'src'))
    if not os.path.isdir(atomworks_src):
        return

    for existing in sys.path:
        try:
            if os.path.realpath(existing) == atomworks_src:
                return
        except (OSError, TypeError):  # some entries may be non-path values
            continue

    sys.path.append(atomworks_src)


_ensure_atomworks_in_path()


def _patch_ipd_safe_eval() -> None:
    """Fallback for ipd.dev.safe_eval when RestrictedPython isn't installed."""
    try:
        from importlib.util import find_spec
        from ipd.dev import safe_eval as ipd_safe_eval  # type: ignore
    except ImportError:
        return

    if find_spec('RestrictedPython') is not None:
        return

    import ast

    def _literal_eval(code: str, **kw):
        try:
            return ast.literal_eval(code)
        except Exception:
            return eval(code, {}, kw)  # noqa: S307

    def _literal_exec(code: str, **kw):
        exec(code, kw)

    ipd_safe_eval.safe_eval = _literal_eval  # type: ignore[attr-defined]
    ipd_safe_eval.safe_exec = _literal_exec  # type: ignore[attr-defined]


_patch_ipd_safe_eval()

import rf2aa as _  # noqa needed for registration
from ipd.dev import install_ipd_pre_commit_hook, lazyimport

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
# import rf_diffusion.sym.rfd_sym_manager  # noqa needed for registration

with contextlib.suppress(ImportError):
    from icecream import ic
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
