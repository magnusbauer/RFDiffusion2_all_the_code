from setuptools import setup

setup(
    name="rf_diffusion",
    packages=[
        'rf_diffusion',
        'rf_se3_diffusion',
        'rf2aa',
    ],
    package_dir={
        'rf2aa': 'RF2-allatom',
    },
    version = '1.0'
)
