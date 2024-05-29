from setuptools import setup

setup(
    name="rf_diffusion",
    #    install_requires=[f'se3_flow_matching @ file://localhost/home/ahern/projects/aa/rf_diffusion_flow/lib/se3_flow_matching/'],
    packages=[
        'rf_diffusion',
        'rf_se3_diffusion',
        'rf2aa',
        'se3_flow_matching',
    ],
    package_dir={
        'rf2aa': 'RF2-allatom',
        'se3_flow_matching': 'lib/se3_flow_matching/se3_flow_matching',
    },
    version='1.0')
