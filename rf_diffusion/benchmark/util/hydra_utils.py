import os
from hydra import compose, initialize

def construct_conf(overrides: list[str]=[], yaml_path: str='config/training/base.yaml'):
    '''
    Make a hydra config object from a yaml configutation file.
    
    Inputs
        overrides: ex - ['inference.cautious=False', 'inference.design_startnum=0']
        yaml_path: Yaml file from which to construct the conf. Can be an absolute path. Then overrides are applied.        
    '''
    yaml_path = os.path.relpath(yaml_path, start=os.path.dirname(os.path.abspath(__file__)))  # hydra requires relative paths.
    config_path, config_name = os.path.split(yaml_path)
    with initialize(version_base=None, config_path=config_path, job_name="test_app"):
        conf = compose(config_name=config_name, overrides=overrides, return_hydra_config=True)
        
    return conf