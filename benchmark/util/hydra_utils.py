import hydra
from hydra.core.hydra_config import HydraConfig
import omegaconf

def to_dotdict(d: dict, prefix: str=None) -> dict:
    '''
    Convert a nested dictionary to a flat dictionary
    with dot-type notation for the keys.

    Handy for converting a hydra config to command line
    overrides.
    '''
    
    dotdict = {}
    
    for k, v in d.items():
        # Resolve appropriate key/prefix
        if prefix is None:
            key = k
        else:
            key = f'{prefix}.{k}'
            
        if isinstance(v, dict):
            child_dotdict = to_dotdict(v, prefix=key)
            dotdict.update(child_dotdict)
        else:
            dotdict[key] = v
            
    return dotdict

def hydra_overrides(conf: HydraConfig) -> str:
    '''
    Convert a HydraConfig to their equivalent command line arguments.
    Useful for passing configuration setting to subprocesses and 
    new slurm jobs.
    
    NOTE: This resolves any variable expansion/interpolation
    in the conf.
    '''
    conf_dict = omegaconf.OmegaConf.to_container(conf, resolve=True)
    conf_dotdict = to_dotdict(conf_dict)
    
    cmd = []
    for k, v in conf_dotdict.items():
        if (v is None) or (v==''):
            # Could this cause problems?
            #continue
            v = 'null'

        if isinstance(v, list):
            # Need to ensure no spaces in the stringified list
            v = '[' + ','.join(v) + ']'

        # Some special characters hydra will try to parse. We need to protect them with quotes.
        v = str(v)
        special_chars = "\()[]{}:=, |'\""
        if any(char in v for char in special_chars):
            v = v.replace('"', '\\"')
            v = v.replace("'", "'\\''")
            cmd.append(f'\'{k}="{v}"\'')
        else:
            cmd.append(f'{k}={v}')
           
    cmd = ' '.join(cmd)
            
    return cmd