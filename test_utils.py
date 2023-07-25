import re
import torch
import json
from icecream import ic
import os
import subprocess
from pathlib import Path
import yaml
import unittest
import pickle
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from rf2aa import tensor_util
from data_loader import (
    get_train_valid_set, loader_pdb, loader_fb, loader_complex, loader_pdb_fixbb, loader_fb_fixbb, loader_complex_fixbb, loader_cn_fixbb, default_dataset_configs,
    DistilledDataset, DistributedWeightedSampler
)
from torch.utils import data
from omegaconf import DictConfig
from data import se3_diffuser


golden_dir = 'goldens'


def golden_path(name, processor_specific=False):
    processor_name = 'any'
    if processor_specific:
        processor_name = processor()
    return Path(os.path.join(golden_dir, f'{name}_{processor_name}'))

def get_golden_path(name):
    p = golden_path()
    if not p.exists():
        raise Exception(f'golden at {p} does not exist, please generate it')
    return p


def write(path, got):
    with open(path, 'wb') as fh:
        pickle.dump(got, fh)

def read(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)

def assert_matches_golden(t, name, got, rewrite=False, processor_specific=False, custom_comparator=None):
    p = golden_path(name, processor_specific=processor_specific)
    if rewrite:
        print(f'rewriting {p}')
        write(p, got)
        # p.write_text(got)
        return
    if not p.exists():
        raise Exception(f'golden at {p} does not exist, please generate it')
    # want = p.read_text()
    want = read(p)
    if custom_comparator:
        diff = custom_comparator(got, want)
        if not diff:
            return
        try:
            jsoned = diff.to_json()
            loaded = json.loads(jsoned)
            fail_msg = json.dumps(loaded, indent=4)
        except Exception as e:
            ic('failed to pretty print output', e)
            fail_msg = json.dumps(diff.pop('tensors unequal', ''), indent=4) + '\n' + str(diff) 
        t.fail(fail_msg)
    else:
        t.assertEqual(got, want)

def assertEqual(t, custom_comparator, got, want):
    diff = custom_comparator(got, want)
    if not diff:
        return
    try:
        jsoned = diff.to_json()
        loaded = json.loads(jsoned)
        fail_msg = json.dumps(loaded, indent=4)
    except Exception as e:
        ic('failed to pretty print output', e)
        fail_msg = json.dumps(diff.pop('tensors unequal', ''), indent=4) + '\n' + str(diff) 
    t.fail(fail_msg)


def processor():
    o = subprocess.run(['lscpu | grep "Model name"'], capture_output=True, shell=True)
    s = o.stdout.decode("utf-8")
    s = s[len("Model name:"):].strip()
    ncpus = available_cpu_count()
    s = f'{s}-{ncpus}CPUs'
    return s

def available_cpu_count():
    """ Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program"""

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        pass

    # Python 2.6+
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # https://github.com/giampaolo/psutil
    try:
        import psutil
        return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
    except (ImportError, AttributeError):
        pass

    # POSIX
    try:
        res = int(os.sysconf('SC_NPROCESSORS_ONLN'))

        if res > 0:
            return res
    except (AttributeError, ValueError):
        pass

    # Windows
    try:
        res = int(os.environ['NUMBER_OF_PROCESSORS'])

        if res > 0:
            return res
    except (KeyError, ValueError):
        pass

    # jython
    try:
        from java.lang import Runtime
        runtime = Runtime.getRuntime()
        res = runtime.availableProcessors()
        if res > 0:
            return res
    except ImportError:
        pass

    # BSD
    try:
        sysctl = subprocess.Popen(['sysctl', '-n', 'hw.ncpu'],
                                  stdout=subprocess.PIPE)
        scStdout = sysctl.communicate()[0]
        res = int(scStdout)

        if res > 0:
            return res
    except (OSError, ValueError):
        pass

    # Linux
    try:
        res = open('/proc/cpuinfo').read().count('processor\t:')

        if res > 0:
            return res
    except IOError:
        pass

    # Solaris
    try:
        pseudoDevices = os.listdir('/devices/pseudo/')
        res = 0
        for pd in pseudoDevices:
            if re.match(r'^cpuid@[0-9]+$', pd):
                res += 1

        if res > 0:
            return res
    except OSError:
        pass

    # Other UNIXes (heuristic)
    try:
        try:
            dmesg = open('/var/run/dmesg.boot').read()
        except IOError:
            dmesgProcess = subprocess.Popen(['dmesg'], stdout=subprocess.PIPE)
            dmesg = dmesgProcess.communicate()[0]

        res = 0
        while '\ncpu' + str(res) + ':' in dmesg:
            res += 1

        if res > 0:
            return res
    except OSError:
        pass

    raise Exception('Can not determine number of CPUs on this system')

def cmp_pretty(got, want, **kwargs):
    diff = tensor_util.cmp(got, want, **kwargs)
    if not diff:
        return
    try:
        jsoned = diff.to_json()
        loaded = json.loads(jsoned)
        return json.dumps(loaded, indent=4)
    except Exception as e:
        ic('failed to pretty print output', e)
        return json.dumps(diff.pop('tensors unequal', ''), indent=4) + '\n' + str(diff)

def where_nan(t):
    if torch.isnan(t).any():
        return (torch.isnan(t).nonzero(), torch.isnan(t).float().mean())
    return None

def assert_no_nan(t):
    msg = where_nan(t)
    if msg:
        raise Exception(msg)

def construct_conf(overrides=None, config_name='debug', train_config_name='train_covale_complex', inference=False):
    if not inference:
        return construct_conf_single(overrides=overrides, config_name=config_name, inference=inference)
    train_conf = construct_conf_single(config_name=train_config_name, inference=False)
    inference_conf = construct_conf_single(overrides=overrides, config_name=config_name, inference=inference)
    OmegaConf.set_struct(train_conf, False)
    OmegaConf.set_struct(inference_conf, False)
    conf = OmegaConf.merge(
        train_conf, inference_conf)
    return conf
    

def construct_conf_single(overrides=None, config_name='debug', inference=False):
    hydra.core.global_hydra.GlobalHydra().clear()
    overrides = overrides or []
    config_path = 'config/training'
    if inference:
        config_path = 'config/inference'
    initialize(version_base=None, config_path=config_path, job_name="test_app")
    conf = compose(config_name=f'{config_name}.yaml', overrides=overrides, return_hydra_config=True)
    return conf

def no_batch_collate_fn(data):
    assert len(data) == 1
    return data[0]

def get_dataloader(conf: DictConfig, epoch=0) -> None:

    if conf.debug:
        ic.configureOutput(includeContext=True)
    diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)
    diffuser.T = conf.diffuser.T
    dataset_configs, homo = default_dataset_configs(conf.dataloader, debug=conf.debug)

    print('Making train sets')
    train_set = DistilledDataset(dataset_configs,
                                    conf.dataloader, diffuser,
                                    conf.preprocess, conf, homo)
    
    train_sampler = DistributedWeightedSampler(dataset_configs,
                                                dataset_options=conf.dataloader['DATASETS'],
                                                dataset_prob=conf.dataloader['DATASET_PROB'],
                                                num_example_per_epoch=conf.epoch_size,
                                                num_replicas=1, rank=0, replacement=True)
    train_sampler.epoch = epoch
    
    # mp.cpu_count()-1
    LOAD_PARAM = {
        'shuffle': False,
        'num_workers': 0,
        'pin_memory': True
    }
    train_loader = data.DataLoader(train_set, sampler=train_sampler, batch_size=conf.batch_size, collate_fn=no_batch_collate_fn, **LOAD_PARAM)
    return train_loader

def loader_out_from_conf(conf, epoch=0):
    dataloader = get_dataloader(conf, epoch)
    for loader_out in dataloader:
        # indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context = loader_out
        # indep.metadata = None
        return loader_out
    
def loader_out_for_dataset(dataset, mask, overrides=[], epoch=0, config_name='debug'):
    conf = construct_conf([
        'dataloader.DATAPKL_AA=aa_dataset_256_subsampled_10.pkl',
        'dataloader.CROP=256',
        f'dataloader.DATASETS={dataset}',
        f'dataloader.DATASET_PROB=[1.0]',
        f'dataloader.DIFF_MASK_PROBS=null',
        f'dataloader.DIFF_MASK_PROBS={{{mask}:1.0}}',
        'debug=True',
        'spoof_item=null',
    ] + overrides, config_name=config_name)
    
    return loader_out_from_conf(conf, epoch=epoch)
