import os

from icecream import ic
import torch
import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf

import rf_diffusion.train_multi_deep


@hydra.main(version_base=None, config_path="config/restart", config_name="base")
def main(conf: DictConfig) -> None:
    '''
    Restarts a training run using the config saved in the model's checkpoint.  Can also be used with the config
    `log_all_metrics` to restart the training with no loss and log many metrics.
    '''
    print(conf)

    if torch.cuda.device_count():
        map_location = {"cuda:%d"%0: "cuda:%d"%0}
    else:
        map_location = {"cuda:%d"%0: "cpu"}
    ic(map_location)
    checkpoint = torch.load(conf.model_ckpt, map_location=map_location, weights_only=False)

    model_conf = checkpoint['conf']
    restart_training_conf = conf.training
    OmegaConf.set_struct(model_conf, False)
    OmegaConf.set_struct(restart_training_conf, False)
    
    training_conf = OmegaConf.merge(
        model_conf, restart_training_conf)
    
    training_conf.rundir = os.path.join(model_conf.rundir, 'restart')
    training_conf.ckpt_load_path = conf.model_ckpt
    print(omegaconf.OmegaConf.to_yaml(training_conf))
    rf_diffusion.train_multi_deep.run(
        training_conf
    )

if __name__ == "__main__":
    main()
