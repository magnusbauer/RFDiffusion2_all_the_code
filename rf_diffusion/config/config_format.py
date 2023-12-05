'''
Functions for checking that the hydra conf object uses supported options
and that sets of options are compatible with each other.
'''

from omegaconf import OmegaConf

from rf_diffusion.benchmark.compile_metrics import flatten_dictionary


obsolete_options_dict = {
    'score_model.weights_path': 'Please use inference.ckpt_path instead.',
}


def alert_obsolete_options(conf):
    '''
    Raises an error if the conf uses any obsolete options.
    '''
    conf = OmegaConf.to_container(conf, resolve=True)
    conf = flatten_dictionary(conf)

    error_msg = []
    used_obsolete_options = set(conf) & set(obsolete_options_dict)
    for opt in used_obsolete_options:
        opt_note = obsolete_options_dict[opt]
        error_msg.append(f'The option "{opt}" is no longer supported. {opt_note}')

    if error_msg:
        raise ValueError('\n'.join(error_msg))


