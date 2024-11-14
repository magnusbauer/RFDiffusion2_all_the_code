import rf_diffusion as rfd
import ipd

class RFDMotifManager(ipd.motif.MotifManager):
    kind = 'rfd'

    def __init__(self, **kw):
        super().__init__(**kw)

    def setup_for_motifs(self, thing):
        if isinstance(thing, rfd.aa_model.Indep):
            ...
        return thing
