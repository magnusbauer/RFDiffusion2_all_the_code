from dataclasses import dataclass

import rf2aa.chemical


# Trying to instantiate rf2aa.chemical.ChemicalData so any env can do it
@dataclass
class ChemConf:
    use_phospate_frames_for_NA=True
    use_lj_params_for_atoms=False

@dataclass
class ChemConfConf:
    chem_params = ChemConf()

rf2aa.chemical.initialize_chemdata(ChemConfConf())
ChemicalData = rf2aa.chemical.ChemicalData