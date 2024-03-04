from dataclasses import dataclass

import rf2aa.chemical


# Trying to instantiate rf2aa.chemical.ChemicalData so any env can do it
@dataclass
class ChemConf:
    use_phospate_frames_for_NA=True

rf2aa.chemical.initialize_chemdata(ChemConf())
ChemicalData = rf2aa.chemical.ChemicalData