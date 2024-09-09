from dataclasses import dataclass
import rf2aa.chemical

@dataclass
class ChemConf:
    use_phospate_frames_for_NA: bool
    use_lj_params_for_atoms: bool

@dataclass
class ChemConfConf:
    chem_params: ChemConf

def initialize_chemical_data(use_phospate_frames_for_NA: bool = True, use_lj_params_for_atoms: bool = False):
    chem_conf = ChemConf(
        use_phospate_frames_for_NA,
        use_lj_params_for_atoms
    )
    chem_conf_conf = ChemConfConf(chem_conf)
    rf2aa.chemical.initialize_chemdata(chem_conf_conf)
    return rf2aa.chemical.ChemicalData

# Default initialization
ChemicalData = initialize_chemical_data()

# Function to reinitialize with custom parameters
def reinitialize_chemical_data(use_phospate_frames_for_NA: bool = True, use_lj_params_for_atoms: bool = False):
    global ChemicalData
    ChemicalData = initialize_chemical_data(use_phospate_frames_for_NA, use_lj_params_for_atoms)
