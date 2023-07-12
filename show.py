import os
import copy

from dev import analyze, show_tip_pa
import atomize
import aa_model

cmd = analyze.cmd

counter = 1

def get_counter():
    global counter
    counter += 1
    return counter - 1


def show_backbone_spheres(selection):
    cmd.hide('everything', selection)
    cmd.alter(f'name CA and {selection}', 'vdw=2.0')
    cmd.set('sphere_transparency', 0.1)
    cmd.show('spheres', f'name CA and {selection}')
    cmd.show('licorice', f'{selection} and (name CA or name C or name N)')

def one(indep, atomizer, name=''):

    if not name:
        name = f'protein_{get_counter()}'

    if atomizer:
        indep = atomize.deatomize(atomizer, indep)
    pdb = f'/tmp/{name}.pdb'
    indep.write_pdb(pdb)

    cmd.load(pdb)
    name = os.path.basename(pdb[:-4])
    cmd.show_as('cartoon', name)
    # show_backbone_spheres('not hetatm')
    cmd.show('licorice', f'hetatm and {name}')
    cmd.color('orange', f'hetatm and elem c and {name}')

def diffused(indep, is_diffused, name):

    indep_motif = copy.deepcopy(indep)
    indep_diffused = copy.deepcopy(indep)
    aa_model.pop_mask(indep_motif, ~is_diffused)
    aa_model.pop_mask(indep_diffused, is_diffused)
    one(indep_motif, None, f'{name}_motif')
    one(indep_diffused, None, f'{name}_diffused')