import os
import copy


from rf_diffusion.dev import show_tip_row
import rf_diffusion.aa_model as aa_model
from rf_diffusion.dev import pymol

cmd = pymol.cmd

counter = 1

def get_counter():
    global counter
    counter += 1
    return counter - 1


def show_backbone_spheres(selection):
    cmd.hide('everything', selection)
    cmd.alter(f'name CA and {selection}', 'vdw=2.0')
    cmd.show('spheres', f'name CA and {selection}')
    cmd.show('licorice', f'{selection} and (name CA or name C or name N)')

def one(indep, atomizer, name=''):

    if not name:
        name = f'protein_{get_counter()}'

    if atomizer:
        indep = atomizer.deatomize(indep)
    pdb = f'/tmp/{name}.pdb'
    names = indep.write_pdb(pdb)

    cmd.load(pdb, name)
    name = os.path.basename(pdb[:-4])
    cmd.show_as('cartoon', name)
    # show_backbone_spheres('not hetatm')
    cmd.show('licorice', f'hetatm and {name}')
    cmd.color('orange', f'hetatm and elem c and {name}')
    return name, names

def AND(*names):
    return show_tip_row.AND(names)

def OR(*names):
    return show_tip_row.OR(names)

def color_masks(name, names, color_by_mask):
    for mask, color in color_by_mask.items():
        index_selectors = names[mask]
        if len(index_selectors) == 0:
            continue
        selector = AND(name, OR(*index_selectors))
        cmd.color(color, selector)

def diffused(indep, is_diffused, name=None):

    if not name:
        name = f'protein_{get_counter()}'

    indep_motif = copy.deepcopy(indep)
    indep_diffused = copy.deepcopy(indep)
    aa_model.pop_mask(indep_motif, ~is_diffused, break_chirals=True)
    aa_model.pop_mask(indep_diffused, is_diffused, break_chirals=True)
    one(indep_motif, None, f'{name}_motif')
    one(indep_diffused, None, f'{name}_diffused')

def color_diffused(indep, is_diffused, name=None):
    name, names = one(indep, None, name)
    show_backbone_spheres(f'{name} and (not hetatm)')
    color_masks(
        name,
        names,
        {
            is_diffused: 'blue',
            ~is_diffused: 'red',
        }
    )
