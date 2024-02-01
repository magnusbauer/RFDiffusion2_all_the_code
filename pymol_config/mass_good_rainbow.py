def mass_rainbow(name):
    for name in cmd.get_names("all"):
        try:
            cmd.spectrum(expression="count", palette="good_blue good_teal good_green good_yellow good_red", selection=name, minimum=None, maximum=None, byres=0, quiet=1)
        except CmdException:
            print('defies laws of pymol, skipping PDB')

cmd.extend("mass_rainbow", mass_rainbow)

def mass_paper_rainbow(name):
    for name in cmd.get_names("all"):
        try:
            cmd.spectrum(expression="count", palette="paper_navaho paper_melon paper_pink paper_purple paper_lightblue paper_blue paper_darkblue", selection=name, minimum=None, maximum=None, byres=0, quiet=1)
            #cmd.spectrum(expression="count", palette="paper_pink paper_yellow paper_blue", selection=name, minimum=None, maximum=None, byres=0, quiet=1)
        except CmdException:
            print('defies laws of pymol, skipping PDB')

cmd.extend("mass_paper_rainbow", mass_paper_rainbow)
 
@cmd.extend
def mass_paper_rainbow_sel(name):
    try:
        cmd.spectrum(expression="count", palette="paper_navaho paper_melon paper_pink paper_purple paper_lightblue paper_blue paper_darkblue", selection=name, minimum=None, maximum=None, byres=0, quiet=1)
        #cmd.spectrum(expression="count", palette="paper_pink paper_yellow paper_blue", selection=name, minimum=None, maximum=None, byres=0, quiet=1)
    except CmdException:
            print('defies laws of pymol, skipping PDB')

#cmd.extend("mass_paper_rainbow_sel", mass_paper_rainbow)
 
