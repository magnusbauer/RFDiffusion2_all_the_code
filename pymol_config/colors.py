import pymol 
from pymol import cmd 

def get_colors():
    """
    Creates extra Sam-approved colors in pymol 
    """
    cmd.set_color("nitrogen",        [2,118,253])
    cmd.set_color("pymol_gray",      [51,51,51])
    cmd.set_color("pymol_black",     [34,34,34])
    cmd.set_color("good_yellow",     [250,199,44])
    cmd.set_color("good_teal",       [41,176,193])
    cmd.set_color("good_green",      [170,195,47])
    cmd.set_color("good_pink",       [236,114,164])
    cmd.set_color("good_blue",       [68,153,231])
    cmd.set_color("good_gray",       [220,220,220])
    cmd.set_color("good_red",        [228,74,62])
    cmd.set_color("good_light_green",[101,179,124])

    #cmd.set_color("paper_yellow",    [1,0.878,0.675])
    #cmd.set_color("paper_pink",      [1.0,0.675,0.718])
    #cmd.set_color("paper_blue",      [0.408,0.525,0.773])
    cmd.set_color("paper_teal",      [ 0.310, 0.725, 0.686 ])
    cmd.set_color("paper_navaho",     [255,224,172]) #FFE0AC1
    cmd.set_color("paper_melon",      [255,198,178]) #FFC6B2
    cmd.set_color("paper_pink",       [255,172,183]) #FFACB7
    cmd.set_color("paper_purple",     [213,154,181]) #D59AB5
    cmd.set_color("paper_lightblue",  [149,150,198]) #9596C6
    cmd.set_color("paper_blue",       [102,134,197]) #6686C5
    cmd.set_color("paper_darkblue",   [75,95,170]) #4B5FAA

def get_lighting():
    """
    Sets lighting conditions 
    """
    cmd.do('set specular, 0')
    cmd.do('set ray_shadow, off')
    cmd.do('set valence, off')
    cmd.do('set antialias, 2')
    cmd.do('set ray_trace_mode, 1')
    cmd.do('set ray_trace_disco_factor, 1')
    cmd.do('set ray_trace_gain, 0.1')
    cmd.do('set power, 0.2')
    cmd.do('set ambient, 0.4')
    cmd.do('set ray_trace_color, gray30')

cmd.extend('get_colors',get_colors)
cmd.extend('get_lighting', get_lighting)