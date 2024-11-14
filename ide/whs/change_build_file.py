#!/usr/bin/env python
import sys
import os
import re

projfile = 'ide/whs/rf_diffusion.sublime-project'
while not os.path.exists(projfile):
    os.chdir('..')

sys.stdout = open('sublime_build.log','w')
sys.stderr = sys.stdout

def find_most_recently_changed_file(starting_directory):
    most_recent_file = None
    most_recent_time = -1
    ignore = set([])
    starting_directory = os.path.abspath(starting_directory)
    for root, dirs, files in os.walk(starting_directory):
        for file in set(files) - ignore:
            if not file.endswith('.py'): continue
            # print(root+'/'+file)
            file_path = os.path.join(root, file)
            try:
                file_mtime = os.path.getmtime(file_path)
                if file_mtime > most_recent_time:
                    most_recent_time = file_mtime
                    most_recent_file = file_path
            except FileNotFoundError:
                continue

    return most_recent_file

fname = find_most_recently_changed_file('.')

sproj_orig = open(projfile).read()

old = r'''"name": "AUTO_TEST_FILE".*?"shell_cmd": "cd \$folder/rf_diffusion; CUDA_VISIBLE_DEVICES='' PYTHONPATH=\.\. python ../ide/runtests.py .*?.py &> \$folder/sublime_build.log"'''
new = f'''"name": "AUTO_TEST_FILE",
         "shell_cmd": "cd $folder/rf_diffusion; CUDA_VISIBLE_DEVICES='' PYTHONPATH=.. python ../ide/runtests.py {fname} &> $folder/sublime_build.log"'''
sproj = re.sub(old, new, sproj_orig, flags=re.DOTALL)

old = r'''"name": "AUTO_TEST_FILE CUDA".*?"shell_cmd": "cd \$folder/rf_diffusion; rm /home/sheffler/.cache/torch_extensions/py310_cu121/./lock; PYTHONPATH=\.\. python ../ide/runtests.py .*?.py &> \$folder/sublime_build.log"'''
new = f'''"name": "AUTO_TEST_FILE CUDA",
         "shell_cmd": "cd $folder/rf_diffusion; rm /home/sheffler/.cache/torch_extensions/py310_cu121/*/lock; PYTHONPATH=.. python ../ide/runtests.py {fname} &> $folder/sublime_build.log"'''
sproj = re.sub(old, new, sproj, flags=re.DOTALL)

if sproj != sproj_orig:
    with open(projfile, 'w') as out:
        out.write(sproj)
        print(fname)
else:
    print('NO CHANGE TO', projfile)
