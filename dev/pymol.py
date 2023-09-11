import sys
import os

import xmlrpc.client as xmlrpclib

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'RF2-allatom'))

def get_cmd(pymol_url='http://chesaw.dhcp.ipd:9123'):
    cmd = xmlrpclib.ServerProxy(pymol_url)
    if 'ipd' not in pymol_url:
        make_network_cmd(cmd)
    return cmd

cmd = None
def init(pymol_url='http://chesaw.dhcp.ipd:9123'):
    global cmd
    cmd = get_cmd(pymol_url)

def make_network_cmd(cmd):
    # old_load = cmd.load
    def new_load(*args, **kwargs):
        path = args[0]
        with open(path) as f:
            contents = f.read()
        # args[0] = contents
        args = (contents,) + args[1:]
        #print('writing contents')
        cmd.read_pdbstr(*args, **kwargs)
    cmd.is_network = True
    cmd.load = new_load