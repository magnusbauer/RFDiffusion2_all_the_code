import torch

_atom_record_format = (
    "ATOM  {atomi:5d} {atomn:^4}{idx:^1}{resn:3s} {chain:1}{resi:4d}{insert:1s}   "
    "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}{seg:-4d}{elem:2s}\n"
)


def format_atom(
        atomi=0,
        atomn='ATOM',
        idx=' ',
        resn='RES',
        chain='A',
        resi=0,
        insert=' ',
        x=0,
        y=0,
        z=0,
        occ=1,
        b=0,
        seg=1,
        elem=''
):
    '''
    Write an ATOM line as would be found in a pdb file

    Args:
        atomi (int): Atom number
        atomn (str): Atom name
        idx (str): A pdb field we never use
        resn (str): Residue name
        chain (str): Chain id
        resi (int): Residue number
        insert (str): A pdb field we never use
        x (float): x coordinate
        y (float): y coordinate
        z (float): z coordinate
        occ (float): occupancy
        b (float): b-factor
        seg (int): segment id
        elem (str): element

    Returns:
        atom_line (str): an atom line including the \n
    '''
    return _atom_record_format.format(**locals())


def draw_points(pts, name, res_name="RES", atom_name="ATOM"):
    '''
    Write a pdb file where pts are written as atoms

    Args:
        pts (torch.Tensor[float]): The atomic coords to write [L,3]
        name (str): The pdb file name (add .pdb youyself)
        res_name (str): The name of the residue that we'll be writing
        atom_name (str): The name of the atom that we'll be writing

    '''
    with open(name, "w") as f:
        for ivert, vert in enumerate(pts):
            f.write(format_atom(ivert%100000, resi=ivert%10000, x=vert[0], y=vert[1], z=vert[2], resn=res_name, atomn=atom_name))

def draw_line(start, direction, length, name):
    '''
    Draw a singular line with atoms to a pdb

    Args:
        start (torch.Tensor): The starting point [3]
        direction (torch.Tensor): The direction we will draw the line in [3]
        length (float): direction * length is the final length of the line
        name (str): The file name to write (add .pdb yourself)
    '''
    draw_lines([start], [direction], length, name)

def draw_lines(starts, directions, length, name, pts_per_line=80):
    '''
    Draw a bunch of lines using atoms to a pdb

    Args:
        starts (torch.Tensor): The starting points [L,3]
        directions (torch.Tensor): The directions we will draw the lines in [L,3]
        length (float): direction * length is the final length of the line
        name (str): The file name to write (add .pdb yourself)
        pts_per_line (int): How dense you want your lines to be
    '''

    if not isinstance(starts, torch.Tensor):
        starts = torch.tensor(starts)
    if not isinstance(directions, torch.Tensor):
        directions = torch.tensor(directions)

    vec = torch.linspace(0, length, pts_per_line)

    pt_collections = starts[:,None] + directions[:,None] * vec[:,None]

    pts = pt_collections.reshape(-1, 3)

    draw_points(pts, name)
