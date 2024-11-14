import omegaconf
import pytest
import torch as th
import yaml

import ipd
import rf_diffusion as rfd

def main():
    test_sym_indep()
    test_sym_indep_gp_atom_lig_asym()
    test_sym_indep_gp_lig_asym_chains()
    test_sym_indep_gp_atom_lig_sym()
    test_renumber_idx()
    print('DONE')

#def construct_conf(name, overrides=list()):
#    fname = os.path.relpath(f'{rfd.projdir}/config/inference', start=os.path.dirname(__file__))
#    hydra.initialize(version_base=None, config_path=fname, job_name="test_app")
#    conf = hydra.compose(config_name=f'{name}.yaml', overrides=overrides, return_hydra_config=True)
#    # This is necessary so that when the model_runner is picking up the overrides, it finds them set on HydraConfig.
#    hydra.core.hydra_config.HydraConfig.instance().set_config(conf)
#    # conf = compose(config_name='aa_small.yaml', overrides=overrides)
#    return conf

def test_sym_indep_gp_atom_lig_asym():
    contigconf = omegaconf.DictConfig(
        yaml.load(
            '''
          contig_atoms:
            A11: NE,CZ,NH1,NH2
            A28: NE,CZ,NH1,NH2
            A39: CD,CE,NZ
            A46: O,C,CA
            A48: CG,OD1,OD2
            A52: CD,OE1,OE2
            A84: CB,OG
            A88: CD,OE1,NE2
          contigs: ["10,A11,A28,A39,A46,A48,A52,A84,A88"]
          has_termini: [True]
    ''', yaml.Loader))
    conf = rfd.test_utils.construct_conf(inference=True, config_name='sym')
    conf.contigmap = contigconf
    conf.inference.contig_as_guidepost = True
    conf.inference.ligand = 'TSA,TSB'  # TODO: chirals don't read properly if not unique lig names
    conf.inference.input_pdb = f'{rfd.projdir}/benchmark/input/sym_gp_AB.pdb'
    conf.sym.contig_is_symmetric = False
    conf.sym.fit = False
    conf.sym.contig_relabel_chains = True
    conf.sym.move_unsym_with_asu = False
    conf.sym.start_radius = 0.0
    conf.sym.sympair_protein_only = False
    conf.sym.symid = 'C1'
    conf.sym.contig_is_symmetric = False
    sym = ipd.sym.create_sym_manager(conf)
    assert sym.symid == 'C1'
    diffuser = rfd.noisers.get(conf.diffuser)
    _, _, indep, *_ = rfd.inference.data_loader.InferenceDataset(conf, diffuser)[0]
    # print(indep.xyz.shape)
    assert th.all(indep.seq >= 21)
    # ipd.showme(indep, name='ASYM')
    symindep = sym(indep)
    # ic(indep.seq.shape,symindep.seq.shape)
    # ic(indep.seq)
    # ic(symindep.seq)
    # indep.print1d()
    # symindep.print1d()
    # ipd.showme(symindep, name='SYM')
    assert th.allclose(indep.seq, symindep.seq)
    assert th.allclose(indep.is_gp, symindep.is_gp)
    assert th.allclose(indep.type(), symindep.type())

def test_sym_indep_gp_lig_asym_chains():
    contigconf = omegaconf.DictConfig(
        yaml.load('''
          contigs: ["10,A11"]
          has_termini: [True]
    ''', yaml.Loader))
    conf = rfd.test_utils.construct_conf(inference=True, config_name='sym')
    conf.contigmap = contigconf
    conf.inference.contig_as_guidepost = True
    conf.inference.ligand = 'TSA,TSB'  # TODO: chirals don't read properly if not unique lig names
    conf.inference.input_pdb = f'{rfd.projdir}/benchmark/input/sym_gp_AB.pdb'
    conf.sym.contig_is_symmetric = False
    conf.sym.fit = False
    conf.sym.move_unsym_with_asu = False
    conf.sym.start_radius = 0.0
    conf.sym.sympair_protein_only = False
    conf.sym.symid = 'C2'
    conf.sym.contig_is_symmetric = False
    sym = ipd.sym.create_sym_manager(conf)
    diffuser = rfd.noisers.get(conf.diffuser)
    _, _, indep, *_ = rfd.inference.data_loader.InferenceDataset(conf, diffuser)[0]
    print(indep.xyz.shape)
    indep.print1d()
    # print(indep.seq)
    assert th.all((indep.seq >= 21) + (indep.seq == 1))
    # ipd.showme(indep, name='ASYM')
    symindep = sym(indep)
    # ipd.showme(symindep, name='SYM')
    assert th.allclose(indep.seq, symindep.seq)
    assert th.allclose(indep.is_gp, symindep.is_gp)
    assert th.allclose(indep.type(), symindep.type())

def make_symgp_conf_sym():
    contigconf = omegaconf.DictConfig(
        yaml.load(
            '''
          contig_atoms:
            A11: NE,CZ,NH1,NH2
            A28: NE,CZ,NH1,NH2
            A39: CD,CE,NZ
            A46: O,C,CA
            A48: CG,OD1,OD2
            A52: CD,OE1,OE2
            A84: CB,OG
            A88: CD,OE1,NE2
            B11: NE,CZ,NH1,NH2
            B28: NE,CZ,NH1,NH2
            B39: CD,CE,NZ
            B46: O,C,CA
            B48: CG,OD1,OD2
            B52: CD,OE1,OE2
            B84: CB,OG
            B88: CD,OE1,NE2
          contigs: ["92,A11,A28,A39,A46,A48,A52,A84,A88_92,B11,B28,B39,B48,B52,B46,B84,B88"]
          has_termini: [True, True]
    ''', yaml.Loader))
    conf = rfd.test_utils.construct_conf(inference=True, config_name='sym')
    conf.contigmap = contigconf
    conf.inference.contig_as_guidepost = True
    conf.inference.ligand = 'TSA'
    conf.inference.input_pdb = f'{rfd.projdir}/benchmark/input/sym_gp_AB.pdb'
    conf.sym.contig_is_symmetric = True
    conf.sym.fit = False
    conf.sym.move_unsym_with_asu = False
    conf.sym.start_radius = 0.0
    conf.sym.sympair_protein_only = False
    return conf

def test_sym_indep_gp_atom_lig_sym():
    conf = make_symgp_conf_sym()
    conf.sym.symid = 'C2'
    assert conf.sym.contig_is_symmetric
    diffuser = rfd.noisers.get(conf.diffuser)
    _, _, indep, *_ = rfd.inference.data_loader.InferenceDataset(conf, diffuser)[0]
    # ipd.showme(indep)
    # symindep = sym(indep)
    # ipd.showme(symindep)

def make_tiny_indep(reload=False):
    # ligand = None
    # ligand = 'mu2'
    input_pdb = f'{rfd.projdir}/benchmark/input/serine_hydrolase/siteD.pdb'
    # cachefile = '_pytest_tmp_tiny_indep.pickle'
    # if not os.path.exists(cachefile) or reload:
    if True:
        conf = rfd.test_utils.construct_conf(
            [
                f'inference.input_pdb={input_pdb}',
                'inference.contig_as_guidepost=False',
                'inference.guidepost_xyz_as_design=False',
                "contigmap.contigs=['1,A1-2']",
                "contigmap.contig_atoms=\"{'A1':['N','CA','C','O','CB']}\"",
            ],
            inference=True,
        )
    # contigs=['5,A1-2,5,A3-3,5,A4-4,5,A5-5,5'],
    # contig_atoms = dict(
    # A2='CB OG'.split(),
    # A3='CG CD2 ND1 CE1 NE2'.split(),
    # A4='CG OD1 OD2'.split(),
    # A5='CG ND2 OD1'.split(),
    # ),
    # has_termini=[True])

    dataset = rfd.inference.data_loader.InferenceDataset(conf)
    _, _, indep_cond, _, is_diffused, atomizer, contig_map, t_step_input, _ = next(iter(dataset))
    return indep_cond

@pytest.mark.fast
def test_sym_indep():
    indep = make_tiny_indep(reload=False)
    indep.is_gp[:] = False
    # print(indep.seq)
    assert th.all(indep.seq == th.tensor([21, 15, 55, 39, 39, 57, 39]))

    sym = ipd.tests.sym.create_test_sym_manager()
    indep2 = sym(indep, isasym=True)
    assert indep == indep2

    sym = ipd.tests.sym.create_test_sym_manager(symid='c2')

    # ic(indep.chirals)
    indep2 = sym(indep, isasym=True)
    # ic(sym.idx)
    # ic(indep2.seq)
    # th.tensor([21, 15,         55, 39, 39, 57, 39]))
    # indep2.seq: tensor([21, 15, 21, 15, 55, 55, 39, 39, 57, 39, 39, 39, 57, 39])
    assert th.all(indep2.seq == th.tensor([21, 15, 21, 15, 55, 39, 39, 57, 39, 55, 39, 39, 57, 39]))
    assert th.all(indep2.idx == th.tensor([0, 1, 202, 203, 404, 405, 406, 407, 408, 609, 610, 611, 612, 613]))
    assert th.all(indep2.type() == th.tensor([0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))
    assert th.all(indep2.same_chain == th.tensor([
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ]))
    assert th.all(indep.bond_feats == th.tensor([
        [0, 0, 6, 0, 0, 0, 0],
        [0, 0, 0, 0, 6, 0, 0],
        [6, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 6, 0, 1, 0, 2, 0],
        [0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ]))
    assert th.all(indep2.bond_feats == th.tensor([
        [0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0],
        [6, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 6, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    ]))

    # ic(indep.chirals)
    # ic(indep2.chirals)

@pytest.mark.fast
def test_renumber_idx():
    same_chain = th.zeros((10, 10), dtype=bool)
    n = 0
    for l in [1, 2, 3, 2, 2]:
        same_chain[n:n + l, n:n + l] = True
        n += l
    idx = rfd.sym.renumber_idx(same_chain)
    assert th.all(idx == th.tensor([0, 201, 202, 403, 404, 405, 606, 607, 808, 809]))
    same_chain = th.zeros((10, 10), dtype=bool)
    n = 0
    for l in [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]:
        same_chain[n:n + l, n:n + l] = True
        n += l
    # ic(same_chain.to(int))
    idx = rfd.sym.renumber_idx(same_chain)
    assert th.all(idx == th.arange(10) * 201)
    n = 0
    for l in [10]:
        same_chain[n:n + l, n:n + l] = True
        n += l
    # ic(same_chain.to(int))
    idx = rfd.sym.renumber_idx(same_chain)
    assert th.all(idx == th.arange(10))

if __name__ == '__main__':
    main()
