from assertpy import assert_that
import pytest
import random
import ipd

@pytest.mark.fast
@pytest.mark.parametrize(
    ['contig', 'valid'],
    [
        (['150'], ['150_150']),
        # (['150-160'], ['156_156']),
        (['11,A163-181,30'], ['11,A163-181,30_11,B163-181,30']),
        (['10_20'], ['10_20_10_20']),
        (['A256-548_80'], ['A256-548_80_B256-548_80']),
    ])
def test_sym_contig(contig, valid):
    sym = ipd.tests.sym.create_test_sym_manager(symid='c2', kind='rf_diffusion')
    sym.opt.contig_is_symmetric = True
    sym.opt.contig_relabel_chains = True
    assert sym.symmetrize_contigs(contig, None, [])[0] == contig
    sym.opt.contig_is_symmetric = False
    state = random.getstate()
    random.seed(0)
    symcontig, _, hastermini = sym.symmetrize_contigs(contig, None, [True])
    assert len(hastermini) == sym.nsub
    random.setstate(state)
    assert assert_that(symcontig).is_equal_to(valid)

# @pytest.mark.fast
# def test_sym_contig_atoms():
# sym = ipd.tests.sym.create_test_sym_manager(symid='c2', kind='rf_diffusion')
# sym.opt.contig_is_symmetric = False
# ctg = ['20,A2-2,0,A3-3,0,A6-6,0']
# atoms = {'A2':'CG,OD1,OD2','A3':'CG,ND2,OD1','A6':'CD,OE1,OE2'}
# symctg, symatom, term = sym.symmetrize_contigs(ctg, atoms, [True])
# print(symctg)
# print(symatom)
# assert symctg == ['20,A2-2,0,A3-3,0,A6-6,0_20,Z2-2,0,Z3-3,0,Z6-6,0']
