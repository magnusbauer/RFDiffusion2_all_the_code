import pytest
import ipd
import rf_diffusion as rfd

def main():
    test_sym_rfi_adapt()

@pytest.mark.skip
def test_sym_rfi_adapt():
    sym = ipd.tests.sym.create_test_sym_manager(symid='c4', kind='rf_diffusion')
    rfi = ipd.dev.load(f"{rfd.projdir}/test_data/pkl/rfi.pickle")
    L = rfi.xyz.shape[1]
    # ic(L, rfi.xyz.shape)
    sym.idx = [(L, 0, L)]
    # ic(sym.masym, sym.masu)
    # asym = sym.asym(rfi)

if __name__ == '__main__':
    main()
