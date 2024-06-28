import os
import pytest

@pytest.fixture(autouse=True)
def change_test_dir(monkeypatch):
    '''force all tests to run in directory that contains this conftest.py file'''
    monkeypatch.chdir(os.path.dirname(__file__))

@pytest.fixture(autouse=True)
def no_cuda_devices():
    '''turn off CUDA for all tests, as currently required'''
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

def pytest_addoption(parser):
    '''
    Add custom command line arguments to pytest.
    '''
    parser.addoption(
        "--show_in_pymol",
        action="store_true",
        default=False,
        help="Show proteins from the dataloader in pymol?"
    )

@pytest.fixture(autouse=True)
def add_custom_option_to_unittest(request):
    '''
    Make values from custom command line arguments available to unittest.TestCase instance
    '''
    if request.instance is not None:
        # Inject the custom option into the unittest.TestCase instance
        setattr(request.instance, 'show_in_pymol', request.config.getoption("--show_in_pymol"))
