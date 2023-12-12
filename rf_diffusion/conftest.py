import pytest

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
