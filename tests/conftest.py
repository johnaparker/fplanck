import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return

    skip_slow = pytest.mark.skip(reason='need --runslow option to run')
    skip_not_implemented = pytest.mark.skip(reason='test not yet implemented')

    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_slow)
        elif 'not_implemented' in item.keywords:
            item.add_marker(skip_not_implemented)
