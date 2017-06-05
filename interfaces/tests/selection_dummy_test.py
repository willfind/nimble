"""
A test which should never be run; demonstrates that UML/runTests.py will correctly avoid
running tests in files associated with interfaces that are not available in the
current UML session (ie ones whose underlying packages are missing).

"""


def test_runTestsSelectionLogic():
    """ runTests incorrectly selecting tests for uncallable interfaces """
    assert False
