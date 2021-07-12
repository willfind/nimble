"""
Test that the nimble package is organized as expected.
"""
from types import ModuleType

import nimble

def test__all__():
    assert hasattr(nimble, '__all__')
    nimbleAll = nimble.__all__
    # update this list if a new submodule is added or existing one is removed
    expectedSubmodules = ['calculate', 'exceptions', 'fill', 'learners',
                          'match', 'random']
    submodules = [attr for attr in nimbleAll
                  if isinstance(getattr(nimble, attr), ModuleType)]
    assert sorted(expectedSubmodules) == sorted(submodules)

    # core should not be included in __all__
    assert 'core' not in nimbleAll

    for name in submodules:
        mod = getattr(nimble, name)
        assert hasattr(mod, '__all__')
