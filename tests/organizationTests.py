
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

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
