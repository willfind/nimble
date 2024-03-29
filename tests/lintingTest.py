
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
Test that the code meets the minimum linter requirements.
"""
import os

import pytest

import nimble
from lint import analyzeWarnings, sortWarnings, printWarnings

@pytest.mark.slow
def testLinterPasses():
    """Test minimum linter requirements are met."""
    configOptions = [nimble.nimblePath]
    config = os.path.abspath('.pylintrc')
    configOptions.append('--rcfile={0}'.format(config))
    configOptions.append('--output-format=json')

    command = " ".join(configOptions)

    getWarnings = analyzeWarnings(command)
    errors, required, _ = sortWarnings(getWarnings)
    lintMsg = 'LINTER: {0} {1}'
    printWarnings(required, lintMsg.format(len(required), 'REQUIRED CHANGES'))
    printWarnings(errors, lintMsg.format(len(errors), 'ERRORS'))
    assert not errors and not required
