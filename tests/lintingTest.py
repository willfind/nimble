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
