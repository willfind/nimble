
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

#!/usr/bin/python

"""
Script acting as the canonical means to run the test suite for the entirety of
nimble. Run as main to execute.
"""

import warnings
import sys

import pytest

if __name__ == '__main__':
    # any args passed to this script will be passed down into pytest
    args = sys.argv
    # if no path provided, discover tests in nimble/ and tests/ directories
    if len(args) == 1:
        args.append('nimble')
        args.append('tests')
    elif args[1].startswith('-'):
        args.insert(1, 'tests')
        args.insert(1, 'nimble')
    # always run any doctests
    args.append('--doctest-modules')
    # disable warnings by default
    if 'warnings' not in args:
        # ignore all warnings unless turned on in test
        args.extend(['-p', 'no:warnings'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pytest.main(args)
    else:
        pytest.main(args)
