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
    # disable faulthandler and warnings by default
    if 'faulthandler' not in args:
        # do not output segfault info from shogun tests
        args.extend(['-p', 'no:faulthandler'])
    if 'warnings' not in args:
        # ignore all warnings unless turned on in test
        args.extend(['-p', 'no:warnings'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pytest.main(args)
    else:
        pytest.main(args)
