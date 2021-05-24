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
    # search current directory for tests if none specified and run doctests
    if len(args) == 1:
        # will find doctests in documentation so ignore
        args.extend(['.', '--ignore', 'documentation'])
    # test doctests
    args.append('--doctest-modules')
    # disable faulthandler and warnings by default
    if 'faulthandler' not in args:
        # do not dump segfaults from shogun tests
        args.extend(['-p', 'no:faulthandler'])
    if 'warnings' not in args:
        # ignore all warnings unless turned on in test
        args.extend(['-p', 'no:warnings'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pytest.main(args)
    else:
        pytest.main(args)
