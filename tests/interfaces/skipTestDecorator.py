"""
Defines a decorator object to help skip tests of missing interfaces
"""

from unittest.case import SkipTest
from functools import wraps

import UML as nimble
from UML.exceptions import InvalidArgumentValue

class SkipMissing(object):
    """
    Decorator object which raises SkipTest if the provided interface is missing

    Whether an interface is present is defined by nimble.helpers.findBestInterface.
    This object will be instantiated once in every interface test file, with
    the appropriate interface name.
    """

    def __init__(self, interfaceName):
        self.interfaceName = interfaceName

        self.missing = False
        try:
            nimble.helpers.findBestInterface(interfaceName)
        except InvalidArgumentValue:
            self.missing = True

    def __call__(self, toWrap):
        @wraps(toWrap)
        def newFunc(*args, **kwargs):
            if self.missing:
                raise SkipTest
            else:
                return toWrap(*args, **kwargs)

        return newFunc
