"""
Defines a subclass of the Axis object, which serves as the primary
base class for read only axis views of data objects.
"""

from __future__ import absolute_import

import nimble
from nimble.utility import inheritDocstringsFactory
from .features import Features
from .dataHelpers import readOnlyException
from .dataHelpers import exceptionDocstringFactory

exceptionDocstring = exceptionDocstringFactory(Features)

@inheritDocstringsFactory(Features)
class FeaturesView(Features):
    """
    Class defining read only view objects, which have the same api as a
    normal Axis object, but disallow all methods which could change the
    data.

    Parameters
    ----------
    source : nimble data object
        The nimble object that this is a view into.
    """

    ####################################
    # Low Level Operations, Disallowed #
    ####################################

    @exceptionDocstring
    def setName(self, oldIdentifier, newName, useLog=None):
        readOnlyException('setName')

    @exceptionDocstring
    def setNames(self, assignments=None, useLog=None):
        readOnlyException('setNames')

    #####################################
    # Structural Operations, Disallowed #
    #####################################

    @exceptionDocstring
    def extract(self, toExtract=None, start=None, end=None, number=None,
                randomize=False, useLog=None):
        readOnlyException('extract')

    @exceptionDocstring
    def delete(self, toDelete=None, start=None, end=None, number=None,
               randomize=False, useLog=None):
        readOnlyException('delete')

    @exceptionDocstring
    def retain(self, toRetain=None, start=None, end=None, number=None,
               randomize=False, useLog=None):
        readOnlyException('retain')

    @exceptionDocstring
    def add(self, toAdd, insertBefore=None, useLog=None):
        readOnlyException('add')

    @exceptionDocstring
    def shuffle(self, useLog=None):
        readOnlyException('shuffle')

    @exceptionDocstring
    def sort(self, sortBy=None, sortHelper=None, useLog=None):
        readOnlyException('sort')

    @exceptionDocstring
    def transform(self, function, features=None, useLog=None):
        readOnlyException('transform')

    ######################################
    # High Level Operations, Disallowed #
    #####################################

    @exceptionDocstring
    def fill(self, match, fill, features=None, returnModified=False,
             useLog=None, **kwarguments):
        readOnlyException('fill')

    @exceptionDocstring
    def normalize(self, subtract=None, divide=None, applyResultTo=None,
                  useLog=None):
        readOnlyException('normalize')

    @exceptionDocstring
    def splitByParsing(self, feature, rule, resultingNames, useLog=None):
        readOnlyException('splitByParsing')
