"""
Defines a subclass of the Axis object, which serves as the primary
base class for read only axis views of data objects.
"""

import nimble
from nimble.utility import inheritDocstringsFactory
from .features import Features
from .dataHelpers import readOnlyException
from .dataHelpers import exceptionDocstringFactory

exceptionDocstring = exceptionDocstringFactory(Features)

@inheritDocstringsFactory(Features)
class FeaturesView(Features):
    """
    Class limiting the Features class to read-only by disallowing
    methods which could change the data.

    Parameters
    ----------
    base : BaseView
        The BaseView instance that will be queried.
    """

    ####################################
    # Low Level Operations, Disallowed #
    ####################################

    @exceptionDocstring
    def setName(self, oldIdentifier, newName, useLog=None):
        readOnlyException('setName')

    @exceptionDocstring
    def setNames(self, assignments, useLog=None):
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
    def insert(self, insertBefore, toInsert, useLog=None):
        readOnlyException('insert')

    @exceptionDocstring
    def append(self, toAppend, useLog=None):
        readOnlyException('append')

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
    def fillMatching(self, fillWith, matchingElements, features=None,
                     returnModified=False, useLog=None, **kwarguments):
        readOnlyException('fill')

    @exceptionDocstring
    def normalize(self, subtract=None, divide=None, applyResultTo=None,
                  useLog=None):
        readOnlyException('normalize')

    @exceptionDocstring
    def splitByParsing(self, feature, rule, resultingNames, useLog=None):
        readOnlyException('splitByParsing')
