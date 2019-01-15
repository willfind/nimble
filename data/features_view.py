"""
Defines a subclass of the Axis object, which serves as the primary
base class for read only axis views of data objects.
"""
from __future__ import absolute_import

from .features import Features
from .dataHelpers import readOnlyException
from .dataHelpers import exceptionDocstringFactory

exceptionDocstring = exceptionDocstringFactory(Features)

class FeaturesView(Features):
    """
    Class defining read only view objects, which have the same api as a
    normal Axis object, but disallow all methods which could change the
    data.

    Parameters
    ----------
    source : UML data object
        The UML object that this is a view into.
    """

    ####################################
    # Low Level Operations, Disallowed #
    ####################################

    @exceptionDocstring
    def setName(self, oldIdentifier, newName):
        readOnlyException('setName')

    @exceptionDocstring
    def setNames(self, assignments=None):
        readOnlyException('setNames')

    #####################################
    # Structural Operations, Disallowed #
    #####################################

    @exceptionDocstring
    def extract(self, toExtract=None, start=None, end=None, number=None,
                randomize=False):
        readOnlyException('extract')

    @exceptionDocstring
    def delete(self, toDelete=None, start=None, end=None, number=None,
               randomize=False):
        readOnlyException('delete')

    @exceptionDocstring
    def retain(self, toRetain=None, start=None, end=None, number=None,
               randomize=False):
        readOnlyException('retain')

    @exceptionDocstring
    def add(self, toAdd, insertBefore=None):
        readOnlyException('add')

    @exceptionDocstring
    def shuffle(self):
        readOnlyException('shuffle')

    @exceptionDocstring
    def sort(self, sortBy=None, sortHelper=None):
        readOnlyException('sort')

    @exceptionDocstring
    def transform(self, function, features=None):
        readOnlyException('transform')

    ######################################
    # High Level Operations, Disallowed #
    #####################################

    @exceptionDocstring
    def fill(self, match, fill, arguments=None, features=None,
             returnModified=False):
        readOnlyException('fill')

    @exceptionDocstring
    def normalize(self, subtract=None, divide=None, applyResultTo=None):
        readOnlyException('normalize')

    @exceptionDocstring
    def splitByParsing(self, feature, rule, resultingNames):
        readOnlyException('splitByParsing')
