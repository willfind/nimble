"""
Defines a subclass of the Axis object, which serves as the primary
base class for read only axis views of data objects.
"""

from __future__ import absolute_import

from UML.docHelpers import inheritDocstringsFactory
from .points import Points
from .dataHelpers import readOnlyException
from .dataHelpers import exceptionDocstringFactory

exceptionDocstring = exceptionDocstringFactory(Points)

@inheritDocstringsFactory(Points)
class PointsView(Points):
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
    def transform(self, function, points=None, useLog=None):
        readOnlyException('transform')

    ######################################
    # High Level Operations, Disallowed #
    #####################################

    @exceptionDocstring
    def fill(self, match, fill, points=None, returnModified=False, useLog=None,
             **kwarguments):
        readOnlyException('fill')

    @exceptionDocstring
    def normalize(self, subtract=None, divide=None, applyResultTo=None,
                  useLog=None):
        readOnlyException('normalize')

    @exceptionDocstring
    def splitByCollapsingFeatures(self, featuresToCollapse, featureForNames,
                                  featureForValues):
        readOnlyException('splitByCollapsingFeatures')

    @exceptionDocstring
    def combineByExpandingFeatures(self, featureWithFeatureNames,
                                   featureWithValues):
        readOnlyException('combineByExpandingFeatures')
