
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
Defines subclasses that serve primary as the base classes for read only
views Base, Axis, Points and Features objects.
"""
from abc import ABCMeta
from contextlib import contextmanager

from nimble._utility import inheritDocstringsFactory
from nimble.exceptions import ImproperObjectAction
from .base import Base
from .points import Points
from .features import Features
from ._dataHelpers import readOnlyException
from ._dataHelpers import exceptionDocstringFactory

baseExceptionDoc = exceptionDocstringFactory(Base)
pointsExceptionDoc = exceptionDocstringFactory(Points)
featuresExceptionDoc = exceptionDocstringFactory(Features)

@inheritDocstringsFactory(Base)
class BaseView(Base, metaclass=ABCMeta):
    """
    Class limiting the Base class to read-only by disallowing methods
    which could change the data.

    Parameters
    ----------
    source : nimble Base object
        The nimble object that this is a view into.
    pointStart : int
        The inclusive index of the first point this view will have
        access to.
    pointEnd : int
        The EXCLUSIVE index defining the last point this view will
        have access to. This internal representation cannot match
        the style of the factory method (in which both start and end
        are inclusive) because we must be able to define empty
        ranges by having start = end
    featureStart : int
        The inclusive index of the first feature this view will have
        access to.
    featureEnd : int
        The EXCLUSIVE index defining the last feature this view will
        have access to. This internal representation cannot match
        the style of the factory method (in which both start and end
        are inclusive) because we must be able to define empty
        ranges by having start = end
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """
    def __init__(self, source, pointStart, pointEnd, featureStart, featureEnd,
                 **kwds):
        self._source = source
        self._pStart = pointStart
        self._pEnd = pointEnd
        self._fStart = featureStart
        self._fEnd = featureEnd
        if len(source._dims) > 2:
            if self._fStart != 0 or self._fEnd != len(source.features):
                msg = "feature limited views are not allowed for data with "
                msg += "more than two dimensions."
                raise ImproperObjectAction(msg)
        if source.points.names is not None:
            nameSlice = slice(self._pStart, self._pEnd)
            kwds['pointNames'] = source.points.namesInverse[nameSlice]
        if source.features.names is not None:
            nameSlice = slice(self._fStart, self._fEnd)
            kwds['featureNames'] = source.features.namesInverse[nameSlice]
        kwds['reuseData'] = True

        super().__init__(**kwds)

    @property
    def absolutePath(self):
        """
        The path to the file this data originated from, in absolute
        form.
        """
        return self._source._absPath

    @property
    def relativePath(self):
        """
        The path to the file this data originated from, in relative
        form.
        """
        return self._source._relPath

    # TODO: retType

    ############################
    # Reimplemented Operations #
    ############################

    def _view_backend(self, pointStart=None, pointEnd=None, featureStart=None,
                      featureEnd=None, dropDimension=False):

        # -1 because _pEnd and _fEnd are exclusive indices,
        # but view takes inclusive

        if pointStart is None:
            psAdj = None if len(self._source.points) == 0 else self._pStart
        else:
            psIndex = self._source.points.getIndex(pointStart)
            psAdj = psIndex + self._pStart

        if pointEnd is None:
            peAdj = None if len(self._source.points) == 0 else self._pEnd - 1
        else:
            peIndex = self._source.points.getIndex(pointEnd)
            peAdj = peIndex + self._pStart

        if featureStart is None:
            fsAdj = None if len(self._source.features) == 0 else self._fStart
        else:
            fsIndex = self._source.features.getIndex(featureStart)
            fsAdj = fsIndex + self._fStart

        if featureEnd is None:
            feAdj = None if len(self._source.features) == 0 else self._fEnd - 1
        else:
            feIndex = self._source.features.getIndex(featureEnd)
            feAdj = feIndex + self._fStart

        return self._source._view_backend(psAdj, peAdj, fsAdj, feAdj,
                                          dropDimension)

    @contextmanager
    def _treatAs2D(self):
        if len(self._dims) > 2:
            savedShape = self._dims
            savedSource = self._source._dims
            self._dims = [len(self.points), len(self.features)]
            self._source._dims = [len(self._source.points),
                                   len(self._source.features)]
            try:
                yield self
            finally:
                self._dims = savedShape
                self._source._dims = savedSource
        else:
            yield self

    # pylint: disable=unused-argument
    ###########################
    # Higher Order Operations #
    ###########################

    @baseExceptionDoc
    def replaceFeatureWithBinaryFeatures(self, featureToReplace, *,
                                         useLog=None):
        readOnlyException("replaceFeatureWithBinaryFeatures")

    @baseExceptionDoc
    def transformFeatureToIntegers(self, featureToConvert, *, useLog=None):
        readOnlyException("transformFeatureToIntegers")

    @baseExceptionDoc
    def _referenceFrom(self, other, **kwargs):
        readOnlyException('_referenceFrom')

    ########################################
    ########################################
    ###   Functions related to logging   ###
    ########################################
    ########################################


    ###############################################################
    ###############################################################
    ###   Subclass implemented information querying functions   ###
    ###############################################################
    ###############################################################


    ##################################################################
    ##################################################################
    ###   Subclass implemented structural manipulation functions   ###
    ##################################################################
    ##################################################################

    @baseExceptionDoc
    def transformElements(self, toTransform, points=None, features=None,
                          preserveZeros=False, skipNoneReturnValues=False,
                          *, useLog=None):
        readOnlyException("transform")

    @baseExceptionDoc
    def transpose(self, *, useLog=None):
        readOnlyException("transpose")

    @baseExceptionDoc
    def replaceRectangle(self, replaceWith, pointStart, featureStart,
                         pointEnd=None, featureEnd=None, *, useLog=None):
        readOnlyException("replaceRectangle")

    @baseExceptionDoc
    def flatten(self, order='point', *, useLog=None):
        readOnlyException("flatten")

    @baseExceptionDoc
    def unflatten(self, dataDimensions, order='point', *, useLog=None):
        readOnlyException("unflatten")

    @baseExceptionDoc
    def merge(self, other, point='strict', feature='union', onFeature=None,
              force=False, *, useLog=None):
        readOnlyException('merge')

    ###############################################################
    ###############################################################
    ###   Subclass implemented numerical operation functions    ###
    ###############################################################
    ###############################################################

    @baseExceptionDoc
    def __imatmul__(self, other):
        readOnlyException("__imatmul__")

    @baseExceptionDoc
    def __imul__(self, other):
        readOnlyException("__imul__")

    @baseExceptionDoc
    def __iadd__(self, other):
        readOnlyException("__iadd__")

    @baseExceptionDoc
    def __isub__(self, other):
        readOnlyException("__isub__")

    @baseExceptionDoc
    def __itruediv__(self, other):
        readOnlyException("__itruediv__")

    @baseExceptionDoc
    def __ifloordiv__(self, other):
        readOnlyException("__ifloordiv__")

    @baseExceptionDoc
    def __imod__(self, other):
        readOnlyException("__imod__")

    @baseExceptionDoc
    def __ipow__(self, other):
        readOnlyException("__ipow__")


@inheritDocstringsFactory(Points)
class PointsView(Points, metaclass=ABCMeta):
    """
    Class limiting the Points class to read-only by disallowing methods
    which could change the data.

    Parameters
    ----------
    base : BaseView
        The BaseView instance that will be queried.
    """
    # pylint: disable=unused-argument

    ####################################
    # Low Level Operations, Disallowed #
    ####################################

    @pointsExceptionDoc
    def setNames(self, assignments, oldIdentifiers=None, *, useLog=None):
        readOnlyException('setNames')

    #####################################
    # Structural Operations, Disallowed #
    #####################################

    @pointsExceptionDoc
    def extract(self, toExtract=None, start=None, end=None, number=None,
                randomize=False, features=None, *, useLog=None):
        readOnlyException('extract')

    @pointsExceptionDoc
    def delete(self, toDelete=None, start=None, end=None, number=None,
               randomize=False, features=None, *, useLog=None):
        readOnlyException('delete')

    @pointsExceptionDoc
    def retain(self, toRetain=None, start=None, end=None, number=None,
               randomize=False, features=None, *, useLog=None):
        readOnlyException('retain')

    @pointsExceptionDoc
    def insert(self, insertBefore, toInsert, *, useLog=None):
        readOnlyException('insert')

    @pointsExceptionDoc
    def append(self, toAppend, *, useLog=None):
        readOnlyException('append')

    @pointsExceptionDoc
    def permute(self, order=None, *, useLog=None):
        readOnlyException('permute')

    @pointsExceptionDoc
    def sort(self, by=None, reverse=False, *, useLog=None):
        readOnlyException('sort')

    @pointsExceptionDoc
    def transform(self, function, points=None, *, useLog=None):
        readOnlyException('transform')

    ######################################
    # High Level Operations, Disallowed #
    #####################################

    @pointsExceptionDoc
    def fillMatching(self, fillWith, matchingElements, points=None,
                     *, useLog=None, **kwarguments):
        readOnlyException('fill')

    @pointsExceptionDoc
    def splitByCollapsingFeatures(self, featuresToCollapse, featureForNames,
                                  featureForValues, *, useLog=None):
        readOnlyException('splitByCollapsingFeatures')

    @pointsExceptionDoc
    def combineByExpandingFeatures(self, featureWithFeatureNames,
                                   featuresWithValues,
                                   modifyDuplicateFeatureNames=False,
                                   *, useLog=None):
        readOnlyException('combineByExpandingFeatures')

    @pointsExceptionDoc
    def replace(self, data, points=None, *, useLog=None, **dataKwds):
        readOnlyException('replace')

@inheritDocstringsFactory(Features)
class FeaturesView(Features, metaclass=ABCMeta):
    """
    Class limiting the Features class to read-only by disallowing
    methods which could change the data.

    Parameters
    ----------
    base : BaseView
        The BaseView instance that will be queried.
    """
    # pylint: disable=unused-argument

    ####################################
    # Low Level Operations, Disallowed #
    ####################################


    @featuresExceptionDoc
    def setNames(self, assignments, oldIdentifiers=None, *, useLog=None):
        readOnlyException('setNames')

    #####################################
    # Structural Operations, Disallowed #
    #####################################

    @featuresExceptionDoc
    def extract(self, toExtract=None, start=None, end=None, number=None,
                randomize=False, points=None, *, useLog=None):
        readOnlyException('extract')

    @featuresExceptionDoc
    def delete(self, toDelete=None, start=None, end=None, number=None,
               randomize=False, points=None, *, useLog=None):
        readOnlyException('delete')

    @featuresExceptionDoc
    def retain(self, toRetain=None, start=None, end=None, number=None,
               randomize=False, points=None, *, useLog=None):
        readOnlyException('retain')

    @featuresExceptionDoc
    def insert(self, insertBefore, toInsert, *, useLog=None):
        readOnlyException('insert')

    @featuresExceptionDoc
    def append(self, toAppend, *, useLog=None):
        readOnlyException('append')

    @featuresExceptionDoc
    def permute(self, order=None, *, useLog=None):
        readOnlyException('permute')

    @featuresExceptionDoc
    def sort(self, by=None, reverse=False, *, useLog=None):
        readOnlyException('sort')

    @featuresExceptionDoc
    def transform(self, function, features=None, *, useLog=None):
        readOnlyException('transform')

    ######################################
    # High Level Operations, Disallowed #
    #####################################

    @featuresExceptionDoc
    def fillMatching(self, fillWith, matchingElements, features=None,
                     *, useLog=None, **kwarguments):
        readOnlyException('fill')

    @featuresExceptionDoc
    def normalize(self, function, applyResultTo=None, features=None,
                  *, useLog=None):
        readOnlyException('normalize')

    @featuresExceptionDoc
    def splitByParsing(self, feature, rule, resultingNames, *, useLog=None):
        readOnlyException('splitByParsing')

    @featuresExceptionDoc
    def replace(self, data, features=None, *, useLog=None, **dataKwds):
        readOnlyException('replace')
