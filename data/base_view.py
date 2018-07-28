"""
Defines a subclass of the base data object, which serves as the primary
base class for read only views of data objects.

"""

from __future__ import division
from __future__ import absolute_import
from .base import Base
from UML.exceptions import ImproperActionException

import copy


class BaseView(Base):
    """
    Class defining read only view objects, which have the same api as a
    normal data object, but disallow all methods which could change the
    data.

    """

    def __init__(self, source, pointStart, pointEnd, featureStart, featureEnd,
                 **kwds):
        """
        Initializes the object which overides all of the funcitonality in
        UML.data.Base to either handle the provided access limits or throw
        exceptions for inappropriate operations.

        source: the UML object that this is a view into.

        pointStart: the inclusive index of the first point this view will have
        access to.

        pointEnd: the EXCLUSIVE index defining the last point this view will
        have access to. This internal representation cannot match the style
        of the factory method (in which both start and end are inclusive)
        because we must be able to define empty ranges by having start = end

        featureStart: the inclusive index of the first feature this view will
        have access to.

        featureEnd: the EXCLUSIVE index defining the last feature this view
        will have access to. This internal representation cannot match the
        style of the factory method (in which both start and end are inclusive)
        because we must be able to define empty ranges by having start = end

        kwds: included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.

        """
        self._source = source
        self._pStart = pointStart
        self._pEnd = pointEnd
        self._fStart = featureStart
        self._fEnd = featureEnd
        #		kwds['name'] = self._source.name
        super(BaseView, self).__init__(**kwds)

    # redifinition from Base, except without the setter, using source
    # object's attributes
    def _getObjName(self):
        return self._name

    name = property(_getObjName, doc="A name to be displayed when printing or logging this object")

    # redifinition from Base, using source object's attributes
    def _getAbsPath(self):
        return self._source._absPath

    absolutePath = property(_getAbsPath, doc="The path to the file this data originated from, in absolute form")

    # redifinition from Base, using source object's attributes
    def _getRelPath(self):
        return self._source._relPath

    relativePath = property(_getRelPath, doc="The path to the file this data originated from, in relative form")

    def _pointNamesCreated(self):
        """
        Returns True if point default names have been created/assigned
        to the object.
        If the object does not have points it returns True.
        """
        if self._source.pointNamesInverse is None:
            return False
        else:
            return True

    def _featureNamesCreated(self):
        """
        Returns True if feature default names have been created/assigned
        to the object.
        If the object does not have features it returns True.
        """
        if self._source.featureNamesInverse is None:
            return False
        else:
            return True

    def _getData(self):
        return self._source.data

    # TODO: retType


    ############################
    # Reimplemented Operations #
    ############################

    def getPointNames(self):
        """Returns a list containing all point names, where their index
        in the list is the same as the index of the point they correspond
        to.

        """
        ret = self._source.getPointNames()
        ret = ret[self._pStart:self._pEnd]

        return ret

    def getFeatureNames(self):
        """Returns a list containing all feature names, where their index
        in the list is the same as the index of the feature they
        correspond to.

        """
        ret = self._source.getFeatureNames()
        ret = ret[self._fStart:self._fEnd]

        return ret

    def getPointName(self, index):
        corrected = index + self._pStart
        return self._source.getPointName(corrected)

    def getPointIndex(self, name):
        possible = self._source.getPointIndex(name)
        if possible >= self._pStart and possible < self._pEnd:
            return possible - self._pStart
        else:
            raise KeyError()

    def getFeatureName(self, index):
        corrected = index + self._fStart
        return self._source.getFeatureName(corrected)

    def getFeatureIndex(self, name):
        possible = self._source.getFeatureIndex(name)
        if possible >= self._fStart and possible < self._fEnd:
            return possible - self._fStart
        else:
            raise KeyError()

    def _copyNames(self, CopyObj):

        if self._pointNamesCreated():
            CopyObj.pointNamesInverse = self.getPointNames()
            CopyObj.pointNames = copy.copy(self._source.pointNames)
            # if CopyObj.getTypeString() == 'DataFrame':
            #     CopyObj.data.index = self.getPointNames()
        else:
            CopyObj.pointNamesInverse = None
            CopyObj.pointNames = None

        if self._featureNamesCreated():
            CopyObj.featureNamesInverse = self.getFeatureNames()
            CopyObj.featureNames = copy.copy(self._source.featureNames)
            # if CopyObj.getTypeString() == 'DataFrame':
            #     CopyObj.data.columns = self.getFeatureNames()
        else:
            CopyObj.featureNamesInverse = None
            CopyObj.featureNames = None

        CopyObj._nextDefaultValueFeature = self._source._nextDefaultValueFeature
        CopyObj._nextDefaultValuePoint = self._source._nextDefaultValuePoint

        if self.points != self._source.points:
            if self._pStart != 0:
                CopyObj.pointNames = {}
                for idx, name in enumerate(CopyObj.pointNamesInverse):
                    CopyObj.pointNames[name] = idx
            else:
                for name in self._source.pointNamesInverse[self._pEnd:self._source.points + 1]:
                    del CopyObj.pointNames[name]

        if self.features != self._source.features:
            if self._fStart != 0:
                CopyObj.featureNames = {}
                for idx, name in enumerate(CopyObj.featureNamesInverse):
                    CopyObj.featureNames[name] = idx
            else:
                for name in self._source.featureNamesInverse[self._fEnd:self._source.features + 1]:
                    del CopyObj.featureNames[name]


    def view(self, pointStart=None, pointEnd=None, featureStart=None,
             featureEnd=None):

        # -1 because _pEnd and _fEnd are exclusive indices, but view takes inclusive

        if pointStart is None:
            psAdj = None if self._source.points == 0 else self._pStart
        else:
            psIndex = self._source._getIndex(pointStart, 'point')
            psAdj = psIndex + self._pStart

        if pointEnd is None:
            peAdj = None if self._source.points == 0 else self._pEnd - 1
        else:
            peIndex = self._source._getIndex(pointEnd, 'point')
            peAdj = peIndex + self._pStart

        if featureStart is None:
            fsAdj = None if self._source.features == 0 else self._fStart
        else:
            fsIndex = self._source._getIndex(featureStart, 'feature')
            fsAdj = fsIndex + self._fStart

        if featureEnd is None:
            feAdj = None if self._source.features == 0 else self._fEnd - 1
        else:
            feIndex = self._source._getIndex(featureEnd, 'feature')
            feAdj = feIndex + self._fStart

        return self._source.view(psAdj, peAdj, fsAdj, feAdj)


    ####################################
    # Low Level Operations, Disallowed #
    ####################################

    def setPointName(self, oldIdentifier, newName):
        """
        Changes the pointName specified by previous to the supplied input name.

        oldIdentifier must be a non None string or integer, specifying either a current pointName
        or the index of a current pointName. newName may be either a string not currently
        in the pointName set, or None for an default pointName. newName cannot begin with the
        default prefix.

        None is always returned.

        """
        self._readOnlyException("setPointName")

    def setFeatureName(self, oldIdentifier, newName):
        """
        Changes the featureName specified by previous to the supplied input name.

        oldIdentifier must be a non None string or integer, specifying either a current featureName
        or the index of a current featureName. newName may be either a string not currently
        in the featureName set, or None for an default featureName. newName cannot begin with the
        default prefix.

        None is always returned.

        """
        self._readOnlyException("setFeatureName")


    def setPointNames(self, assignments=None):
        """
        Rename all of the point names of this object according to the values
        specified by the assignments parameter. If given a list, then we use
        the mapping between names and array indices to define the point
        names. If given a dict, then that mapping will be used to define the
        point names. If assignments is None, then all point names will be
        given new default values. If assignment is an unexpected type, the names
        are not strings, the names are not unique, or point indices are missing,
        then an ArgumentException will be raised. None is always returned.

        """
        self._readOnlyException("setPointNames")

    def setFeatureNames(self, assignments=None):
        """
        Rename all of the feature names of this object according to the values
        specified by the assignments parameter. If given a list, then we use
        the mapping between names and array indices to define the feature
        names. If given a dict, then that mapping will be used to define the
        feature names. If assignments is None, then all feature names will be
        given new default values. If assignment is an unexpected type, the names
        are not strings, the names are not unique, or feature indices are missing,
        then an ArgumentException will be raised. None is always returned.

        """
        self._readOnlyException("setFeatureNames")


    ###########################
    # Higher Order Operations #
    ###########################

    def dropFeaturesContainingType(self, typeToDrop):
        """
        Modify this object so that it no longer contains features which have the specified
        type as values. None is always returned.

        """
        self._readOnlyException("dropFeaturesContainingType")

    def replaceFeatureWithBinaryFeatures(self, featureToReplace):
        """
        Modify this object so that the chosen feature is removed, and binary valued
        features are added, one for each possible value seen in the original feature.
        None is always returned.

        """
        self._readOnlyException("replaceFeatureWithBinaryFeatures")

    def transformFeatureToIntegers(self, featureToConvert):
        """
        Modify this object so that the chosen feature in removed, and a new integer
        valued feature is added with values 0 to n-1, one for each of n values present
        in the original feature. None is always returned.

        """
        self._readOnlyException("transformFeatureToIntegers")

    def extractPointsByCoinToss(self, extractionProbability):
        """
        Return a new object containing a randomly selected sample of points
        from this object, where a random experiment is performed for each
        point, with the chance of selection equal to the extractionProbabilty
        parameter. Those selected values are also removed from this object.

        """
        self._readOnlyException("extractPointsByCoinToss")

    def shufflePoints(self):
        """
        Permute the indexing of the points so they are in a random order. Note: this relies on
        python's random.shuffle() so may not be sufficiently random for large number of points.
        See shuffle()'s documentation. None is always returned.

        """
        self._readOnlyException("shufflePoints")

    def shuffleFeatures(self):
        """
        Permute the indexing of the features so they are in a random order. Note: this relies on
        python's random.shuffle() so may not be sufficiently random for large number of features.
        See shuffle()'s documentation. None is always returned.

        """
        self._readOnlyException("shuffleFeatures")

    def normalizePoints(self, subtract=None, divide=None, applyResultTo=None):
        """
        Modify all points in this object according to the given
        operations.

        applyResultTo: default None, if a UML object is given, then
        perform the same operations to it as are applied to the calling
        object. However, if a statistical method is specified as subtract
        or divide, then	concrete values are first calculated only from
        querying the calling object, and the operation is performed on
        applyResultTo using the results; as if a UML object was given
        for the subtract or divide arguments.

        subtract: what should be subtracted from data. May be a fixed
        numerical value, a string defining a statistical function (all of
        the same ones callable though pointStatistics), or a UML data
        object. If a vector shaped object is given, then the value
        associated with each point will be subtracted from all values of
        that point. Otherwise, the values in the object are used for
        elementwise subtraction. Default None - equivalent to subtracting
        0.

        divide: defines the denominator for dividing the data. May be a
        fixed numerical value, a string defining a statistical function
        (all of the same ones callable though pointStatistics), or a UML
        data object. If a vector shaped object is given, then the value
        associated with each point will be used in division of all values
        for that point. Otherwise, the values in the object are used for
        elementwise division. Default None - equivalent to dividing by
        1.

        Returns None while having affected the data of the calling
        object and applyResultTo (if non-None).

        """
        self._readOnlyException("normalizePoints")

    def normalizeFeatures(self, subtract=None, divide=None, applyResultTo=None):
        """
        Modify all features in this object according to the given
        operations.

        applyResultTo: default None, if a UML object is given, then
        perform the same operations to it as are applied to the calling
        object. However, if a statistical method is specified as subtract
        or divide, then	concrete values are first calculated only from
        querying the calling object, and the operation is performed on
        applyResultTo using the results; as if a UML object was given
        for the subtract or divide arguments.

        subtract: what should be subtracted from data. May be a fixed
        numerical value, a string defining a statistical function (all of
        the same ones callable though featureStatistics), or a UML data
        object. If a vector shaped object is given, then the value
        associated with each feature will be subtracted from all values of
        that feature. Otherwise, the values in the object are used for
        elementwise subtraction. Default None - equivalent to subtracting
        0.

        divide: defines the denominator for dividing the data. May be a
        fixed numerical value, a string defining a statistical function
        (all of the same ones callable though featureStatistics), or a UML
        data object. If a vector shaped object is given, then the value
        associated with each feature will be used in division of all values
        for that feature. Otherwise, the values in the object are used for
        elementwise division. Default None - equivalent to dividing by
        1.

        Returns None while having affected the data of the calling
        object and applyResultTo (if non-None).

        """
        self._readOnlyException("normalizeFeatures")


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

    def transpose(self):
        """
        Function to transpose the data, ie invert the feature and point indices of the data.
        The point and feature names are also swapped. None is always returned.

        """
        self._readOnlyException("transpose")

    def appendPoints(self, toAppend):
        """
        Append the points from the toAppend object to the bottom of the features in this object.

        toAppend cannot be None, and must be a kind of data representation object with the same
        number of features as the calling object. None is always returned.

        """
        self._readOnlyException("appendPoints")

    def appendFeatures(self, toAppend):
        """
        Append the features from the toAppend object to right ends of the points in this object

        toAppend cannot be None, must be a kind of data representation object with the same
        number of points as the calling object, and must not share any feature names with the calling
        object. None is always returned.

        """
        self._readOnlyException("appendFeatures")

    def sortPoints(self, sortBy=None, sortHelper=None):
        """
        Modify this object so that the points are sorted in place, where sortBy may
        indicate the feature to sort by or None if the entire point is to be taken as a key,
        sortHelper may either be comparator, a scoring function, or None to indicate the natural
        ordering. None is always returned.
        """
        self._readOnlyException("sortPoints")

    def sortFeatures(self, sortBy=None, sortHelper=None):
        """
        Modify this object so that the features are sorted in place, where sortBy may
        indicate the feature to sort by or None if the entire point is to be taken as a key,
        sortHelper may either be comparator, a scoring function, or None to indicate the natural
        ordering.  None is always returned.

        """
        self._readOnlyException("sortFeatures")

    def extractPoints(self, toExtract=None, start=None, end=None, number=None, randomize=False):
        """
        Modify this object, removing those points that are specified by the input, and returning
        an object containing those removed points.

        toExtract may be a single identifier (name and/or index), a list of identifiers,
        a function that when given a point will return True if it is to be extracted, or a
        filter function, as a string, containing a comparison operator between a feature name
        and a value (i.e 'feat1<10')

        number is the quantity of points that are to be extracted, the default None means
        unrestricted extraction.

        start and end are parameters indicating range based extraction: if range based
        extraction is employed, toExtract must be None, and vice versa. If only one of start
        and end are non-None, the other defaults to 0 and self.points respectably.

        randomize indicates whether random sampling is to be used in conjunction with the number
        parameter, if randomize is False, the chosen points are determined by point order,
        otherwise it is uniform random across the space of possible removals.

        """
        self._readOnlyException("extractPoints")

    def extractFeatures(self, toExtract=None, start=None, end=None, number=None, randomize=False):
        """
        Modify this object, removing those features that are specified by the input, and returning
        an object containing those removed features.

        toExtract may be a single identifier (name and/or index), a list of identifiers,
        a function that when given a feature will return True if it is to be extracted, or a
        filter function, as a string, containing a comparison operator between a point name
        and a value (i.e 'point1<10')

        number is the quantity of features that are to be extracted, the default None means
        unrestricted extraction.

        start and end are parameters indicating range based extraction: if range based
        extraction is employed, toExtract must be None, and vice versa. If only one of start
        and end are non-None, the other defaults to 0 and self.features respectably.

        randomize indicates whether random sampling is to be used in conjunction with the number
        parameter, if randomize is False, the chosen features are determined by feature order,
        otherwise it is uniform random across the space of possible removals.

        """
        self._readOnlyException("extractFeatures")

    def deletePoints(self, toDelete=None, start=None, end=None, number=None, randomize=False):
        """
        Modify this object, removing those points that are specified by the input.

        toDelete may be a single identifier (name and/or index), a list of identifiers,
        a function that when given a point will return True if it is to be deleted, or a
        filter function, as a string, containing a comparison operator between a feature name
        and a value (i.e 'feat1<10')

        number is the quantity of points that are to be deleted, the default None means
        unrestricted deletion.

        start and end are parameters indicating range based deletion: if range based
        deletion is employed, toDelete must be None, and vice versa. If only one of start
        and end are non-None, the other defaults to 0 and self.points respectably.

        randomize indicates whether random sampling is to be used in conjunction with the number
        parameter, if randomize is False, the chosen points are determined by point order,
        otherwise it is uniform random across the space of possible removals.

        """
        self._readOnlyException("deletePoints")

    def deleteFeatures(self, toDelete=None, start=None, end=None, number=None, randomize=False):
        """
        Modify this object, removing those features that are specified by the input.

        toDelete may be a single identifier (name and/or index), a list of identifiers,
        a function that when given a feature will return True if it is to be deleted, or a
        filter function, as a string, containing a comparison operator between a point name
        and a value (i.e 'point1<10')

        number is the quantity of features that are to be deleted, the default None means
        unrestricted deleted.

        start and end are parameters indicating range based deletion: if range based
        deletion is employed, toDelete must be None, and vice versa. If only one of start
        and end are non-None, the other defaults to 0 and self.features respectably.

        randomize indicates whether random sampling is to be used in conjunction with the number
        parameter, if randomize is False, the chosen features are determined by feature order,
        otherwise it is uniform random across the space of possible removals.

        """
        self._readOnlyException("deleteFeatures")

    def retainPoints(self, toRetain=None, start=None, end=None, number=None, randomize=False):
        """
        Modify this object, retaining those points that are specified by the input.

        toRetain may be a single identifier (name and/or index), a list of identifiers,
        a function that when given a point will return True if it is to be retained, or a
        filter function, as a string, containing a comparison operator between a feature name
        and a value (i.e 'feat1<10')

        number is the quantity of points that are to be retained, the default None means
        unrestricted retention.

        start and end are parameters indicating range based retention: if range based
        retention is employed, toRetain must be None, and vice versa. If only one of start
        and end are non-None, the other defaults to 0 and self.points respectably.

        randomize indicates whether random sampling is to be used in conjunction with the number
        parameter, if randomize is False, the chosen points are determined by point order,
        otherwise it is uniform random across the space of possible retentions.

        """
        self._readOnlyException("retainPoints")

    def retainFeatures(self, toRetain=None, start=None, end=None, number=None, randomize=False):
        """
        Modify this object, retaining those features that are specified by the input.

        toRetain may be a single identifier (name and/or index), a list of identifiers,
        a function that when given a feature will return True if it is to be deleted, or a
        filter function, as a string, containing a comparison operator between a point name
        and a value (i.e 'point1<10')

        number is the quantity of features that are to be retained, the default None means
        unrestricted retention.

        start and end are parameters indicating range based retention: if range based
        retention is employed, toRetain must be None, and vice versa. If only one of start
        and end are non-None, the other defaults to 0 and self.features respectably.

        randomize indicates whether random sampling is to be used in conjunction with the number
        parameter, if randomize is False, the chosen features are determined by feature order,
        otherwise it is uniform random across the space of possible retentions.

        """
        self._readOnlyException("retainFeatures")

    def referenceDataFrom(self, other):
        """
        Modifies the internal data of this object to refer to the same data as other. In other
        words, the data wrapped by both the self and other objects resides in the
        same place in memory. Other must be an object of the same type as
        the calling object. Also, the shape of other should be consistent with the set
        of featureNames currently in this object. None is always returned.

        """
        self._readOnlyException("referenceDataFrom")

    def transformEachPoint(self, function, points=None):
        """
        Modifies this object to contain the results of the given function
        calculated on the specified points in this object.

        function must not be none and accept the view of a point as an argument

        points may be None to indicate application to all points, a single point
        ID or a list of point IDs to limit application only to those specified.

        """
        self._readOnlyException("transformEachPoint")


    def transformEachFeature(self, function, features=None):
        """
        Modifies this object to contain the results of the given function
        calculated on the specified features in this object.

        function must not be none and accept the view of a feature as an argument

        features may be None to indicate application to all features, a single
        feature ID or a list of feature IDs to limit application only to those
        specified.

        """
        self._readOnlyException("transformEachFeature")

    def transformEachElement(self, function, points=None, features=None, preserveZeros=False,
                             skipNoneReturnValues=False):
        """
        Modifies this object to contain the results of calling function(elementValue)
        or function(elementValue, pointNum, featureNum) for each element.

        points: Limit to only elements of the specified points; may be None for
        all points, a single ID, or a list of IDs.

        features: Limit to only elements of the specified features; may be None for
        all features, a single ID, or a list of IDs.

        preserveZeros: If True it does not apply the function to elements in
        the data that are 0, and that 0 is not modified.

        skipNoneReturnValues: If True, any time function() returns None, the
        value originally in the data will remain unmodified.

        """
        self._readOnlyException("transformEachElement")

    def fillWith(self, values, pointStart, featureStart, pointEnd, featureEnd):
        """
        Revise the contents of the calling object so that it contains the provided
        values in the given location.

        values - Either a constant value or a UML object whose size is consistent
        with the given start and end indices.

        pointStart - the inclusive ID of the first point in the calling object
        whose contents will be modified.

        featureStart - the inclusive ID of the first feature in the calling object
        whose contents will be modified.

        pointEnd - the inclusive ID of the last point in the calling object
        whose contents will be modified.

        featureEnd - the inclusive ID of the last feature in the calling object
        whose contents will be modified.

        """
        self._readOnlyException("fillWith")

    def handleMissingValues(self, method='remove points', features=None, arguments=None, alsoTreatAsMissing=[], markMissing=False):
        """
        This function is to remove, replace or impute missing values in an UML container data object.

        method - a str. It can be 'remove points', 'remove features', 'feature mean', 'feature median', 'feature mode', 'zero', 'constant', 'forward fill'
        'backward fill', 'interpolate'

        features - can be None to indicate all features, or a str to indicate the name of a single feature, or an int to indicate
        the index of a feature, or a list of feature names, or a list of features' indices, or a list of mix of feature names and feature indices. In this function, only those features in the input 'features'
        will be processed.

        arguments - for some kind of methods, we need to setup arguments.
        for method = 'remove points', 'remove features', arguments can be 'all' or 'any' etc.
        for method = 'constant', arguments must be a value
        for method = 'interpolate', arguments can be a dict which stores inputs for numpy.interp

        alsoTreatAsMissing -  a list. In this function, numpy.NaN and None are always treated as missing. You can add extra values which
        should be treated as missing values too, in alsoTreatAsMissing.

        markMissing: True or False. If it is True, then extra columns for those features will be added, in which 0 (False) or 1 (True) will be filled to
        indicate if the value in a cell is originally missing or not.

        """
        self._readOnlyException("handleMissingValues")

    def flattenToOnePoint(self):
        """
        Adjust this object in place so that the same values are all in a single point.

        Each feature in the result maps to exactly one value from the original object.
        The order of values respects the point order from the original object,
        if there were n features in the original, the first n values in the result
        will exactly match the first point, the nth to (2n-1)th values will exactly
        match the original second point, etc. The feature names will be transformed
        such that the value at the intersection of the "pn_i" named point and "fn_j"
        named feature from the original object will have a feature name of "fn_j | pn_i".
        The single point will have a name of "Flattened".

        Raises: ImproperActionException if an axis has length 0

        """
        self._readOnlyException("flattenToOnePoint")

    def flattenToOneFeature(self):
        """
        Adjust this object in place so that the same values are all in a single feature.

        Each point in the result maps to exactly one value from the original object.
        The order of values respects the feature order from the original object,
        if there were n points in the original, the first n values in the result
        will exactly match the first feature, the nth to (2n-1)th values will exactly
        match the original second feature, etc. The point names will be transformed
        such that the value at the intersection of the "pn_i" named point and "fn_j"
        named feature from the original object will have a point name of "pn_i | fn_j".
        The single feature will have a name of "Flattened".

        Raises: ImproperActionException if an axis has length 0

        """
        self._readOnlyException("flattenToOneFeature")

    def unflattenFromOnePoint(self, numPoints):
        """
        Adjust this point vector in place to an object that it could have been flattend from.

        This is an inverse of the method flattenToOnePoint: if an object foo with n
        points calls the flatten method, then this method with n as the argument, the
        result should be identical to the original foo. It is not limited to objects
        that have previously had flattenToOnePoint called on them; any object whose
        structure and names are consistent with a previous call to flattenToOnePoint
        may call this method. This includes objects with all default names.

        Raises: ArgumentException if numPoints does not divide the length of the point
        vector.

        Raises: ImproperActionException if an axis has length 0, there is more than one
        point, or the names are inconsistent with a previous call to flattenToOnePoint.

        """
        self._readOnlyException("unflattenFromOnePoint")


    def unflattenFromOneFeature(self, numFeatures):
        """
        Adjust this feature vector in place to an object that it could have been flattend from.

        This is an inverse of the method flattenToOneFeature: if an object foo with n
        features calls the flatten method, then this method with n as the argument, the
        result should be identical to the original foo. It is not limited to objects
        that have previously had flattenToOneFeature called on them; any object whose
        structure and names are consistent with a previous call to flattenToOneFeature
        may call this method. This includes objects with all default names.

        Raises: ArgumentException if numPoints does not divide the length of the point
        vector.

        Raises: ImproperActionException if an axis has length 0, there is more than one
        point, or the names are inconsistent with a previous call to flattenToOnePoint.

        """
        self._readOnlyException("unflattenFromOneFeature")


    ###############################################################
    ###############################################################
    ###   Subclass implemented numerical operation functions    ###
    ###############################################################
    ###############################################################

    def elementwiseMultiply(self, other):
        """
        Perform element wise multiplication of this UML data object against the
        provided other UML data object, with the result being stored in-place in
        the calling object. Both objects must contain only numeric data. The
        pointCount and featureCount of both objects must be equal. The types of
        the two objects may be different. None is always returned.

        """
        self._readOnlyException("elementwiseMultiply")

    def elementwisePower(self, other):
        self._readOnlyException("elementwisePower")

    def __imul__(self, other):
        """
        Perform in place matrix multiplication or scalar multiplication, depending in the
        input 'other'

        """
        self._readOnlyException("__imul__")

    def __iadd__(self, other):
        """
        Perform in-place addition on this object, element wise if 'other' is a UML data
        object, or element wise with a scalar if other is some kind of numeric
        value.

        """
        self._readOnlyException("__iadd__")

    def __isub__(self, other):
        """
        Subtract (in place) from this object, element wise if 'other' is a UML data
        object, or element wise with a scalar if other is some kind of numeric
        value.

        """
        self._readOnlyException("__isub__")

    def __idiv__(self, other):
        """
        Perform division (in place) using this object as the numerator, element
        wise if 'other' is a UML data object, or element wise by a scalar if other
        is some kind of numeric value.

        """
        self._readOnlyException("__idiv__")

    def __itruediv__(self, other):
        """
        Perform true division (in place) using this object as the numerator, element
        wise if 'other' is a UML data object, or element wise by a scalar if other
        is some kind of numeric value.

        """
        self._readOnlyException("__itruediv__")

    def __ifloordiv__(self, other):
        """
        Perform floor division (in place) using this object as the numerator, element
        wise if 'other' is a UML data object, or element wise by a scalar if other
        is some kind of numeric value.

        """
        self._readOnlyException("__ifloordiv__")

    def __imod__(self, other):
        """
        Perform mod (in place) using the elements of this object as the dividends,
        element wise if 'other' is a UML data object, or element wise by a scalar
        if other is some kind of numeric value.

        """
        self._readOnlyException("__imod__")

    def __ipow__(self, other):
        """
        Perform in-place exponentiation (iterated __mul__) using the elements
        of this object as the bases, element wise if 'other' is a UML data
        object, or element wise by a scalar if other is some kind of numeric
        value.

        """
        self._readOnlyException("__ipow__")


    ####################
    ####################
    ###   Helpers    ###
    ####################
    ####################

    def _readOnlyException(self, name):
        msg = "The " + name + " method is disallowed for View objects. View "
        msg += "objects are read only, yet this method modifies the object"
        raise ImproperActionException(msg)
