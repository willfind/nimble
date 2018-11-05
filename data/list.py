"""
Class extending Base, using a list of lists to store data.

"""

from __future__ import division
from __future__ import absolute_import
import copy
import numpy
import numbers
import itertools

import UML
from .base import Base, cmp_to_key
from .base_view import BaseView
from .dataHelpers import inheritDocstringsFactory
from .dataHelpers import reorderToMatchList
from .dataHelpers import DEFAULT_PREFIX
from UML.exceptions import ArgumentException, PackageException
from UML.randomness import pythonRandom
import six
from six.moves import range
from functools import reduce

scipy = UML.importModule('scipy.io')
pd = UML.importModule('pandas')

allowedItemType = (numbers.Number, six.string_types)
def isAllowedSingleElement(x):
    """
    This function is to determine if an element is an allowed single element
    """
    if isinstance(x, allowedItemType):
        return True

    if hasattr(x, '__len__'):#not a single element
        return False

    if x is None or x != x:#None and np.NaN are allowed
        return True

    return

@inheritDocstringsFactory(Base)
class List(Base):
    """
    Class providing implementations of data manipulation operations on data stored
    in a list of lists implementation, where the outer list is a list of
    points of data, and each inner list is a list of values for each feature.
    """

    def __init__(self, data, featureNames=None, reuseData=False, shape=None, checkAll=True, elementType=None, **kwds):
        """
        data can be a list, a np matrix or a ListPassThrough
        reuseData only works when input data is a list
        if checkAll is True, then it will do validity check for all elements
        """
        if (not isinstance(data, (list, numpy.matrix))) and 'PassThrough' not in str(type(data)):
            msg = "the input data can only be a list or a numpy matrix or ListPassThrough."
            raise ArgumentException(msg)

        if isinstance(data, list):
            #case1: data=[]. self.data will be [], shape will be (0, shape[1]) or (0, len(featureNames)) or (0, 0)
            if len(data) == 0:
                if shape:
                    shape = (0, shape[1])
                else:
                    shape = (0, len(featureNames) if featureNames else 0)
            elif isAllowedSingleElement(data[0]):
            #case2: data=['a', 'b', 'c'] or [1,2,3]. self.data will be [[1,2,3]], shape will be (1, 3)
                if checkAll:#check all items
                    for i in data:
                        if not isAllowedSingleElement(i):
                            msg = 'invalid input data format.'
                            raise ArgumentException(msg)
                shape = (1, len(data))
                data = [data]
            elif isinstance(data[0], list) or hasattr(data[0], 'setLimit'):
            #case3: data=[[1,2,3], ['a', 'b', 'c']] or [[]] or [[], []]. self.data will be = data, shape will be (len(data), len(data[0]))
            #case4: data=[<UML.data.list.FeatureViewer object at 0x43fd410>]
                numFeatures = len(data[0])
                if checkAll:#check all items
                    for i in data:
                        if len(i) != numFeatures:
                            msg = 'invalid input data format.'
                            raise ArgumentException(msg)
                        for j in i:
                            if not isAllowedSingleElement(j):
                                msg = '%s is invalid input data format.'%j
                                raise ArgumentException(msg)
                shape = (len(data), numFeatures)

            if reuseData:
                data = data
            else:
                data = [copy.deepcopy(i) for i in data]#copy.deepcopy(data)
                #this is to convert a list x=[[1,2,3]]*2 to a list y=[[1,2,3], [1,2,3]]
                #the difference is that x[0] is x[1], but y[0] is not y[1]

        if isinstance(data, numpy.matrix):
            #case5: data is a numpy matrix. shape is already in np matrix
            shape = data.shape
            data = data.tolist()

        if len(data) == 0:
            #case6: data is a ListPassThrough associated with empty list
            data = []

        self._numFeatures = shape[1]
        self.data = data
        self._elementType = elementType

        kwds['featureNames'] = featureNames
        kwds['shape'] = shape
        super(List, self).__init__(**kwds)


    def _transpose_implementation(self):
        """
        Function to transpose the data, ie invert the feature and point indices of the data.

        This is not an in place operation, a new list of lists is constructed.
        """
        tempFeatures = len(self.data)
        transposed = []
        #load the new data with an empty point for each feature in the original
        for i in range(self.features):
            transposed.append([])
        for point in self.data:
            for i in range(len(point)):
                transposed[i].append(point[i])

        self.data = transposed
        self._numFeatures = tempFeatures

    def _addPoints_implementation(self, toAdd, insertBefore):
        """
        Insert the points from the toAdd object below the provided index in
        this object, the remaining points from this object will continue below
        the inserted points

        """
        insertedLength = self.points + toAdd.points
        insertRange = range(insertBefore, insertBefore + toAdd.points)
        insertIndex = 0
        selfIndex = 0
        allData = []
        for pointIndex in range(insertedLength):
            if pointIndex in insertRange:
                allData.append(toAdd.data[insertIndex])
                insertIndex += 1
            else:
                allData.append(self.data[selfIndex])
                selfIndex += 1
        self.data = allData

    def _addFeatures_implementation(self, toAdd, insertBefore):
        """
        Insert the features from the toAdd object to the right of the
        provided index in this object, the remaining points from this object
        will continue to the right of the inserted points

        """
        for i in range(self.points):
            startData = self.data[i][:insertBefore]
            endData = self.data[i][insertBefore:]
            allPointData = startData + list(toAdd.data[i]) + endData
            self.data[i] = allPointData
        self._numFeatures = self._numFeatures + toAdd.features


    def _sortPoints_implementation(self, sortBy, sortHelper):
        return self._sort_generic_implementation(sortBy, sortHelper, 'point')

    def _sortFeatures_implementation(self, sortBy, sortHelper):
        return self._sort_generic_implementation(sortBy, sortHelper, 'feature')

    def _sort_generic_implementation(self, sortBy, sortHelper, axis):
        if axis == 'point':
            test = self.pointView(0)
            viewIter = self.pointIterator()
            indexGetter = self.getPointIndex
            nameGetter = self.getPointName
            nameGetterStr = 'getPointName'
            names = self.getPointNames()
        else:
            test = self.featureView(0)
            viewIter = self.featureIterator()
            indexGetter = self.getFeatureIndex
            nameGetter = self.getFeatureName
            nameGetterStr = 'getFeatureName'
            names = self.getFeatureNames()

        if isinstance(sortHelper, list):
            sortData = numpy.array(self.data, dtype=numpy.object_)
            if axis == 'point':
                sortData = sortData[sortHelper, :]
            else:
                sortData = sortData[:, sortHelper]
            self.data = sortData.tolist()
            newNameOrder = [names[idx] for idx in sortHelper]
            return newNameOrder

        scorer = None
        comparator = None
        try:
            sortHelper(test)
            scorer = sortHelper
        except TypeError:
            pass
        try:
            sortHelper(test, test)
            comparator = sortHelper
        except TypeError:
            pass

        if sortHelper is not None and scorer is None and comparator is None:
            raise ArgumentException("sortHelper is neither a scorer or a comparator")

        # make array of views
        viewArray = []
        for v in viewIter:
            viewArray.append(v)

        if comparator is not None:
            # try:
            #     viewArray.sort(cmp=comparator)#python2
            # except:
            viewArray.sort(key=cmp_to_key(comparator))#python2 and 3
            indexPosition = []
            for i in range(len(viewArray)):
                index = indexGetter(getattr(viewArray[i], nameGetterStr)(0))
                indexPosition.append(index)
        else:
            #scoreArray = viewArray
            scoreArray = []
            if scorer is not None:
                # use scoring function to turn views into values
                for i in range(len(viewArray)):
                    scoreArray.append(scorer(viewArray[i]))
            else:
                for i in range(len(viewArray)):
                    scoreArray.append(viewArray[i][sortBy])

            # use numpy.argsort to make desired index array
            # this results in an array whole ith index contains the the
            # index into the data of the value that should be in the ith
            # position
            indexPosition = numpy.argsort(scoreArray)

        # run through target axis and change indices
        if axis == 'point':
            source = copy.copy(self.data)
            for i in range(len(self.data)):
                self.data[i] = source[indexPosition[i]]
        else:
            for i in range(len(self.data)):
                currPoint = self.data[i]
                temp = copy.copy(currPoint)
                for j in range(len(indexPosition)):
                    currPoint[j] = temp[indexPosition[j]]

        # we convert the indices of the their previous location into their feature names
        newNameOrder = []
        for i in range(len(indexPosition)):
            oldIndex = indexPosition[i]
            newName = nameGetter(oldIndex)
            newNameOrder.append(newName)
        return newNameOrder


    def _structuralBackend_implementation(self, structure, axis, targetList):
        """
        Backend for extractPoints/Features, deletePoints/Features, retainPoints/Features, and
        copyPoints/Features. Returns a new object containing only the points in targetList and
        performs some modifications to the original object if necessary. This function does not
        perform all of the modification or process how each function handles the returned value,
        these are managed separately by each frontend function.
        """
        pnames = []
        fnames = []
        data = numpy.matrix(self.data, dtype=object)

        if axis == 'point':
            keepList = [idx for idx in range(self.points) if idx not in targetList]
            satisfying = data[targetList, :]
            if structure != 'copy':
                keep = data[keepList, :]
                self.data = keep.tolist()

            for index in targetList:
                pnames.append(self.getPointName(index))
            fnames = self.getFeatureNames()

        else:
            if self.data == []:
                # create empty matrix with correct shape
                data = numpy.matrix(numpy.empty((self.points,self.features)), dtype=object)

            keepList = [idx for idx in range(self.features) if idx not in targetList]
            satisfying = data[:, targetList]
            if structure != 'copy':
                keep = data[:, keepList]
                self.data = keep.tolist()

            for index in targetList:
                fnames.append(self.getFeatureName(index))
            pnames = self.getPointNames()

            if structure != 'copy':
                self._numFeatures = self._numFeatures - len(targetList)

        return List(satisfying, pointNames=pnames, featureNames=fnames, reuseData=True)


    def _mapReducePoints_implementation(self, mapper, reducer):
        mapResults = {}
        # apply the mapper to each point in the data
        for i in range(self.points):
            currResults = mapper(self.pointView(i))
            # the mapper will return a list of key value pairs
            for (k, v) in currResults:
                # if key is new, we must add an empty list
                if k not in mapResults:
                    mapResults[k] = []
                # append this value to the list of values associated with the key
                mapResults[k].append(v)

        # apply the reducer to the list of values associated with each key
        ret = []
        for mapKey in mapResults.keys():
            mapValues = mapResults[mapKey]
            # the reducer will return a tuple of a key to a value
            redRet = reducer(mapKey, mapValues)
            if redRet is not None:
                (redKey, redValue) = redRet
                ret.append([redKey, redValue])
        return List(ret, reuseData=True)

    def _getTypeString_implementation(self):
        return 'List'

    def _isIdentical_implementation(self, other):
        if not isinstance(other, List):
            return False
        if self.points != other.points:
            return False
        if self.features != other.features:
            return False
        for index in range(self.points):
            if self.data[index] != other.data[index]:
                return False
        return True

    def _writeFile_implementation(self, outPath, format, includePointNames, includeFeatureNames):
        """
        Function to write the data in this object to a file using the specified
        format. outPath is the location (including file name and extension) where
        we want to write the output file. includeNames is boolean argument
        indicating whether the file should start with comment lines designating
        pointNames and featureNames.

        """
        if format not in ['csv', 'mtx']:
            msg = "Unrecognized file format. Accepted types are 'csv' and 'mtx'. They may "
            msg += "either be input as the format parameter, or as the extension in the "
            msg += "outPath"
            raise ArgumentException(msg)

        if format == 'csv':
            return self._writeFileCSV_implementation(outPath, includePointNames, includeFeatureNames)
        if format == 'mtx':
            return self._writeFileMTX_implementation(outPath, includePointNames, includeFeatureNames)

    def _writeFileCSV_implementation(self, outPath, includePointNames, includeFeatureNames):
        """
        Function to write the data in this object to a CSV file at the designated
        path.

        """
        outFile = open(outPath, 'w')

        if includeFeatureNames:
            def combine(a, b):
                return a + ',' + b

            fnames = self.getFeatureNames()
            fnamesLine = reduce(combine, fnames)
            fnamesLine += '\n'
            if includePointNames:
                outFile.write('point_names,')

            outFile.write(fnamesLine)

        for point in self.pointIterator():
            currPname = point.getPointName(0)
            first = True
            if includePointNames:
                outFile.write(currPname)
                first = False

            for value in point:
                if not first:
                    outFile.write(',')
                outFile.write(str(value))
                first = False
            outFile.write('\n')
        outFile.close()

    def _writeFileMTX_implementation(self, outPath, includePointNames, includeFeatureNames):
        """
        Function to write the data in this object to a matrix market file at the designated
        path.

        """
        outFile = open(outPath, 'w')
        outFile.write("%%MatrixMarket matrix array real general\n")

        def writeNames(nameList):
            for i, n in enumerate(nameList):
                if i == 0:
                    outFile.write('%#')
                else:
                    outFile.write(',')
                outFile.write(n)
            outFile.write('\n')

        if includePointNames:
            writeNames(self.getPointNames())
        else:
            outFile.write('%#\n')
        if includeFeatureNames:
            writeNames(self.getFeatureNames())
        else:
            outFile.write('%#\n')

        outFile.write(str(self.points) + " " + str(self.features) + "\n")

        for j in range(self.features):
            for i in range(self.points):
                value = self.data[i][j]
                outFile.write(str(value) + '\n')
        outFile.close()

    def _referenceDataFrom_implementation(self, other):
        if not isinstance(other, List):
            raise ArgumentException("Other must be the same type as this object")

        self.data = other.data
        self._numFeatures = other._numFeatures

    def _copyAs_implementation(self, format):

        if format == 'Sparse':
            if self.points == 0 or self.features == 0:
                emptyData = numpy.empty(shape=(self.points, self.features))
                return UML.createData('Sparse', emptyData)
            return UML.createData('Sparse', self.data)

        if format is None or format == 'List':
            if self.points == 0 or self.features == 0:
                emptyData = numpy.empty(shape=(self.points, self.features))
                return UML.createData('List', emptyData)
            else:
                return UML.createData('List', self.data)
        if format == 'Matrix':
            if self.points == 0 or self.features == 0:
                emptyData = numpy.empty(shape=(self.points, self.features))
                return UML.createData('Matrix', emptyData)
            else:
                return UML.createData('Matrix', self.data)
        if format == 'DataFrame':
            if self.points == 0 or self.features == 0:
                emptyData = numpy.empty(shape=(self.points, self.features))
                return UML.createData('DataFrame', emptyData)
            else:
                return UML.createData('DataFrame', self.data)
        if format == 'pythonlist':
            return copy.deepcopy(self.data)
        if format == 'numpyarray':
            if self.points == 0 or self.features == 0:
                return numpy.empty(shape=(self.points, self.features))
            return numpy.array(self.data, dtype=self._elementType)
        if format == 'numpymatrix':
            if self.points == 0 or self.features == 0:
                return numpy.matrix(numpy.empty(shape=(self.points, self.features)))
            return numpy.matrix(self.data)
        if format == 'scipycsc':
            if not scipy:
                msg = "scipy is not available"
                raise PackageException(msg)
            return scipy.sparse.csc_matrix(numpy.array(self.data))
        if format == 'scipycsr':
            if not scipy:
                msg = "scipy is not available"
                raise PackageException(msg)
            return scipy.sparse.csr_matrix(numpy.array(self.data))



    def _calculateForEachElement_implementation(self, function, points, features,
                                                preserveZeros, outputType):
        return self._calculateForEachElementGenericVectorized(
               function, points, features, outputType)


    def _transformEachPoint_implementation(self, function, points):
        for i, p in enumerate(self.pointIterator()):
            if points is not None and i not in points:
                continue
            currRet = function(p)
            if len(currRet) != self.features:
                msg = "function must return an iterable with as many elements as features in this object"
                raise ArgumentException(msg)

            self.data[i] = currRet

    def _transformEachFeature_implementation(self, function, features):
        for j, f in enumerate(self.featureIterator()):
            if features is not None and j not in features:
                continue
            currRet = function(f)
            if len(currRet) != self.points:
                msg = "function must return an iterable with as many elements as points in this object"
                raise ArgumentException(msg)

            for i in range(self.points):
                self.data[i][j] = currRet[i]

    def _transformEachElement_implementation(self, toTransform, points, features, preserveZeros, skipNoneReturnValues):
        oneArg = False
        try:
            toTransform(0, 0, 0)
        except TypeError:
            if isinstance(toTransform, dict):
                oneArg = None
            else:
                oneArg = True

        IDs = itertools.product(range(self.points), range(self.features))
        for (i, j) in IDs:
            currVal = self.data[i][j]

            if points is not None and i not in points:
                continue
            if features is not None and j not in features:
                continue
            if preserveZeros and currVal == 0:
                continue

            if oneArg is None:
                if currVal in toTransform.keys():
                    currRet = toTransform[currVal]
                else:
                    continue
            elif oneArg:
                currRet = toTransform(currVal)
            else:
                currRet = toTransform(currVal, i, j)

            if skipNoneReturnValues and currRet is None:
                continue

            self.data[i][j] = currRet


    def _fillWith_implementation(self, values, pointStart, featureStart, pointEnd, featureEnd):
        if not isinstance(values, UML.data.Base):
            values = [values] * (featureEnd - featureStart + 1)
            for p in range(pointStart, pointEnd + 1):
                self.data[p][featureStart:featureEnd + 1] = values
        else:
            for p in range(pointStart, pointEnd + 1):
                self.data[p][featureStart:featureEnd + 1] = values.data[p - pointStart]

    def _handleMissingValues_implementation(self, method='remove points', featuresList=None, arguments=None, alsoTreatAsMissing=[], markMissing=False):
        """
        This function is to
        1. drop points or features with missing values
        2. fill missing values with mean, median, mode, or zero or a constant value
        3. fill missing values by forward or backward filling
        4. imput missing values via linear interpolation

        Detailed steps are:
        1. from alsoTreatAsMissing, generate a Set for elements which are not None nor NaN but should be considered as missing
        2. from featuresList, generate 2 dicts missingIdxDictFeature and missingIdxDictPoint to store locations of missing values
        3. replace missing values in features in the featuresList with NaN
        4. based on method and arguments, process self.data
        5. update points and features information.
        """
        def featureMeanMedianMode(func):
            featureMean = self.calculateForEachFeature(func, features=featuresList)
            for tmpItem in missingIdxDictFeature.items():
                j = tmpItem[0]
                for i in tmpItem[1]:
                    self.data[i][j] = featureMean[0, j]

        alsoTreatAsMissingSet = set(alsoTreatAsMissing)
        missingIdxDictFeature = {i: [] for i in featuresList}
        missingIdxDictPoint = {i: [] for i in range(self.points)}
        for i in range(self.points):
            for j in featuresList:
                tmpV = self.data[i][j]
                if tmpV in alsoTreatAsMissingSet or (tmpV!=tmpV) or tmpV is None:
                    #numpy.NaN and None are always treated as missing
                    self.data[i][j] = None if tmpV is None else numpy.NaN
                    missingIdxDictPoint[i].append(j)
                    missingIdxDictFeature[j].append(i)
        featureNames = self.getFeatureNames()
        pointNames = self.getPointNames()
        if markMissing:
            #add extra columns to indicate if the original value was missing or not
            extraFeatureNames = []
            extraDummy = []
            for tmpItem in missingIdxDictFeature.items():
                extraFeatureNames.append(self.getFeatureName(tmpItem[0]) + '_missing')
            for tmpItem in missingIdxDictPoint.items():
                extraDummy.append([True if i in tmpItem[1] else False for i in featuresList])

            #extraDummy = numpy.matrix(extraDummy).transpose()
        # import pdb; pdb.set_trace()
        #from now, based on method and arguments, process self.data
        if method == 'remove points':
            msg = 'for method = "remove points", the arguments can only be all( or None) or any.'
            if arguments is None or arguments.lower() == 'any':
                missingIdx = [i[0] for i in missingIdxDictPoint.items() if len(i[1]) > 0]
            elif arguments.lower() == 'all':
                missingIdx = [i[0] for i in missingIdxDictPoint.items() if len(i[1]) == self.features]
            else:
                raise ArgumentException(msg)
            nonmissingIdx = [i for i in range(self.points) if i not in missingIdx]
            if len(nonmissingIdx) == 0:
                msg = 'All data are removed. Please use another method or other arguments.'
                raise ArgumentException(msg)
            pointNames = [self.getPointName(i) for i in nonmissingIdx]
            if len(missingIdx) > 0:
                self.data = [self.data[i] for i in nonmissingIdx]
                if markMissing:
                    extraDummy = [extraDummy[i] for i in nonmissingIdx]
        elif method == 'remove features':
            msg = 'for method = "remove features", the arguments can only be all( or None) or any.'
            if arguments is None or arguments.lower() == 'any':
                missingIdx = [i[0] for i in missingIdxDictFeature.items() if len(i[1]) > 0]
            elif arguments.lower() == 'all':
                missingIdx = [i[0] for i in missingIdxDictFeature.items() if len(i[1]) == self.points]
            else:
                raise ArgumentException(msg)
            nonmissingIdx = [i for i in range(self.features) if i not in missingIdx]
            if len(nonmissingIdx) == 0:
                msg = 'All data are removed. Please use another method or other arguments.'
                raise ArgumentException(msg)
            featureNames = [self.getFeatureName(i) for i in nonmissingIdx]
            if len(missingIdx) > 0:
                self.data = [[i[j] for j in nonmissingIdx] for i in self.data]
                if markMissing:
                    extraDummy = [[i[j] for j in nonmissingIdx] for i in extraDummy]
                    extraFeatureNames = [extraFeatureNames[i] for i in nonmissingIdx]
        elif method == 'feature mean':
            featureMeanMedianMode(UML.calculate.mean)
        elif method == 'feature median':
            featureMeanMedianMode(UML.calculate.median)
        elif method == 'feature mode':
            featureMeanMedianMode(UML.calculate.mode)
        elif method == 'zero':
            for tmpItem in missingIdxDictFeature.items():
                j = tmpItem[0]
                for i in tmpItem[1]:
                    self.data[i][j] = 0
        elif method == 'constant':
            msg = 'for method = "constant", the arguments must be the constant.'
            if arguments is not None:
                for tmpItem in missingIdxDictFeature.items():
                    j = tmpItem[0]
                    for i in tmpItem[1]:
                        self.data[i][j] = arguments
            else:
                raise ArgumentException(msg)
        elif method == 'forward fill':
            for tmpItem in missingIdxDictFeature.items():
                    j = tmpItem[0]
                    for i in tmpItem[1]:
                        if i > 0:
                            self.data[i][j] = self.data[i-1][j]
        elif method == 'backward fill':
            for tmpItem in missingIdxDictFeature.items():
                    j = tmpItem[0]
                    for i in sorted(tmpItem[1], reverse=True):
                        if i < self.points - 1:
                            self.data[i][j] = self.data[i+1][j]
        elif method == 'interpolate':
            for tmpItem in missingIdxDictFeature.items():
                j = tmpItem[0]
                interpX = tmpItem[1]
                if len(interpX) == 0:
                    continue
                if arguments is None:
                    xp = [i for i in range(self.points) if i not in interpX]
                    fp = [self.data[i][j] for i in xp]
                    tmpArguments = {'x': interpX, 'xp': xp, 'fp': fp}
                elif isinstance(arguments, dict):
                    tmpArguments = arguments.copy()
                    tmpArguments['x'] = interpX
                else:
                    msg = 'for method = "interpolate", the arguments must be None or a dict.'
                    raise ArgumentException(msg)

                tmpV = numpy.interp(**tmpArguments)
                for k, i in enumerate(interpX):
                    self.data[i][j] = tmpV[k]

        if markMissing:
            for i in range(len(self.data)):
                self.data[i].extend(extraDummy[i])
            featureNames += extraFeatureNames
        pCount, fCount = len(self.data), len(self.data[0])
        self._featureCount = fCount
        self.setFeatureNames(featureNames)
        self._pointCount = pCount
        self.setPointNames(pointNames)

    def _flattenToOnePoint_implementation(self):
        onto = self.data[0]
        for i in range(1,self.points):
            onto += self.data[1]
            del self.data[1]

        self._numFeatures = len(onto)

    def _flattenToOneFeature_implementation(self):
        result = []
        for i in range(self.features):
            for p in self.data:
                result.append([p[i]])

        self.data = result
        self._numFeatures = 1

    def _unflattenFromOnePoint_implementation(self, numPoints):
        result = []
        numFeatures = self.features // numPoints
        for i in range(numPoints):
            temp = self.data[0][(i*numFeatures):((i+1)*numFeatures)]
            result.append(temp)

        self.data = result
        self._numFeatures = numFeatures

    def _unflattenFromOneFeature_implementation(self, numFeatures):
        result = []
        numPoints = self.points // numFeatures
        # reconstruct the shape we want, point by point. We access the singleton
        # values from the current data in an out of order iteration
        for i in range(numPoints):
            temp = []
            for j in range(i, self.points, numPoints):
                temp += self.data[j]
            result.append(temp)

        self.data = result
        self._numFeatures = numFeatures

    def _merge_implementation(self, other, point, feature, onFeature, matchingFtIdx):
        if onFeature:
            if feature == "intersection" or feature == "left":
                onIdxLoc = matchingFtIdx[0].index(self.getFeatureIndex(onFeature))
                onIdxL = onIdxLoc
                onIdxR = onIdxLoc
                right = [[row[i] for i in matchingFtIdx[1]] for row in other.data]
                # matching indices in right were sorted when slicing above
                if len(right) > 0:
                    matchingFtIdx[1] = list(range(len(right[0])))
                else:
                    matchingFtIdx[1] = []
                if feature == "intersection":
                    self.data = [[row[i] for i in matchingFtIdx[0]] for row in self.data]
                    # matching indices in left were sorted when slicing above
                    if len(self.data) > 0:
                        matchingFtIdx[0] = list(range(len(self.data[0])))
                    else:
                        matchingFtIdx[0] = []
            else:
                onIdxL = self.getFeatureIndex(onFeature)
                onIdxR = other.getFeatureIndex(onFeature)
                right = copy.copy(other.data)
        else:
            # using pointNames, prepend pointNames to left and right lists
            onIdxL = 0
            onIdxR = 0
            left = []
            right = []

            def ptNameGetter(obj, idx, suffix):
                if obj._pointNamesCreated():
                    name = obj.getPointName(idx)
                    if not name.startswith(DEFAULT_PREFIX):
                        return name
                    else:
                        return name + suffix
                else:
                    return DEFAULT_PREFIX + str(idx) + suffix

            if feature == "intersection":
                for i, pt in enumerate(self.data):
                    ptL = [ptNameGetter(self, i, '_l')]
                    intersect = [val for idx, val in enumerate(pt) if idx in matchingFtIdx[0]]
                    self.data[i] = ptL + intersect
                for i, pt in enumerate(other.data):
                    ptR = [ptNameGetter(other, i, '_r')]
                    ptR.extend([pt[i] for i in matchingFtIdx[1]])
                    right.append(ptR)
                # matching indices were sorted above
                # this also accounts for prepended column
                if len(self.data) > 0:
                    matchingFtIdx[0] = list(range(len(self.data[0])))
                else:
                    matchingFtIdx[0] = []
                matchingFtIdx[1] = matchingFtIdx[0]
            elif feature == "left":
                for i, pt in enumerate(self.data):
                    ptL = [ptNameGetter(self, i, '_l')]
                    self.data[i] = ptL + pt
                for i, pt in enumerate(other.data):
                    ptR = [ptNameGetter(other, i, '_r')]
                    ptR.extend([pt[i] for i in matchingFtIdx[1]])
                    right.append(ptR)
                # account for new column in matchingFtIdx
                matchingFtIdx[0] = list(map(lambda x: x + 1, matchingFtIdx[0]))
                matchingFtIdx[0].insert(0, 0)
                # matching indices were sorted when slicing above
                # this also accounts for prepended column
                matchingFtIdx[1] = list(range(len(right[0])))
            else:
                for i, pt in enumerate(self.data):
                    ptL = [ptNameGetter(self, i, '_l')]
                    self.data[i] = ptL + pt
                for i, pt in enumerate(other.data):
                    ptR = [ptNameGetter(other, i, '_r')]
                    ptR.extend(pt)
                    right.append(ptR)
                matchingFtIdx[0] = list(map(lambda x: x + 1, matchingFtIdx[0]))
                matchingFtIdx[0].insert(0, 0)
                matchingFtIdx[1] = list(map(lambda x: x + 1, matchingFtIdx[1]))
                matchingFtIdx[1].insert(0, 0)
        left = self.data

        matched = []
        merged = []
        unmatchedPtCountR = len(right[0]) - len(matchingFtIdx[1])
        matchMapper = {}
        for pt in left:
            match = [right[i] for i in range(len(right)) if right[i][onIdxR] == pt[onIdxL]]
            if len(match) > 0:
                matchMapper[pt[onIdxL]] = match

        for ptL in left:
            target = ptL[onIdxL]
            if target in matchMapper:
                matchesR = matchMapper[target]
                for ptR in matchesR:
                    # check for conflicts between matching features
                    matches = [ptL[i] == ptR[j] for i, j in zip(matchingFtIdx[0], matchingFtIdx[1])]
                    nansL = [ptL[i] != ptL[i] for i in matchingFtIdx[0]]
                    nansR = [ptR[j] != ptR[j] for j in matchingFtIdx[1]]
                    acceptableValues = [m + nL + nR for m, nL, nR in zip(matches, nansL, nansR)]
                    if not all(acceptableValues):
                        msg = "The objects contain different values for the same feature"
                        raise ArgumentException(msg)
                    if sum(nansL) > 0:
                        # fill any nan values in left with the corresponding right value
                        for i, value in enumerate(ptL):
                            if value != value and i in matchingFtIdx[0]:
                                ptL[i] = ptR[matchingFtIdx[1][matchingFtIdx[0].index(i)]]
                    ptR = [ptR[i] for i in range(len(ptR)) if i not in matchingFtIdx[1]]
                    pt = ptL + ptR
                    merged.append(pt)
                matched.append(target)
            elif point == 'union' or point == 'left':
                ptR = [numpy.nan] * (len(right[0]) - len(matchingFtIdx[1]))
                pt = ptL + ptR
                merged.append(pt)

        if point == 'union':
            notMatchingR = [i for i in range(len(right[0])) if i not in matchingFtIdx[1]]
            for row in right:
                target = row[onIdxR]
                if target not in matched:
                    pt = [numpy.nan] * (len(left[0]) + unmatchedPtCountR)
                    for i, j in zip (matchingFtIdx[0], matchingFtIdx[1]):
                        pt[i] = row[j]
                    pt[len(left[0]):] = [row[i] for i in range(len(right[0])) if i not in matchingFtIdx[1]]
                    merged.append(pt)

        self._featureCount = len(left[0]) + unmatchedPtCountR
        self._pointCount = len(merged)
        if onFeature is None:
            # remove point names feature
            merged = [row[1:] for row in merged]
            self._featureCount -= 1

        self.data = merged

    def _getitem_implementation(self, x, y):
        return self.data[x][y]

    def _view_implementation(self, pointStart, pointEnd, featureStart, featureEnd):
        class ListView(BaseView, List):
            def __init__(self, **kwds):
                super(ListView, self).__init__(**kwds)

            def _copyAs_implementation(self, format):
                # we only want to change how List and pythonlist copying is done
                # we also temporarily convert self.data to a python list for copyAs
                if self._pointNamesCreated():
                    pNames = self.getPointNames()
                else:
                    pNames = False
                if self._featureNamesCreated():
                    fNames = self.getFeatureNames()
                else:
                    fNames = False

                if (self.points == 0 or self.features == 0) and format != 'List':
                    emptyStandin = numpy.empty((self.points, self.features))
                    intermediate = UML.createData('Matrix', emptyStandin)
                    return intermediate.copyAs(format)

                listForm = [[self._source.data[pID][fID] for fID in range(self._fStart, self._fEnd)]
                            for pID in range(self._pStart, self._pEnd)]
                if format is None:
                    format = 'List'
                if format != 'List' and format != 'pythonlist':
                    origData = self.data
                    self.data = listForm
                    res = super(ListView, self)._copyAs_implementation(
                        format)
                    self.data = origData
                    return res

                if format == 'List':
                    return UML.createData('List', listForm, pointNames=pNames,
                                          featureNames=fNames)
                else:
                    return listForm


        class FeatureViewer(object):
            def __init__(self, source, fStart, fEnd):
                self.source = source
                self.fStart = fStart
                self.fRange = fEnd - fStart

            def setLimit(self, pIndex):
                self.limit = pIndex

            def __getitem__(self, key):
                if key < 0 or key >= self.fRange:
                    raise IndexError("")

                return self.source.data[self.limit][key + self.fStart]

            def __len__(self):
                return self.fRange

            def __eq__(self, other):
                for i, val in enumerate(self):
                    if val != other[i]:
                        return False
                return True

            def __ne__(self, other):
                return not self.__eq__(other)

        class ListPassThrough(object):
            def __init__(self, source, pStart, pEnd, fStart, fEnd):
                self.source = source
                self.pStart = pStart
                self.pEnd = pEnd
                self.pRange = pEnd - pStart
                self.fStart = fStart
                self.fEnd = fEnd
                self.fviewer = FeatureViewer(self.source, fStart, fEnd)

            def __getitem__(self, key):
                if key < 0 or key >= self.pRange:
                    raise IndexError("")

                self.fviewer.setLimit(key + self.pStart)
                return self.fviewer

            def __len__(self):
                return self.pRange

            def __array__(self, dtype=None):
                tmpArray = numpy.array(self.source.data, dtype=dtype)
                return tmpArray[self.pStart:self.pEnd, self.fStart:self.fEnd]

        kwds = {}
        kwds['data'] = ListPassThrough(self, pointStart, pointEnd, featureStart, featureEnd)
        kwds['source'] = self
        kwds['pointStart'] = pointStart
        kwds['pointEnd'] = pointEnd
        kwds['featureStart'] = featureStart
        kwds['featureEnd'] = featureEnd
        kwds['reuseData'] = True
        kwds['shape'] = (pointEnd - pointStart, featureEnd - featureStart)

        return ListView(**kwds)

    def _validate_implementation(self, level):
        assert len(self.data) == self.points
        assert self._numFeatures == self.features

        if level > 0:
            if len(self.data) > 0:
                expectedLength = len(self.data[0])
            for point in self.data:
            #				assert isinstance(point, list)
                assert len(point) == expectedLength

    def _containsZero_implementation(self):
        """
        Returns True if there is a value that is equal to integer 0 contained
        in this object. False otherwise

        """
        for point in self.pointIterator():
            for i in range(len(point)):
                if point[i] == 0:
                    return True
        return False

    def _nonZeroIteratorPointGrouped_implementation(self):
        class nzIt(object):
            def __init__(self, source):
                self._source = source
                self._pIndex = 0
                self._pStop = source.points
                self._fIndex = 0
                self._fStop = source.features

            def __iter__(self):
                return self

            def next(self):
                while (self._pIndex < self._pStop):
                    value = self._source.data[self._pIndex][self._fIndex]

                    self._fIndex += 1
                    if self._fIndex >= self._fStop:
                        self._fIndex = 0
                        self._pIndex += 1

                    if value != 0:
                        return value

                raise StopIteration

            def __next__(self):
                return self.next()

        return nzIt(self)

    def _nonZeroIteratorFeatureGrouped_implementation(self):
        class nzIt(object):
            def __init__(self, source):
                self._source = source
                self._pIndex = 0
                self._pStop = source.points
                self._fIndex = 0
                self._fStop = source.features

            def __iter__(self):
                return self

            def next(self):
                while (self._fIndex < self._fStop):
                    value = self._source.data[self._pIndex][self._fIndex]

                    self._pIndex += 1
                    if self._pIndex >= self._pStop:
                        self._pIndex = 0
                        self._fIndex += 1

                    if value != 0:
                        return value

                raise StopIteration

            def __next__(self):
                return self.next()

        return nzIt(self)


    def _mul__implementation(self, other):
        if isinstance(other, UML.data.Base):
            return self._matrixMultiply_implementation(other)
        else:
            ret = self.copy()
            ret._scalarMultiply_implementation(other)
            return ret

    def _matrixMultiply_implementation(self, other):
        """
        Matrix multiply this UML data object against the provided other UML data
        object. Both object must contain only numeric data. The featureCount of
        the calling object must equal the pointCount of the other object. The
        types of the two objects may be different, and the return is guaranteed
        to be the same type as at least one out of the two, to be automatically
        determined according to efficiency constraints.

        """
        ret = []
        for sPoint in self.pointIterator():
            retP = []
            for oFeature in other.featureIterator():
                runningTotal = 0
                for index in range(other.points):
                    runningTotal += sPoint[index] * oFeature[index]
                retP.append(runningTotal)
            ret.append(retP)
        return List(ret)


    def _elementwiseMultiply_implementation(self, other):
        """
        Perform element wise multiplication of this UML data object against the
        provided other UML data object. Both objects must contain only numeric
        data. The pointCount and featureCount of both objects must be equal. The
        types of the two objects may be different, but the returned object will
        be the inplace modification of the calling object.

        """
        for pNum in range(self.points):
            for fNum in range(self.features):
                # Divided by 1 to make it raise if it involves non-numeric types ('str')
                self.data[pNum][fNum] *= other[pNum, fNum] / 1

    def _scalarMultiply_implementation(self, scalar):
        """
        Multiply every element of this UML data object by the provided scalar.
        This object must contain only numeric data. The 'scalar' parameter must
        be a numeric data type. The returned object will be the inplace modification
        of the calling object.

        """
        for point in self.data:
            for i in range(len(point)):
                point[i] *= scalar

    def outputMatrixData(self):
        """
        convert slef.data to a numpy matrix
        """
        if len(self.data) == 0:# in case, self.data is []
            return numpy.matrix(numpy.empty([len(self.getPointNames()), len(self.getFeatureNames())]))

        return numpy.matrix(self.data)
