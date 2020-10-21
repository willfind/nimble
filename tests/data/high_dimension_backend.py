"""

"""
import sys
from io import StringIO
import tempfile

import numpy
import pandas as pd
import scipy.sparse
from nose.tools import raises

import nimble
from nimble.exceptions import ImproperObjectAction
from .baseObject import DataTestObject

############################
# Data objects for testing #
############################
def makeTensorData(matrix):
    rank3List = [matrix, matrix, matrix]
    rank4List = [rank3List, rank3List, rank3List]
    rank5List = [rank4List, rank4List, rank4List]

    return  [rank3List, rank4List, rank5List]

matrix = [[0, 1, 2, 3, 0], [4, 5, 0, 6, 7], [8, 0, 9, 0, 8]]
tensors = makeTensorData(matrix)

emptyTensors = makeTensorData([[], [], []])

nzMatrix = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [-1, -2, -3, -4, -5]]

nzTensors = makeTensorData(nzMatrix)

class HighDimensionSafe(DataTestObject):

    def test_highDimension_equality(self):
        for tens1, tens2 in zip(tensors, tensors):
            toTest1 = self.constructor(tens1)
            toTest2 = self.constructor(tens2)
            assert toTest1.isIdentical(toTest2)
            assert toTest2.isIdentical(toTest1)
            assert toTest1 == toTest2
            assert toTest2 == toTest1

        vector1 = [0, 1, 2.000000001, 3, 0]
        vector2 = [4, 5, 0, 6, 7.0000000002]
        vector3 = [8, 0, 8.9999999999, 0, 8]
        matrix = [vector1, vector2, vector3]

        apprxTensors = makeTensorData(matrix)

        for tens1, tens2 in zip(tensors, apprxTensors):
            toTest1 = self.constructor(tens1)
            toTest2 = self.constructor(tens2)
            assert toTest1.isApproximatelyEqual(toTest2)
            assert toTest2.isApproximatelyEqual(toTest1)

        for tens1, tens2 in zip(tensors, tensors[1:]):
            toTest1 = self.constructor(tens1)
            toTest2 = self.constructor(tens2)
            assert not toTest1.isIdentical(toTest2)
            assert not toTest2.isIdentical(toTest1)
            assert not toTest1.isApproximatelyEqual(toTest2)
            assert not toTest2.isApproximatelyEqual(toTest1)
            assert toTest1 != toTest2
            assert toTest2 != toTest1

    def test_highDimension_trainAndTestSets(self):
        for tensor in tensors:
            toTest = self.constructor(tensor)
            train, test = toTest.trainAndTestSets(0.33)
            assert train._pointCount == 2
            assert len(train._shape) > 2
            assert test._pointCount == 1
            assert len(train._shape) > 2

            try:
                fourTuple = toTest.trainAndTestSets(0.33, labels=0)
                assert False
            except ImproperObjectAction:
                pass

    def test_highDimension_stringRepresentations(self):
        stdoutBackup = sys.stdout
        for tensor in tensors:
            toTest = self.constructor(tensor, name='test')
            expData = toTest.copy().data # get 2D data
            exp = self.constructor(expData, name='test')
            assert len(exp._shape) == 2
            assert toTest.toString() == exp.toString()
            assert toTest.__str__() == exp.__str__()
            assert toTest.__repr__() == exp.__repr__()


            try:
                stdout1 = StringIO()
                stdout2 = StringIO()

                sys.stdout = stdout1
                show = toTest.show('testing')
                sys.stdout = stdout2
                expShow = exp.show('testing')
                stdout1.seek(0)
                stdout2.seek(0)
                testLines = stdout1.readlines()
                expLines = stdout2.readlines()
                assert len(testLines) > 0
                for l1, l2 in zip(testLines, expLines):
                    if l1.startswith('test :'):
                        stringHD = ' x '.join(map(str, toTest._shape)) + ' \n'
                        string2D = '3pt x {0}ft \n'.format(exp._featureCount)
                        assert l1.endswith(stringHD)
                        assert l2.endswith(string2D)
                    else:
                        assert l1 == l2
            finally:
                sys.stdout = stdoutBackup

    def test_highDimension_copy(self):
        for tensorList in [tensors, emptyTensors]:
            for tensor in tensorList:
                toTest = self.constructor(tensor)
                for rType in nimble.core.data.available:
                    testCopy = toTest.copy(rType)
                    exp = nimble.data(rType, tensor)
                    assert toTest._shape == testCopy._shape
                    assert testCopy == exp

                listCopy = toTest.copy('python list')
                assert listCopy == tensor

                arrCopy = toTest.copy('numpy array')
                assert numpy.array_equal(arrCopy, numpy.array(tensor))
                assert arrCopy.shape == toTest.dimensions

                objArr = numpy.empty(toTest._shape[:2], dtype=numpy.object_)
                for i, lst in enumerate(tensor):
                    objArr[i] = lst

                matCopy = toTest.copy('numpy matrix')
                assert numpy.array_equal(matCopy, numpy.matrix(objArr))

                cooCopy = toTest.copy('scipy coo')
                expCoo = scipy.sparse.coo_matrix(objArr)
                # coo __eq__ fails for object dtype b/c attempt conversion to csr
                assert numpy.array_equal(cooCopy.data, expCoo.data)
                assert numpy.array_equal(cooCopy.row, expCoo.row)
                assert numpy.array_equal(cooCopy.col, expCoo.col)

                dfCopy = toTest.copy('pandas dataframe')
                assert numpy.array_equal(dfCopy, pd.DataFrame(objArr))

                for cType in ['listofdict', 'dictoflist', 'scipycsc',
                              'scipycsr']:
                    try:
                        toTest.copy(cType)
                        assert False
                    except ImproperObjectAction:
                        pass

                try:
                    toTest.copy('pythonlist', outputAs1D=True)
                    assert False
                except ImproperObjectAction:
                    pass

                try:
                    toTest.copy('pythonlist', rowsArePoints=False)
                    assert False
                except ImproperObjectAction:
                    pass

    def test_highDimension_views(self):
        for tensor in tensors:
            toTest = self.constructor(tensor)
            testView = toTest.view()
            assert toTest._shape == testView._shape

            ptView = toTest.pointView(1)
            assert ptView._shape[0] == toTest._shape[1]
            assert ptView._shape[1:] == toTest._shape[2:]

            ptsView = toTest.view(pointStart=1, pointEnd=2)
            assert ptsView._shape[0] == 2
            assert ptsView._shape[1:] == toTest._shape[1:]

            try:
                ftsView = toTest.view(featureStart=1, featureEnd=2)
                assert False # expected ImproperObjectAction
            except ImproperObjectAction:
                pass

    def test_highDimension_save(self):
        for tensor in tensors:
            toSave = self.constructor(tensor)
            toSaveShape = toSave._shape
            assert len(toSave._shape) > 2

            with tempfile.NamedTemporaryFile(suffix=".nimd") as tmpFile:
                toSave.save(tmpFile.name)
                loadObj = nimble.loadData(tmpFile.name)

            assert loadObj._shape == toSaveShape
            assert toSave.isIdentical(loadObj)
            assert loadObj.isIdentical(toSave)

    def test_highDimension_getTypeString(self):
        retType = self.constructor([]).getTypeString()
        for tensor in tensors:
            toTest = self.constructor(tensor)
            assert toTest.getTypeString() == retType

    def test_highDimension_posNegAbs(self):
        mixedSigns = [[0, -1, -2, 3, 0], [4, -5, 0, -6, 7], [8, 0, -9, 0, -8]]
        mixedTensors = makeTensorData(mixedSigns)
        oppSigns = [[0, 1, 2, -3, 0], [-4, 5, 0, 6, -7], [-8, 0, 9, 0, 8]]
        expNegative = makeTensorData(oppSigns)
        expAbsolute = tensors

        for mixed, neg, ab in zip(mixedTensors, expNegative, expAbsolute):
            toTest = self.constructor(mixed)
            expNeg = self.constructor(neg)
            expAbs = self.constructor(ab)
            assert +toTest == toTest
            assert -toTest == expNeg
            assert abs(toTest) == expAbs

    def test_highDimension_containsZero(self):
        noZeros = [[1, 1, 2, -3, 3], [-4, 5, -5, 6, -7], [-8, 9, 9, -9, 8]]
        noZerosTensors = makeTensorData(noZeros)

        for tensor in tensors:
            toTest = self.constructor(tensor)
            assert toTest.containsZero()
        for nzTensor in noZerosTensors:
            toTest = self.constructor(nzTensor)
            assert not toTest.containsZero()

    def test_highDimension_binaryOperations(self):
        ops = ['__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__',
               '__mod__', '__radd__', '__rsub__', '__rmul__', '__rtruediv__',
               '__rfloordiv__', '__rmod__']
        for op in ops:
            for tensor in nzTensors:
                toTest = self.constructor(tensor)
                ret = getattr(toTest, op)(3)
                assert ret._shape == toTest._shape

                ret = getattr(toTest, op)(toTest)
                assert ret._shape == toTest._shape

    def test_highDimension_flattenAndUnflatten(self):

        def flattenTensor(tensor, store=None):
            if store is None:
                store = []
            if not isinstance(tensor, list):
                store.append(tensor)
            else:
                for pt in tensor:
                    flattenTensor(pt, store)

            return store

        for tensor in tensors:
            flatPt = flattenTensor(tensor)
            orig = self.constructor(tensor)
            expPt = self.constructor(flatPt, pointNames=['Flattened'])
            # sanity check flattenTensor worked as expected
            assert expPt._shape[0] == 1 and len(expPt._shape) == 2

            toTestPt = orig.copy()
            toTestPt.flatten()
            assert toTestPt == expPt

            toTestPt.unflatten(orig._shape)
            assert toTestPt == orig

            toTestFt = orig.copy()
            try:
                toTestFt.flatten(order='feature')
                assert False
            except ImproperObjectAction:
                pass

            flat = expPt.copy()
            try:
                flat.unflatten(orig._shape, order='feature')
                assert False
            except ImproperObjectAction:
                pass

    def test_highDimension_points_iter(self):
        for idx, tensor in enumerate(tensors):
            flattenedLen = 15
            for i in range(idx % 3):
                flattenedLen *= 3
            ftNames = ['ft' + str(x) for x in range(flattenedLen)]
            ptNames = ['a', 'b', 'c']
            toTest = self.constructor(tensor, pointNames=ptNames,
                                      featureNames=ftNames)
            for pt in toTest.points:
                assert pt._shape == toTest._shape[1:]
                assert not pt.points._namesCreated()
                assert not pt.features._namesCreated()

    def test_highDimension_points_getitem(self):
        for tensor in tensors:
            toTest = self.constructor(tensor)
            for i in range(len(toTest.points)):
                pt = toTest.points[i]
                assert pt._shape == toTest._shape[1:]

    def test_highDimension_points_copy(self):
        params = [([[0, 1]], {}),
                  ([['a', 'b']], {}),
                  ([], {'start': 0, 'end': 1}),
                  ([], {'start': 'a', 'end': 'b'}),
                  ([], {'number': 2})]
        for tensor in tensors:
            for args, kwargs in params:
                orig = self.constructor(tensor, pointNames=['a', 'b', 'c'])
                testCopy = orig.copy()

                exp1 = self.constructor(tensor[:2], pointNames=['a', 'b'])
                exp2 = self.constructor(tensor[2:], pointNames=['c'])

                retCopy = testCopy.points.copy(*args, **kwargs)
                assert retCopy == exp1
                assert testCopy == orig

        params = [lambda pt: True, 'ft0=0']
        for i, tensor in enumerate(tensors):
            for arg in params:
                ftNames = ['ft' + str(j) for j in range(5 * (3 ** (i + 1)))]
                testCopy = self.constructor(tensor, pointNames=['a', 'b', 'c'],
                                            featureNames=ftNames)
                try:
                    retCopy = testCopy.points.copy(arg)
                    assert False
                except ImproperObjectAction:
                    pass

    def test_highDimension_axis_count(self):
        for tensor in tensors:
            toTest = self.constructor(tensor)

            try:
                toTest.points.count(lambda pt: True)
                assert False
            except ImproperObjectAction:
                pass

            try:
                toTest.features.count(lambda ft: True)
                assert False
            except ImproperObjectAction:
                pass

    def test_highDimension_axis_unique(self):
        for tensor in tensors:
            toTest = self.constructor(tensor)
            unique = toTest.points.unique()
            assert unique._shape[0] == 1
            assert unique._shape[1:] == toTest._shape[1:]

class HighDimensionModifying(DataTestObject):

    def test_highDimension_referenceDataFrom(self):
        toTest3D = self.constructor(tensors[0])
        toTest4D = self.constructor(tensors[1])
        toTest5D = self.constructor(tensors[2])

        for tensor1 in tensors:
            for tensor2 in tensors:
                testTensor = self.constructor(tensor1)
                refTensor = self.constructor(tensor2)
                if testTensor != refTensor:
                    testTensor._referenceDataFrom(refTensor)
                    assert testTensor == refTensor

    def test_highDimension_inplaceBinaryOperations(self):
        ops = ['__iadd__', '__isub__', '__imul__', '__itruediv__',
               '__ifloordiv__']
        for op in ops:
            for tensor in nzTensors:
                toTest = self.constructor(tensor)
                ret = getattr(toTest, op)(2)
                assert ret._shape == toTest._shape

                toTest = self.constructor(tensor)
                ret = getattr(toTest, op)(toTest)
                assert ret._shape == toTest._shape

    def test_highDimension_sort(self):

        tensor3D = [[[2]], [[3]], [[1]]]
        tensor4D = [[[[2]]], [[[3]]], [[[1]]]]
        tensor5D = [[[[[2]]]], [[[[3]]]], [[[[1]]]]]
        sortTensors = [tensor3D, tensor4D, tensor5D]

        def getValue(obj):
            try:
                return int(obj[0])
            except TypeError:
                return getValue(obj.points[0])

        for tensor in sortTensors:
            toTest = self.constructor(tensor, pointNames=['b', 'c', 'a'])

            toTest.points.sort()
            assert getValue(toTest.points[0]) == 1
            assert getValue(toTest.points[1]) == 2
            assert getValue(toTest.points[2]) == 3
            assert toTest.points.getNames() == ['a', 'b', 'c']

            toTest.points.sort(reverse=True)
            assert getValue(toTest.points[0]) == 3
            assert getValue(toTest.points[1]) == 2
            assert getValue(toTest.points[2]) == 1
            assert toTest.points.getNames() == ['c', 'b', 'a']

            def valSort(pt):
                return getValue(pt)

            toTest.points.sort(valSort)
            assert getValue(toTest.points[0]) == 1
            assert getValue(toTest.points[1]) == 2
            assert getValue(toTest.points[2]) == 3
            assert toTest.points.getNames() == ['a', 'b', 'c']

            toTest.points.sort(valSort, reverse=True)
            assert getValue(toTest.points[0]) == 3
            assert getValue(toTest.points[1]) == 2
            assert getValue(toTest.points[2]) == 1
            assert toTest.points.getNames() == ['c', 'b', 'a']

            def valCompare(pt1, pt2):
                return getValue(pt1) - getValue(pt2)

            toTest.points.sort(valCompare)
            assert getValue(toTest.points[0]) == 1
            assert getValue(toTest.points[1]) == 2
            assert getValue(toTest.points[2]) == 3
            assert toTest.points.getNames() == ['a', 'b', 'c']

            toTest.points.sort(valCompare, reverse=True)
            assert getValue(toTest.points[0]) == 3
            assert getValue(toTest.points[1]) == 2
            assert getValue(toTest.points[2]) == 1
            assert toTest.points.getNames() == ['c', 'b', 'a']

            try:
                toTest.features.sort(by=lambda ft: -1)
                assert False
            except ImproperObjectAction:
                pass

            try:
                toTest.points.sort(0)
                assert False
            except ImproperObjectAction:
                pass

            try:
                toTest.features.sort(0)
                assert False
            except ImproperObjectAction:
                pass

    def test_highDimension_insertAndAppend(self):
        for tensor in tensors:
            toTest = self.constructor(tensor)
            origShape = toTest._shape
            numPts = len(toTest.points)

            toTest.points.insert(0, toTest)
            assert toTest._shape[0] == numPts * 2
            assert toTest._shape[1:] == origShape[1:]

            toTest.points.append(toTest)
            assert toTest._shape[0] == numPts * 4
            assert toTest._shape[1:] == origShape[1:]

            try:
                toTest.features.insert(0, toTest)
                assert False
            except ImproperObjectAction:
                pass

            try:
                toTest.features.append(toTest)
                assert False
            except ImproperObjectAction:
                pass

    def test_highDimension_insertAndAppend_wrongShape(self):
        for tens1 in tensors:
            for tens2 in tensors:
                toTest = self.constructor(tens1)
                toInsert = self.constructor(tens2)
                if toTest._shape != toInsert._shape:
                    try:
                        toTest.points.insert(0, toInsert)
                        assert False
                    except ImproperObjectAction:
                        pass

                    try:
                        toTest.points.append(toInsert)
                        assert False
                    except ImproperObjectAction:
                        pass

    def test_highDimension_permute(self):
        for tensor in tensors:
            toTest = self.constructor(tensor, pointNames=['a', 'b', 'c'])
            origShape = toTest._shape
            permuted = False
            for i in range(5):
                toTest.points.permute()
                assert toTest._shape == origShape
                if toTest.points.getNames() != ['a', 'b', 'c']:
                    permuted = True
                    break
            assert permuted

            try:
                toTest.features.permute()
                assert False
            except ImproperObjectAction:
                pass

            toTest = self.constructor(tensor, pointNames=['a', 'b', 'c'])

            toTest.points.permute([2, 0, 1])

            expData = [tensor[2], tensor[0], tensor[1]]
            exp = self.constructor(expData, pointNames=['c', 'a', 'b'])
            assert toTest == exp

            try:
                toTest.features.permute([2, 0, 1])
                assert False
            except ImproperObjectAction:
                pass


    def test_highDimension_points_structuralModifying(self):
        params = [([[0, 1]], {}),
                  ([['a', 'b']], {}),
                  ([], {'start': 0, 'end': 1}),
                  ([], {'start': 'a', 'end': 'b'}),
                  ([], {'number': 2})]
        for tensor in tensors:
            for args, kwargs in params:
                orig = self.constructor(tensor, pointNames=['a', 'b', 'c'])
                testExtract = orig.copy()
                testDelete = orig.copy()
                testRetain = orig.copy()

                exp1 = self.constructor(tensor[:2], pointNames=['a', 'b'])
                exp2 = self.constructor(tensor[2:], pointNames=['c'])

                retExtract = testExtract.points.extract(*args, **kwargs)
                assert retExtract == exp1
                assert testExtract == exp2

                retDelete = testDelete.points.delete(*args, **kwargs)
                assert retDelete is None
                assert testDelete == exp2

                retRetain = testRetain.points.retain(*args, **kwargs)
                assert retRetain is None
                assert testRetain == exp1

        params = [lambda pt: True, 'ft0=0']
        for i, tensor in enumerate(tensors):
            for arg in params:
                ftNames = ['ft' + str(j) for j in range(5 * (3 ** (i + 1)))]
                orig = self.constructor(tensor, pointNames=['a', 'b', 'c'],
                                        featureNames=ftNames)
                testExtract = orig.copy()
                testDelete = orig.copy()
                testRetain = orig.copy()

                try:
                    retExtract = testExtract.points.extract(arg)
                    assert False
                except ImproperObjectAction:
                    pass

                try:
                    retDelete = testDelete.points.delete(arg)
                    assert False
                except ImproperObjectAction:
                    pass

                try:
                    retRetain = testRetain.points.retain(arg)
                    assert False
                except ImproperObjectAction:
                    pass

    def test_highDimension_axis_repeat(self):
        for tensor in tensors:
            toTest = self.constructor(tensor)
            origShape = toTest._shape
            repeated = toTest.points.repeat(2, True)
            assert toTest._shape == origShape
            assert repeated._shape[0] == origShape[0] * 2
            assert repeated._shape[1:] == origShape[1:]

            try:
                repeated = toTest.features.repeat(2, True)
                assert False
            except ImproperObjectAction:
                pass

    def test_highDimension_disallowed(self):
        # Goal here is to make sure that all functions that we have not
        # explicitly allowed for high dimension data are wrapped in the
        # limitTo2D decorator. This should ensure  that any new functionality
        # is automatically flagged if not wrapped with limitTo2D or explicitly
        # allowed below.

        def getNimbleDefined(cls):
            nimbleDefined = set()
            ignore = ['__init__', '__init_subclass__', '__subclasshook__']
            objectDir = dir(object)
            for attr in dir(cls):
                if attr in ignore:
                    continue
                if attr.startswith('_') and not attr.startswith('__'):
                    continue
                bound = getattr(cls, attr)
                if not callable(bound) and attr != 'T':
                    continue
                if attr in objectDir:
                    # add only if we redefined it
                    if getattr(object, attr) != bound:
                        nimbleDefined.add(attr)
                else:
                    nimbleDefined.add(attr)
            return nimbleDefined

        def isLimitedTo2D(obj, method, callable=True):
            # Unfortunately, using mock is problematic when dealing with
            # decorators, so instead this checks that the exception message
            # matches our expectations.
            endStr = "{0} is not permitted when the ".format(method)
            endStr += "data has more than two dimensions"
            if callable:
                try:
                    getattr(obj, method)()
                    return False
                except ImproperObjectAction as ioa:
                    return str(ioa) == endStr
            else:
                try:
                    getattr(obj, method)
                    return False
                except ImproperObjectAction as ioa:
                    return str(ioa) == endStr

        toTest = self.constructor(tensors[0])

        baseUser = getNimbleDefined(nimble.core.data.Base)
        baseAllowed = set((
            '__bool__', '__eq__', '__ne__', '__add__', '__radd__', '__iadd__',
            '__sub__', '__rsub__', '__isub__', '__mul__', '__rmul__',
            '__imul__', '__truediv__', '__rtruediv__', '__itruediv__',
            '__floordiv__', '__rfloordiv__', '__ifloordiv__', '__mod__',
            '__rmod__', '__imod__', '__pow__', '__rpow__', '__ipow__',
            '__str__', '__repr__', '__pos__', '__neg__', '__abs__', '__copy__',
            '__deepcopy__', 'nameIsDefault', 'isApproximatelyEqual',
            'trainAndTestSets', 'summaryReport', 'isIdentical', 'writeFile',
            'getTypeString', 'pointView', 'view', 'validate', 'containsZero',
            'save', 'toString', 'show', 'referenceDataFrom', 'copy',
            'flatten', 'unflatten',))
        baseDisallowed = baseUser.difference(baseAllowed)

        for method in baseDisallowed:
            if method == 'T': # only property we need to test
                assert isLimitedTo2D(toTest, method, callable=False)
            else:
                assert isLimitedTo2D(toTest, method)

        axisAllowed = set((
            '__len__', 'getName', 'getNames', 'setName', 'setNames',
            'getIndex', 'getIndices', 'hasName',))
        ptUser = getNimbleDefined(nimble.core.data.Points)
        ptAllowed = set((
            '__iter__', '__getitem__', 'copy', 'extract', 'delete', 'retain',
            'sort', 'insert', 'append', 'permute', 'repeat', 'unique',))
        ptAllAllowed = axisAllowed.union(ptAllowed)
        ptDisallowed = ptUser.difference(ptAllAllowed)

        ftUser = getNimbleDefined(nimble.core.data.Features)
        # only the Axis methods are allowed for features
        ftDisallowed = ftUser.difference(axisAllowed)

        ptAxis = getattr(toTest, 'points')
        for method in ptDisallowed:
            assert isLimitedTo2D(ptAxis, method)

        ftAxis = getattr(toTest, 'features')
        for method in ftDisallowed:
            assert isLimitedTo2D(ftAxis, method)

class HighDimensionAll(HighDimensionSafe, HighDimensionModifying):
    pass
