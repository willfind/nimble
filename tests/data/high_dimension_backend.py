"""

"""
import sys
from io import StringIO
import tempfile
import re

import numpy as np
import pandas as pd
import scipy.sparse

import nimble
from nimble.exceptions import ImproperObjectAction
from tests.helpers import raises
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


class HighDimensionSafeSparseUnsafe(DataTestObject):
    def test_highDimension_stringRepresentations(self):
        stdoutBackup = sys.stdout
        for tensor in tensors:
            toTest = self.constructor(tensor, name='test')
            dims = ' x '.join(map(str, toTest._dims))
            expData = toTest.copy()._data # get 2D data
            exp = self.constructor(expData, name='test')
            shape = '3pt x {0}ft'.format(len(exp.features))
            assert len(exp._dims) == 2
            assert toTest.toString() == exp.toString()
            testStr = str(toTest).split('\n')
            expStr = str(exp).split('\n')
            assert dims in testStr[0] and shape in expStr[0]
            assert testStr[1:] == expStr[1:]
            testRepr = repr(toTest).split('\n')
            expRepr = repr(exp).split('\n')
            assert dims in testRepr[0] and shape in expRepr[0]
            assert testRepr[1:] == expRepr[1:]

            try:
                stdout1 = StringIO()
                stdout2 = StringIO()

                sys.stdout = stdout1
                show = toTest.show()
                sys.stdout = stdout2
                expShow = exp.show()
                stdout1.seek(0)
                stdout2.seek(0)
                testLines = stdout1.readlines()
                expLines = stdout2.readlines()
                assert len(testLines) > 0
                for l1, l2 in zip(testLines, expLines):
                    if l1.startswith('"test"'):
                        both = dims + " dimensions encoded as " + shape
                        assert l1.endswith(both + '\n')
                        assert l2.endswith(shape + '\n')
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
                    exp = nimble.data(tensor, returnType=rType)
                    assert toTest._dims == testCopy._dims
                    assert testCopy == exp

                listCopy = toTest.copy('python list')
                assert listCopy == tensor

                arrCopy = toTest.copy('numpy array')
                assert np.array_equal(arrCopy, np.array(tensor))
                assert arrCopy.shape == toTest.dimensions

                objArr = np.empty(toTest._dims[:2], dtype=np.object_)
                for i, lst in enumerate(tensor):
                    objArr[i] = lst

                matCopy = toTest.copy('numpy matrix')
                assert np.array_equal(matCopy, np.matrix(objArr))

                cooCopy = toTest.copy('scipy coo')
                expCoo = scipy.sparse.coo_matrix(objArr)
                # coo __eq__ fails for object dtype b/c attempt conversion to csr
                assert np.array_equal(cooCopy.data, expCoo.data)
                assert np.array_equal(cooCopy.row, expCoo.row)
                assert np.array_equal(cooCopy.col, expCoo.col)

                dfCopy = toTest.copy('pandas dataframe')
                assert np.array_equal(dfCopy, pd.DataFrame(objArr))

                for cType in ['listofdict', 'dictoflist', 'scipycsc',
                              'scipycsr']:
                    with raises(ImproperObjectAction):
                        toTest.copy(cType)

                with raises(ImproperObjectAction):
                    toTest.copy('pythonlist', outputAs1D=True)

                with raises(ImproperObjectAction):
                    toTest.copy('pythonlist', rowsArePoints=False)

    def test_highDimension_views(self):
        for tensor in tensors:
            toTest = self.constructor(tensor)
            testView = toTest.view()
            assert toTest._dims == testView._dims

            ptView = toTest.pointView(1)
            assert ptView._dims[0] == toTest._dims[1]
            assert ptView._dims[1:] == toTest._dims[2:]

            ptsView = toTest.view(pointStart=1, pointEnd=2)
            assert ptsView._dims[0] == 2
            assert ptsView._dims[1:] == toTest._dims[1:]

            with raises(ImproperObjectAction):
                ftsView = toTest.view(featureStart=1, featureEnd=2)

    def test_highDimension_save(self):
        for tensor in tensors:
            toSave = self.constructor(tensor)
            toSaveShape = toSave._dims
            toSaveType = toSave.getTypeString()
            assert len(toSave._dims) > 2

            with tempfile.NamedTemporaryFile(suffix=".pickle") as tmpFile:
                toSave.save(tmpFile.name)
                pklObj = nimble.data(tmpFile.name, returnType=toSaveType)

            with tempfile.NamedTemporaryFile(suffix=".hdf5") as tmpFile:
                toSave.save(tmpFile.name)
                hdf5Obj = nimble.data(tmpFile.name, returnType=toSaveType)

            assert pklObj._dims == hdf5Obj._dims == toSaveShape
            assert toSave.isIdentical(pklObj)

            assert toSave.isIdentical(hdf5Obj)
            assert pklObj.isIdentical(toSave)
            assert hdf5Obj.isIdentical(toSave)
            
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
    
    def test_highDimension_binaryOperations(self):
        ops = ['__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__',
               '__mod__', '__radd__', '__rsub__', '__rmul__', '__rtruediv__',
               '__rfloordiv__', '__rmod__']
        for op in ops:
            for tensor in nzTensors:
                toTest = self.constructor(tensor)
                ret = getattr(toTest, op)(3)
                assert ret._dims == toTest._dims

                ret = getattr(toTest, op)(toTest)
                assert ret._dims == toTest._dims

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
            assert expPt._dims[0] == 1 and len(expPt._dims) == 2

            toTestPt = orig.copy()
            toTestPt.flatten()
            assert toTestPt == expPt

            toTestPt.unflatten(orig._dims)
            assert toTestPt == orig

            toTestFt = orig.copy()
            with raises(ImproperObjectAction):
                toTestFt.flatten(order='feature')

            flat = expPt.copy()
            with raises(ImproperObjectAction):
                flat.unflatten(orig._dims, order='feature')

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
                assert pt._dims == toTest._dims[1:]
                assert not pt.points._namesCreated()
                assert not pt.features._namesCreated()
                
    def test_highDimension_points_getitem(self):
        for tensor in tensors:
            toTest = self.constructor(tensor)
            for i in range(len(toTest.points)):
                pt = toTest.points[i]
                assert pt._dims == toTest._dims[1:]

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
                with raises(ImproperObjectAction):
                    retCopy = testCopy.points.copy(arg)
    
    def test_highDimension_axis_unique(self):
        for tensor in tensors:
            toTest = self.constructor(tensor)
            unique = toTest.points.unique()
            assert unique._dims[0] == 1
            assert unique._dims[1:] == toTest._dims[1:]
    
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
            assert len(train.points) == 2
            assert len(train._dims) > 2
            assert len(test.points) == 1
            assert len(test._dims) > 2

            with raises(ImproperObjectAction):
                fourTuple = toTest.trainAndTestSets(0.33, labels=0)

    def test_highDimension_trainAndTestSets_dataObject_nimbleLabel(self):
        objectLabel = nimble.data([['dog'], ['cat'], ['cat']])
        for tensor in tensors:
            toTest = self.constructor(tensor)
            trainX, trainY, testX, testY = toTest.trainAndTestSets(0.33,labels=objectLabel)
            assert len(trainX.points) == 2
            assert len(trainX._dims) > 2
            assert len(testX.points) == 1
            assert len(testX._dims) > 2
            assert len(trainY.points) == 2
            assert len(testY.points) == 1
    
    def test_highDimension_getTypeString(self):
        retType = self.constructor([]).getTypeString()
        for tensor in tensors:
            toTest = self.constructor(tensor)
            assert toTest.getTypeString() == retType

    def test_highDimension_containsZero(self):
        noZeros = [[1, 1, 2, -3, 3], [-4, 5, -5, 6, -7], [-8, 9, 9, -9, 8]]
        noZerosTensors = makeTensorData(noZeros)

        for tensor in tensors:
            toTest = self.constructor(tensor)
            assert toTest.containsZero()
        for nzTensor in noZerosTensors:
            toTest = self.constructor(nzTensor)
            assert not toTest.containsZero()

    def test_highDimension_axis_count(self):
        for tensor in tensors:
            toTest = self.constructor(tensor)

            with raises(ImproperObjectAction):
                toTest.points.count(lambda pt: True)

            with raises(ImproperObjectAction):
                toTest.features.count(lambda ft: True)



class HighDimensionSafeSparseSafe(DataTestObject):
        pass
    

class HighDimensionModifyingSparseUnsafe(DataTestObject):

    def test_showIndicesInsteadOfNames(self):
        '''Test that show() works with indices instead of names.'''

        testData = nimble.data([[6666666666, 11111111, 99999999, 555555555],
                                [6666666666, 11111111, 22222222, 555555555]],
                               featureNames=['0000_0000_0000_0000', '1111_1111_1111_1111',
                                             '2222_2222_2222_2222', '3333_3333_3333_3333'],
                               pointNames=['A', 'B'])

        old_output = sys.stdout
        temp_output = StringIO()
        sys.stdout = temp_output
        testData.show(includePointNames=True, includeFeatureNames=False)
        sys.stdout = old_output

        printed_out = re.search('(\\n *)(.*?)\\n', temp_output.getvalue()).group(2)
        indexCharList = printed_out.split(' ')
        no_of_index_chars = sum(len(s) for s in indexCharList if s)
        assert no_of_index_chars == 4

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

            with raises(ImproperObjectAction):
                toTest.features.sort(by=lambda ft: -1)

            with raises(ImproperObjectAction):
                toTest.points.sort(0)

            with raises(ImproperObjectAction):
                toTest.features.sort(0)

    def test_highDimension_insertAndAppend(self):
        for tensor in tensors:
            toTest = self.constructor(tensor)
            origShape = toTest._dims
            numPts = len(toTest.points)

            toTest.points.insert(0, toTest)
            assert toTest._dims[0] == numPts * 2
            assert toTest._dims[1:] == origShape[1:]

            toTest.points.append(toTest)
            assert toTest._dims[0] == numPts * 4
            assert toTest._dims[1:] == origShape[1:]

            with raises(ImproperObjectAction):
                toTest.features.insert(0, toTest)

            with raises(ImproperObjectAction):
                toTest.features.append(toTest)
    
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

                with raises(ImproperObjectAction):
                    retExtract = testExtract.points.extract(arg)

                with raises(ImproperObjectAction):
                    retDelete = testDelete.points.delete(arg)

                with raises(ImproperObjectAction):
                    retRetain = testRetain.points.retain(arg)

    def test_highDimension_axis_repeat(self):
        for tensor in tensors:
            toTest = self.constructor(tensor)
            origShape = toTest._dims
            repeated = toTest.points.repeat(2, True)
            assert toTest._dims == origShape
            assert repeated._dims[0] == origShape[0] * 2
            assert repeated._dims[1:] == origShape[1:]

            with raises(ImproperObjectAction):
                repeated = toTest.features.repeat(2, True)

    def test_highDimension_axis_replace(self):
        for tensor in tensors:
            toTest = self.constructor(tensor)
            replacement = self.constructor([tensor[0]])

            toTest.points.replace(replacement, 1)

            assert toTest.points[0] == toTest.points[1]

            toTest = self.constructor(tensor)

            expShape = toTest._dims

            exp0 = toTest.points[0]
            exp1 = toTest.points[1]
            replacement = self.constructor(tensor[:2])

            toTest.points.replace(replacement, [1, 2])

            assert toTest.points[1] == exp0
            assert toTest.points[2] == exp1
            assert toTest._dims == expShape
    
    def test_highDimension_strings(self):
        """Test that string representations work for high dimensions"""
        for tensor in tensors:
            toTest = self.constructor(tensor)
            str(toTest)
            repr(toTest)
            toTest.show()
            str(toTest.points)
            repr(toTest.points)
            str(toTest.features)
            repr(toTest.features)

class HighDimensionModifyingSparseSafe(DataTestObject):
  
    def test_highDimension_referenceFrom(self):
        toTest3D = self.constructor(tensors[0])
        toTest4D = self.constructor(tensors[1])
        toTest5D = self.constructor(tensors[2])

        for tensor1 in tensors:
            for tensor2 in tensors:
                testTensor = self.constructor(tensor1)
                refTensor = self.constructor(tensor2)
                if testTensor != refTensor:
                    testTensor._referenceFrom(refTensor)
                    assert testTensor == refTensor

    def test_highDimension_inplaceBinaryOperations(self):
        ops = ['__iadd__', '__isub__', '__imul__', '__itruediv__',
               '__ifloordiv__']
        for op in ops:
            for tensor in nzTensors:
                toTest = self.constructor(tensor)
                ret = getattr(toTest, op)(2)
                assert ret._dims == toTest._dims

                toTest = self.constructor(tensor)
                ret = getattr(toTest, op)(toTest)
                assert ret._dims == toTest._dims

    def test_highDimension_insertAndAppend_wrongShape(self):
        for tens1 in tensors:
            for tens2 in tensors:
                toTest = self.constructor(tens1)
                toInsert = self.constructor(tens2)
                if toTest._dims != toInsert._dims:
                    with raises(ImproperObjectAction):
                        toTest.points.insert(0, toInsert)

                    with raises(ImproperObjectAction):
                        toTest.points.append(toInsert)

    def test_highDimension_permute(self):
        for tensor in tensors:
            toTest = self.constructor(tensor, pointNames=['a', 'b', 'c'])
            origShape = toTest._dims
            permuted = False
            for i in range(5):
                toTest.points.permute()
                assert toTest._dims == origShape
                if toTest.points.getNames() != ['a', 'b', 'c']:
                    permuted = True
                    break
            assert permuted

            with raises(ImproperObjectAction):
                toTest.features.permute()

            toTest = self.constructor(tensor, pointNames=['a', 'b', 'c'])

            toTest.points.permute([2, 0, 1])

            expData = [tensor[2], tensor[0], tensor[1]]
            exp = self.constructor(expData, pointNames=['c', 'a', 'b'])
            assert toTest == exp

            with raises(ImproperObjectAction):
                toTest.features.permute([2, 0, 1])


    def test_highDimension_disallowed(self):
        # Goal here is to make sure that all functions that we have not
        # explicitly allowed for high dimension data are wrapped in the
        # limitTo2D decorator. This should ensure  that any new functionality
        # is automatically flagged if not wrapped with limitTo2D or explicitly
        # allowed below.

        def getNimbleDefined(cls):
            nimbleDefined = set()
            ignore = ['__init__', '__init_subclass__', '__subclasshook__',
                      '__class__']
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
            '__deepcopy__', '__getattr__', 'isApproximatelyEqual',
            'trainAndTestSets',
            'report', 'isIdentical', 'getTypeString', 'pointView', 'view',
            'checkInvariants', 'containsZero', 'save', 'toString', 'show',
            'copy', 'flatten', 'unflatten',
            'min', 'max', 'mean', 'median', 'uniqueCount',
            'proportionMissing', 'proportionZero'
            ))
        baseDisallowed = baseUser.difference(baseAllowed)

        for method in baseDisallowed:
            if method == 'T': # only property we need to test
                assert isLimitedTo2D(toTest, method, callable=False)
            else:
                assert isLimitedTo2D(toTest, method)

        axisAllowed = set((
            '__len__', 'getName', 'getNames', 'setNames',
            'getIndex', 'getIndices', 'hasName',))
        ptUser = getNimbleDefined(nimble.core.data.Points)
        ptAllowed = set((
            '__iter__', '__getitem__', 'copy', 'extract', 'delete', 'retain',
            'sort', 'insert', 'append', 'permute', 'repeat', 'replace',
            'unique',))
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
