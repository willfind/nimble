"""

"""
import sys
from io import StringIO

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

    def test_highDimension_len(self):
        for tensor in tensors:
            toTest = self.constructor(tensor)
            try:
                len(toTest)
                assert False # expected ImproperObjectAction
            except ImproperObjectAction:
                pass

    def test_highDimension_bool(self):
        for tensor in tensors:
            toTest = self.constructor(tensor)
            assert bool(toTest)

        for tensor in emptyTensors:
            toTest = self.constructor(tensor)
            assert not bool(toTest)

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
            expData = toTest.copy().data # get flattened data
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
        for tensor in tensors:
            toTest = self.constructor(tensor)
            testCopy = toTest.copy()
            assert toTest._shape == testCopy._shape

    def test_highDimension_views(self):
        for tensor in tensors:
            toTest = self.constructor(tensor)
            testView = toTest.view()
            assert toTest._shape == testView._shape

            ptView = toTest.pointView(1)
            assert ptView._shape == toTest._shape[1:]

            ptsView = toTest.view(pointStart=1, pointEnd=2)
            assert ptsView._shape[0] == 2
            assert ptsView._shape[1:] == toTest._shape[1:]

            try:
                ftView = toTest.featureView(1)
                assert False # expected ImproperObjectAction
            except ImproperObjectAction:
                pass

            try:
                ftsView = toTest.view(featureStart=1, featureEnd=2)
                assert False # expected ImproperObjectAction
            except ImproperObjectAction:
                pass

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
                assert ret._shape == toTest._shape

                ret = getattr(toTest, op)(toTest)
                assert ret._shape == toTest._shape

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
                    testTensor.referenceDataFrom(refTensor)
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

class HighDimensionAll(HighDimensionSafe, HighDimensionModifying):
    pass
