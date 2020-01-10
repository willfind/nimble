"""

Methods tested in this file:

In object StructureDataSafe:
copy, points.copy, features.copy

In object StructureModifying:
__init__,  transpose, T, points.insert, features.insert, points.sort,
features.sort, points.extract, features.extract, points.delete,
features.delete, points.retain, features.retain, referenceDataFrom,
points.transform, features.transform, transformElements, replaceRectangle,
flattenToOnePoint, flattenToOneFeature, merge, unflattenFromOnePoint,
unflattenFromOneFeature, points.append, features.append,
"""

import tempfile
import os
import os.path
import copy
try:
    from unittest import mock #python >=3.3
except ImportError:
    import mock

import numpy
from nose.tools import *

import nimble
from nimble import match
from nimble import createData
from nimble.data import List
from nimble.data import Matrix
from nimble.data import DataFrame
from nimble.data import Sparse
from nimble.data import BaseView
from nimble.data.dataHelpers import DEFAULT_PREFIX
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentTypeCombination
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.exceptions import ImproperObjectAction
from nimble.randomness import numpyRandom
from nimble.utility import ImportModule
from nimble.utility import cooMatrixToArray

from .baseObject import DataTestObject
from ..assertionHelpers import logCountAssertionFactory
from ..assertionHelpers import noLogEntryExpected, oneLogEntryExpected
from ..assertionHelpers import assertNoNamesGenerated
from ..assertionHelpers import CalledFunctionException, calledException

scipy = ImportModule('scipy')
pd = ImportModule('pandas')

preserveName = "PreserveTestName"
preserveAPath = os.path.join(os.getcwd(), "correct", "looking", "path")
preserveRPath = os.path.relpath(preserveAPath)
preservePair = (preserveAPath, preserveRPath)


### Helpers used by tests in the test class ###

twoLogEntriesExpected = logCountAssertionFactory(2)

def passThrough(value):
    return value


def plusOne(value):
    return (value + 1)


def plusOneOnlyEven(value):
    if value % 2 == 0:
        return (value + 1)
    else:
        return None

def noChange(value):
    return value

def allTrue(value):
    return True

def allFalse(value):
    return False

def oneOrFour(point):
    if 1 in point or 4 in point:
        return True
    return False

def absoluteOne(feature):
    if 1 in feature or -1 in feature:
        return True
    return False

def evenOnly(feature):
    return feature[0] % 2 == 0


class StructureShared(DataTestObject):
    """
    Test backends shared between the data safe and data modifying subobject
    test sets.

    """

    ###################################################################
    # common backend for exceptions extract, delete, retain, and copy #
    ###################################################################

    def back_structural_randomizeNoNumber(self, structure, axis):
        if axis == 'point':
            toCall = 'points'
        else:
            toCall = 'features'
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        callAxis = getattr(toTest, toCall)
        ret = getattr(callAxis, structure)([0,1,2], randomize=True)

    def back_structural_list_numberGreaterThanTargeted(self, structure, axis):
        if axis == 'point':
            toCall = 'points'
        else:
            toCall = 'features'
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        callAxis = getattr(toTest, toCall)
        ret = getattr(callAxis, structure)([0,1], number=3)

    def back_structural_function_numberGreaterThanTargeted(self, structure, axis):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']

        if axis == 'point':
            toCall = 'points'
            def selTwo(p):
                return p.points.getName(0) in pointNames[:2]
        else:
            toCall = 'features'
            def selTwo(f):
                return f.features.getName(0) in featureNames[:2]

        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        callAxis = getattr(toTest, toCall)
        ret = getattr(callAxis, structure)(selTwo, number=3)

    def back_structural_range_numberGreaterThanTargeted(self, structure, axis):
        if axis == 'point':
            toCall = 'points'
        else:
            toCall = 'features'
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        callAxis = getattr(toTest, toCall)
        ret = getattr(callAxis, structure)(start=0, end=1, number=3)


class StructureDataSafe(StructureShared):

    def test_objectValidationSetup(self):
        """ Test that object validation has been setup """
        assert hasattr(nimble.data.Base, 'objectValidation')
        assert hasattr(nimble.data.Features, 'objectValidation')
        assert hasattr(nimble.data.Points, 'objectValidation')

    ##########################
    # T (transpose property) #
    ##########################

    def test_T_empty(self):
        """ Test T property on different kinds of emptiness """
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)
        orig = toTest.copy()

        exp1 = [[], []]
        exp1 = numpy.array(exp1)
        ret1 = self.constructor(exp1)
        assert ret1.isIdentical(toTest.T)
        assert toTest.isIdentical(orig)


    @logCountAssertionFactory(0)
    def test_T_handmade(self):
        """ Test T property function against handmade output """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        dataTrans = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

        toTest = self.constructor(copy.deepcopy(data))
        dataObjOrig = toTest.copy()
        dataObjT = self.constructor(copy.deepcopy(dataTrans))

        assert toTest.T is not None
        assert dataObjT.isIdentical(toTest.T)
        assert toTest.isIdentical(dataObjOrig)

        assertNoNamesGenerated(toTest)

    def test_T_handmadeWithZeros(self):
        """ Test T property function against handmade output """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0], [11, 12, 13]]
        dataTrans = [[1, 4, 7, 0, 11], [2, 5, 8, 0, 12], [3, 6, 9, 0, 13]]

        toTest = self.constructor(copy.deepcopy(data))
        dataObjOrig = toTest.copy()
        dataObjT = self.constructor(copy.deepcopy(dataTrans))

        assert dataObjT.isIdentical(toTest.T)
        assert toTest.isIdentical(dataObjOrig)


    def test_T_handmadeWithAxisNames(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]]
        dataTrans = [[1, 4, 7, 0], [2, 5, 8, 0], [3, 6, 9, 0]]

        origPointNames = ['1','2','3','4']
        origFeatureNames = ['a','b','c']
        transPointNames = origFeatureNames
        transFeatureNames = origPointNames

        toTest = self.constructor(copy.deepcopy(data), pointNames=origPointNames,
                                  featureNames=origFeatureNames)
        dataObjOrig = toTest.copy()
        dataObjT = self.constructor(copy.deepcopy(dataTrans), pointNames=transPointNames,
                                    featureNames=transFeatureNames)

        dotT = toTest.T
        assert dotT.points.getNames() == transPointNames
        assert dotT.features.getNames() == transFeatureNames
        assert dotT.isIdentical(dataObjT)
        assert toTest.isIdentical(dataObjOrig)


    def test_T_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0], [11, 12, 13]]

        dataObj = self.constructor(copy.deepcopy(data))
        dataObj._name = "TestName"
        if isinstance(dataObj, BaseView):
            dataObj._source._absPath = "TestAbsPath"
            dataObj._source._relPath = "TestRelPath"
        else:
            dataObj._absPath = "TestAbsPath"
            dataObj._relPath = "TestRelPath"

        dotT = dataObj.T
        assert dotT.name == "TestName"
        assert dotT.absolutePath == "TestAbsPath"
        assert dotT.relativePath == 'TestRelPath'

    ########
    # copy #
    ########
    @noLogEntryExpected
    def test_copy_withZeros(self):
        """ Test copy() produces an equal object and doesn't just copy the references """
        data1 = [[1, 2, 3, 0], [1, 0, 3, 0], [2, 4, 6, 0], [0, 0, 0, 0]]
        featureNames = ['one', 'two', 'three', 'four']
        pointNames = ['1', 'one', '2', '0']
        orig = self.constructor(data1, pointNames=pointNames, featureNames=featureNames)

        dup1 = orig.copy()
        dup2 = orig.copy(to=orig.getTypeString())

        assert orig.isIdentical(dup1)
        assert dup1.isIdentical(orig)

        assert orig.data is not dup1.data

        assert orig.isIdentical(dup2)
        assert dup2.isIdentical(orig)

        assert orig.data is not dup2.data

    @noLogEntryExpected
    def test_copy_Pempty(self):
        """ test copy() produces the correct outputs when given an point empty object """
        data = [[], []]
        data = numpy.array(data).T

        orig = self.constructor(data)
        sparseObj = createData(returnType="Sparse", data=data, useLog=False)
        listObj = createData(returnType="List", data=data, useLog=False)
        matixObj = createData(returnType="Matrix", data=data, useLog=False)
        dataframeObj = createData(returnType="DataFrame", data=data, useLog=False)

        copySparse = orig.copy(to='Sparse')
        assert copySparse.isIdentical(sparseObj)
        assert sparseObj.isIdentical(copySparse)

        copyList = orig.copy(to='List')
        assert copyList.isIdentical(listObj)
        assert listObj.isIdentical(copyList)

        copyMatrix = orig.copy(to='Matrix')
        assert copyMatrix.isIdentical(matixObj)
        assert matixObj.isIdentical(copyMatrix)

        copyDataFrame = orig.copy(to='DataFrame')
        assert copyDataFrame.isIdentical(copyDataFrame)
        assert dataframeObj.isIdentical(copyDataFrame)

        pyList = orig.copy(to='python list')
        assert pyList == []

        numpyArray = orig.copy(to='numpy array')
        assert numpy.array_equal(numpyArray, data)

        numpyMatrix = orig.copy(to='numpy matrix')
        assert numpy.array_equal(numpyMatrix, numpy.matrix(data))

        if scipy:
            scipyCsr = orig.copy(to='scipy csr')
            assert numpy.array_equal(cooMatrixToArray(scipyCsr), data)

            scipyCsc = orig.copy(to='scipy csc')
            assert numpy.array_equal(cooMatrixToArray(scipyCsc), data)

            scipyCoo = orig.copy(to='scipy coo')
            assert numpy.array_equal(cooMatrixToArray(scipyCoo), data)

        if pd:
            pandasDF = orig.copy(to='pandas dataframe')
            assert numpy.array_equal(pandasDF, data)

        listOfDict = orig.copy(to='list of dict')
        assert listOfDict == []

        dictOfList = orig.copy(to='dict of list')
        assert all(key.startswith(DEFAULT_PREFIX) for key in dictOfList.keys())
        assert all(val == [] for val in dictOfList.values())

    @noLogEntryExpected
    def test_copy_Fempty(self):
        """ test copy() produces the correct outputs when given an feature empty object """
        data = [[], []]
        data = numpy.array(data)

        orig = self.constructor(data)
        sparseObj = createData(returnType="Sparse", data=data, useLog=False)
        listObj = createData(returnType="List", data=data, useLog=False)
        matixObj = createData(returnType="Matrix", data=data, useLog=False)
        dataframeObj = createData(returnType="DataFrame", data=data, useLog=False)

        copySparse = orig.copy(to='Sparse')
        assert copySparse.isIdentical(sparseObj)
        assert sparseObj.isIdentical(copySparse)

        copyList = orig.copy(to='List')
        assert copyList.isIdentical(listObj)
        assert listObj.isIdentical(copyList)

        copyMatrix = orig.copy(to='Matrix')
        assert copyMatrix.isIdentical(matixObj)
        assert matixObj.isIdentical(copyMatrix)

        copyDataFrame = orig.copy(to='DataFrame')
        assert copyDataFrame.isIdentical(copyDataFrame)
        assert dataframeObj.isIdentical(copyDataFrame)

        pyList = orig.copy(to='python list')
        assert pyList == [[], []]

        numpyArray = orig.copy(to='numpy array')
        assert numpy.array_equal(numpyArray, data)

        numpyMatrix = orig.copy(to='numpy matrix')
        assert numpy.array_equal(numpyMatrix, numpy.matrix(data))

        if scipy:
            scipyCsr = orig.copy(to='scipy csr')
            assert numpy.array_equal(cooMatrixToArray(scipyCsr), data)

            scipyCsc = orig.copy(to='scipy csc')
            assert numpy.array_equal(cooMatrixToArray(scipyCsc), data)

            scipyCoo = orig.copy(to='scipy coo')
            assert numpy.array_equal(cooMatrixToArray(scipyCoo), data)

        if pd:
            pandasDF = orig.copy(to='pandas dataframe')
            assert numpy.array_equal(pandasDF, data)

        listOfDict = orig.copy(to='list of dict')
        assert listOfDict == [{}, {}]

        dictOfList = orig.copy(to='dict of list')
        assert dictOfList == {}

    @noLogEntryExpected
    def test_copy_Trueempty(self):
        """ test copy() produces the correct outputs when given a point and feature empty object """
        data = numpy.empty(shape=(0, 0))

        orig = self.constructor(data)
        sparseObj = createData(returnType="Sparse", data=data, useLog=False)
        listObj = createData(returnType="List", data=data, useLog=False)
        matixObj = createData(returnType="Matrix", data=data, useLog=False)
        dataframeObj = createData(returnType="DataFrame", data=data, useLog=False)

        copySparse = orig.copy(to='Sparse')
        assert copySparse.isIdentical(sparseObj)
        assert sparseObj.isIdentical(copySparse)

        copyList = orig.copy(to='List')
        assert copyList.isIdentical(listObj)
        assert listObj.isIdentical(copyList)

        copyMatrix = orig.copy(to='Matrix')
        assert copyMatrix.isIdentical(matixObj)
        assert matixObj.isIdentical(copyMatrix)

        copyDataFrame = orig.copy(to='DataFrame')
        assert copyDataFrame.isIdentical(copyDataFrame)
        assert dataframeObj.isIdentical(copyDataFrame)

        pyList = orig.copy(to='python list')
        assert pyList == []

        numpyArray = orig.copy(to='numpy array')
        assert numpy.array_equal(numpyArray, data)

        numpyMatrix = orig.copy(to='numpy matrix')
        assert numpy.array_equal(numpyMatrix, numpy.matrix(data))

        if scipy:
            scipyCsr = orig.copy(to='scipy csr')
            assert numpy.array_equal(cooMatrixToArray(scipyCsr), data)

            scipyCsc = orig.copy(to='scipy csc')
            assert numpy.array_equal(cooMatrixToArray(scipyCsc), data)

            scipyCoo = orig.copy(to='scipy coo')
            assert numpy.array_equal(cooMatrixToArray(scipyCoo), data)

        if pd:
            pandasDF = orig.copy(to='pandas dataframe')
            assert numpy.array_equal(pandasDF, data)

        listOfDict = orig.copy(to='list of dict')
        assert listOfDict == []

        dictOfList = orig.copy(to='dict of list')
        assert dictOfList == {}

    @noLogEntryExpected
    def test_copy_rightTypeTrueCopy(self):
        """ Test copy() will return all of the right type and do not show each other's modifications"""

        data = [[1, 2, 3], [1, 0, 3], [2, 4, 6], [0, 0, 0], ['a', 'b', 'c']]
        featureNames = ['one', 'two', 'three']
        pointNames = ['1', 'one', '2', '0', 'str']
        orig = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        sparseObj = createData(returnType="Sparse", data=data, pointNames=pointNames,
                               featureNames=featureNames, useLog=False)
        listObj = createData(returnType="List", data=data, pointNames=pointNames,
                             featureNames=featureNames, useLog=False)
        matixObj = createData(returnType="Matrix", data=data, pointNames=pointNames,
                              featureNames=featureNames, useLog=False)
        dataframeObj = createData(returnType="DataFrame", data=data,
                                  pointNames=pointNames, featureNames=featureNames,
                                  useLog=False)

        pointsShuffleIndices = [4, 3, 1, 2, 0]
        featuresShuffleIndices = [1, 2, 0]

        copySparse = orig.copy(to='Sparse')
        assert copySparse.isIdentical(sparseObj)
        assert sparseObj.isIdentical(copySparse)
        assert type(copySparse) == Sparse
        copySparse.features.setName('two', '2', useLog=False)
        copySparse.points.setName('one', 'WHAT', useLog=False)
        assert 'two' in orig.features.getNames()
        assert 'one' in orig.points.getNames()
        copySparse.points.sort(sortHelper=pointsShuffleIndices, useLog=False)
        copySparse.features.sort(sortHelper=featuresShuffleIndices, useLog=False)
        assert orig[0, 0] == 1

        copyList = orig.copy(to='List')
        assert copyList.isIdentical(listObj)
        assert listObj.isIdentical(copyList)
        assert type(copyList) == List
        copyList.features.setName('two', '2', useLog=False)
        copyList.points.setName('one', 'WHAT', useLog=False)
        assert 'two' in orig.features.getNames()
        assert 'one' in orig.points.getNames()
        copyList.points.sort(sortHelper=pointsShuffleIndices, useLog=False)
        copyList.features.sort(sortHelper=featuresShuffleIndices, useLog=False)
        assert orig[0, 0] == 1

        copyMatrix = orig.copy(to='Matrix')
        assert copyMatrix.isIdentical(matixObj)
        assert matixObj.isIdentical(copyMatrix)
        assert type(copyMatrix) == Matrix
        copyMatrix.features.setName('two', '2', useLog=False)
        copyMatrix.points.setName('one', 'WHAT', useLog=False)
        assert 'two' in orig.features.getNames()
        assert 'one' in orig.points.getNames()
        copyMatrix.points.sort(sortHelper=pointsShuffleIndices, useLog=False)
        copyMatrix.features.sort(sortHelper=featuresShuffleIndices, useLog=False)
        assert orig[0, 0] == 1

        copyDataFrame = orig.copy(to='DataFrame')
        assert copyDataFrame.isIdentical(dataframeObj)
        assert dataframeObj.isIdentical(copyDataFrame)
        assert type(copyDataFrame) == DataFrame
        copyDataFrame.features.setName('two', '2', useLog=False)
        copyDataFrame.points.setName('one', 'WHAT', useLog=False)
        assert 'two' in orig.features.getNames()
        assert 'one' in orig.points.getNames()
        copyDataFrame.points.sort(sortHelper=pointsShuffleIndices, useLog=False)
        copyDataFrame.features.sort(sortHelper=featuresShuffleIndices, useLog=False)
        assert orig[0, 0] == 1

        pyList = orig.copy(to='python list')
        assert type(pyList) == list
        pyList[0][0] = 5
        assert orig[0, 0] == 1

        numpyArray = orig.copy(to='numpy array')
        assert type(numpyArray) == type(numpy.array([]))
        numpyArray[0, 0] = 5
        assert orig[0, 0] == 1

        numpyMatrix = orig.copy(to='numpy matrix')
        assert type(numpyMatrix) == type(numpy.matrix([]))
        numpyMatrix[0, 0] = 5
        assert orig[0, 0] == 1

        if scipy:
            # copying to scipy requires numeric values only
            numeric = self.constructor(data[:4], pointNames=pointNames[:4],
                                       featureNames=featureNames)
            spcsc = numeric.copy(to='scipy csc')
            assert type(spcsc) == type(scipy.sparse.csc_matrix([]))
            spcsc[0, 0] = 5
            assert numeric[0, 0] == 1

            spcsr = numeric.copy(to='scipy csr')
            assert type(spcsr) == type(scipy.sparse.csr_matrix([]))
            spcsr[0, 0] = 5
            assert numeric[0, 0] == 1

            spcoo = numeric.copy(to='scipy coo')
            assert type(spcoo) == type(scipy.sparse.coo_matrix([]))
            spcoo.data[(spcoo.row == 0) & (spcoo.col == 0)] = 5
            assert numeric[0, 0] == 1

        if pd:
            pandasDF = orig.copy(to='pandas dataframe')
            assert type(pandasDF) == type(pd.DataFrame([]))
            pandasDF.iloc[0, 0] = 5
            assert orig[0, 0] == 1

        listOfDict = orig.copy(to='list of dict')
        assert type(listOfDict) == list
        assert type(listOfDict[0]) == dict
        listOfDict[0]['one'] = 5
        assert orig[0, 0] == 1

        dictOfList = orig.copy(to='dict of list')
        assert type(dictOfList) == dict
        assert type(dictOfList['one']) == list
        dictOfList['one'][0] = 5
        assert orig[0, 0] == 1

    @noLogEntryExpected
    def test_copy_rowsArePointsFalse(self):
        """ Test copy() will return data in the right places when rowsArePoints is False"""
        data = [[1, 2, 3], [1, 0, 3], [2, 4, 6], [0, 0, 0]]
        dataT = numpy.array(data).T

        featureNames = ['one', 'two', 'three']
        pointNames = ['1', 'one', '2', '0']
        orig = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        for retType in nimble.data.available:

            out = orig.copy(to=retType, rowsArePoints=False)
            desired = nimble.createData(retType, dataT,
                                        pointNames=featureNames,
                                        featureNames=pointNames, useLog=False)
            assert out == desired

        out = orig.copy(to='pythonlist', rowsArePoints=False)
        assert out == dataT.tolist()

        out = orig.copy(to='numpyarray', rowsArePoints=False)
        assert numpy.array_equal(out, dataT)

        out = orig.copy(to='numpymatrix', rowsArePoints=False)
        assert numpy.array_equal(out, dataT)

        if scipy:
            out = orig.copy(to='scipycsr', rowsArePoints=False)
            assert numpy.array_equal(cooMatrixToArray(out), dataT)

            out = orig.copy(to='scipycsc', rowsArePoints=False)
            assert numpy.array_equal(cooMatrixToArray(out), dataT)

            out = out = orig.copy(to='scipycoo', rowsArePoints=False)
            assert numpy.array_equal(cooMatrixToArray(out), dataT)

        if pd:
            out = orig.copy(to='pandasdataframe', rowsArePoints=False)
            assert numpy.array_equal(out, dataT)

        out = orig.copy(to='list of dict', rowsArePoints=False)

        desired = self.constructor(dataT, pointNames=featureNames, featureNames=pointNames)
        desired = desired.copy(to='list of dict')

        assert out == desired

        out = orig.copy(to='dict of list', rowsArePoints=False)

        desired = self.constructor(dataT, pointNames=featureNames, featureNames=pointNames)
        desired = desired.copy(to='dict of list')

        assert out == desired

    def test_copy_outputAs1DWrongFormat(self):
        """ Test copy will raise exception when given an unallowed format """
        data = [[1, 2, 3], [1, 0, 3], [2, 4, 6], [0, 0, 0]]
        featureNames = ['one', 'two', 'three']
        pointNames = ['1', 'one', '2', '0']
        orig = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        try:
            orig.copy(to="List", outputAs1D=True)
            assert False
        except InvalidArgumentValueCombination as ivc:
            pass
        try:
            orig.copy(to="Matrix", outputAs1D=True)
            assert False
        except InvalidArgumentValueCombination as ivc:
            pass
        try:
            orig.copy(to="Sparse", outputAs1D=True)
            assert False
        except InvalidArgumentValueCombination as ivc:
            pass
        try:
            orig.copy(to="numpy matrix", outputAs1D=True)
            assert False
        except InvalidArgumentValueCombination as ivc:
            pass
        if scipy:
            try:
                orig.copy(to="scipy csr", outputAs1D=True)
                assert False
            except InvalidArgumentValueCombination as ivc:
                pass
            try:
                orig.copy(to="scipy csc", outputAs1D=True)
                assert False
            except InvalidArgumentValueCombination as ivc:
                pass
            try:
                orig.copy(to="scipy coo", outputAs1D=True)
                assert False
            except InvalidArgumentValueCombination as ivc:
                pass
        if pd:
            try:
                orig.copy(to='pandas dataframe', outputAs1D=True)
            except InvalidArgumentValueCombination as ivc:
                pass
        try:
            orig.copy(to="list of dict", outputAs1D=True)
            assert False
        except InvalidArgumentValueCombination as ivc:
            pass
        try:
            orig.copy(to="dict of list", outputAs1D=True)
            assert False
        except InvalidArgumentValueCombination as ivc:
            pass

    @raises(ImproperObjectAction)
    def test_copy_outputAs1DWrongShape(self):
        """ Test copy will raise exception when given an unallowed shape """
        data = [[1, 2, 3], [1, 0, 3], [2, 4, 6], [0, 0, 0]]
        featureNames = ['one', 'two', 'three']
        pointNames = ['1', 'one', '2', '0']
        orig = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        orig.copy(to="numpy array", outputAs1D=True)

    @noLogEntryExpected
    def test_copy_outputAs1DTrue(self):
        """ Test copy() will return successfully output 1d for all allowable possibilities"""
        dataPv = [[1, 2, 0, 3]]
        dataFV = [[1], [2], [3], [0]]
        origPV = self.constructor(dataPv)
        origFV = self.constructor(dataFV)

        outPV = origPV.copy(to='python list', outputAs1D=True)
        assert outPV == [1, 2, 0, 3]

        outFV = origFV.copy(to='numpy array', outputAs1D=True)
        assert numpy.array_equal(outFV, numpy.array([1, 2, 3, 0]))

    def test_copy_NameAndPath(self):
        """ Test copy() will preserve name and path attributes"""

        data = [[1, 2, 3], [1, 0, 3], [2, 4, 6], [0, 0, 0]]
        name = 'copyTestName'
        orig = self.constructor(data)
        with tempfile.NamedTemporaryFile(suffix=".csv") as source:
            orig.writeFile(source.name, 'csv', includeNames=False)
            orig = self.constructor(source.name, name=name)
            path = source.name

        assert orig.name == name
        assert orig.path == path
        assert orig.absolutePath == path
        assert orig.relativePath == os.path.relpath(path)

        copySparse = orig.copy(to='Sparse')
        assert copySparse.name == orig.name
        assert copySparse.path == orig.path
        assert copySparse.absolutePath == path
        assert copySparse.relativePath == os.path.relpath(path)

        copyList = orig.copy(to='List')
        assert copyList.name == orig.name
        assert copyList.path == orig.path
        assert copyList.absolutePath == path
        assert copyList.relativePath == os.path.relpath(path)

        copyMatrix = orig.copy(to='Matrix')
        assert copyMatrix.name == orig.name
        assert copyMatrix.path == orig.path
        assert copyMatrix.absolutePath == path
        assert copyMatrix.relativePath == os.path.relpath(path)

        copyDataFrame = orig.copy(to='DataFrame')
        assert copyDataFrame.name == orig.name
        assert copyDataFrame.path == orig.path
        assert copyDataFrame.absolutePath == path
        assert copyDataFrame.relativePath == os.path.relpath(path)

    @raises(CalledFunctionException)
    @mock.patch('nimble.data.Base.copy', calledException)
    def test_copy__copy__(self):
        toTest = self.constructor([[1,2,],[3,4]], pointNames=['a', 'b'])
        ret = copy.copy(toTest)

    @raises(CalledFunctionException)
    @mock.patch('nimble.data.Base.copy', calledException)
    def test_copy__deepcopy__(self):
        toTest = self.constructor([[1,2,],[3,4]], pointNames=['a', 'b'])
        ret = copy.deepcopy(toTest)

    ###############
    # points.copy #
    ###############

    @raises(CalledFunctionException)
    @mock.patch('nimble.data.axis.constructIndicesList', calledException)
    def test_points_copy_calls_constructIndicesList(self):
        toTest = self.constructor([[1,2,],[3,4]], pointNames=['a', 'b'])

        ret = toTest.points.copy(['a', 'b'])

    @oneLogEntryExpected
    def test_points_copy_handmadeSingle(self):
        """ Test points.copy() against handmade output when copying one point """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        copy1 = toTest.points.copy(0)
        exp1 = self.constructor([[1, 2, 3]])
        assert copy1.isIdentical(exp1)
        expEnd = self.constructor(data)
        assert toTest.isIdentical(expEnd)

        # Check that names have not been generated unnecessarily
        assertNoNamesGenerated(toTest)
        assertNoNamesGenerated(copy1)

    def test_points_copy_index_NamePath_Preserve(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        # need to set source paths for view objects
        if isinstance(toTest, nimble.data.BaseView):
            toTest._source._absPath = 'testAbsPath'
            toTest._source._relPath = 'testRelPath'
        else:
            toTest._absPath = 'testAbsPath'
            toTest._relPath = 'testRelPath'
        toTest._name = 'testName'

        ext1 = toTest.points.copy(0)

        assert ext1.nameIsDefault()
        assert ext1.path == 'testAbsPath'
        assert ext1.absolutePath == 'testAbsPath'
        assert ext1.relativePath == 'testRelPath'

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

    def test_points_copy_ListIntoPEmpty(self):
        """ Test points.copy() by copying a list of all points """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        expRet = self.constructor(data)
        expTest = toTest.copy()
        ret = toTest.points.copy([0, 1, 2, 3])

        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(expTest)

    @twoLogEntriesExpected
    def test_points_copy_handmadeListSequence(self):
        """ Test points.copy() against handmade output for multiple copies """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        names = ['1', '4', '7', '10']
        toTest = self.constructor(data, pointNames=names)
        ext1 = toTest.points.copy('1')
        exp1 = self.constructor([[1, 2, 3]], pointNames=['1'])
        assert ext1.isIdentical(exp1)
        ext2 = toTest.points.copy([1, 2])
        exp2 = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=['4', '7'])
        assert ext2.isIdentical(exp2)
        expEnd = self.constructor(data, pointNames=names)
        assert toTest.isIdentical(expEnd)

    def test_points_copy_handmadeListOrdering(self):
        """ Test points.copy() against handmade output for out of order copying """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
        names = ['1', '4', '7', '10', '13']
        toTest = self.constructor(data, pointNames=names)
        ext1 = toTest.points.copy([3, 4, 1])
        exp1 = self.constructor([[10, 11, 12], [13, 14, 15], [4, 5, 6]], pointNames=['10', '13', '4'])
        assert ext1.isIdentical(exp1)
        expEnd = self.constructor(data, pointNames=names)
        assert toTest.isIdentical(expEnd)

    def test_points_copy_List_trickyOrdering(self):
        data = [[0], [2], [2], [2], [0], [0], [0], [0], [2], [0]]
        toCopy = [6, 5, 3, 9]

        toTest = self.constructor(data)

        ret = toTest.points.copy(toCopy)

        expRaw = [[0], [0], [2], [0]]
        expRet = self.constructor(expRaw)

        expTest = self.constructor(data)

        assert ret == expRet
        assert toTest == expTest

    def test_points_copy_function_selectionGap(self):
        data = [[0], [2], [2], [2], [0], [0], [0], [0], [2], [0]]
        copyIndices = [3, 5, 6, 9]
        pnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        def sel(point):
            if int(point.points.getName(0)) in copyIndices:
                return True
            else:
                return False

        toTest = self.constructor(data, pointNames=pnames)
        ret = toTest.points.copy(sel)

        expRaw = [[2], [0], [0], [0]]
        expNames = ['3', '5', '6', '9']
        expRet = self.constructor(expRaw, pointNames=expNames)
        expTest = self.constructor(data, pointNames=pnames)
        assert ret == expRet
        assert toTest == expTest


    def test_points_copy_functionIntoPEmpty(self):
        """ Test points.copy() by copying all points using a function """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        expRet = self.constructor(data)

        ret = toTest.points.copy(allTrue)
        expTest = self.constructor(data)

        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(expTest)

    def test_points_copy_function_returnPointEmpty(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        expTest = self.constructor(data)

        ret = toTest.points.copy(allFalse)

        data = [[], [], []]
        data = numpy.array(data).T
        expRet = self.constructor(data)

        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(expTest)

    def test_points_copy_handmadeFunction(self):
        """ Test points.copy() against handmade output for function copying """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        ext = toTest.points.copy(oneOrFour)
        exp = self.constructor([[1, 2, 3], [4, 5, 6]])
        assert ext.isIdentical(exp)
        expEnd = self.constructor(data)
        assert toTest.isIdentical(expEnd)

    def test_points_copy_func_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        # need to set source paths for view objects
        if isinstance(toTest, nimble.data.BaseView):
            toTest._source._absPath = 'testAbsPath'
            toTest._source._relPath = 'testRelPath'
        else:
            toTest._absPath = 'testAbsPath'
            toTest._relPath = 'testRelPath'
        toTest._name = 'testName'

        ext = toTest.points.copy(oneOrFour)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

        assert ext.nameIsDefault()
        assert ext.absolutePath == 'testAbsPath'
        assert ext.relativePath == 'testRelPath'

    def test_points_copy_handmadeFuncionWithFeatureNames(self):
        """ Test points.copy() against handmade output for function copying with featureNames"""
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)

        ext = toTest.points.copy(oneOrFour)
        exp = self.constructor([[1, 2, 3], [4, 5, 6]], featureNames=featureNames)
        assert ext.isIdentical(exp)
        expEnd = self.constructor(data, featureNames=featureNames)
        assert toTest.isIdentical(expEnd)

    @raises(InvalidArgumentType)
    def test_points_copy_exceptionStartInvalidType(self):
        """ Test points.copy() for InvalidArgumentType when start is not a valid ID type"""
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.copy(start=1.1, end=2)

    @raises(IndexError)
    def test_points_copy_exceptionEndInvalid(self):
        """ Test points.copy() for IndexError when end is not a valid Point index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.copy(start=1, end=5)

    @raises(InvalidArgumentValueCombination)
    def test_points_copy_exceptionInversion(self):
        """ Test points.copy() for InvalidArgumentValueCombination when start comes after end """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.copy(start=2, end=0)

    @raises(InvalidArgumentValueCombination)
    def test_points_copy_exceptionInversionPointName(self):
        """ Test points.copy() for InvalidArgumentValueCombination when start comes after end as FeatureNames"""
        pointNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames)
        toTest.points.copy(start="two", end="one")

    @raises(InvalidArgumentValue)
    def test_points_copy_exceptionDuplicates(self):
        """ Test points.copy() for InvalidArgumentValueCombination when toCopy contains duplicates """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.points.copy([0, 1, 0])

    def test_points_copy_handmadeRange(self):
        """ Test points.copy() against handmade output for range copying """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.points.copy(start=1, end=2)

        expectedRet = self.constructor([[4, 5, 6], [7, 8, 9]])
        expectedTest = self.constructor(data)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_points_copy_range_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        # need to set source paths for view objects
        if isinstance(toTest, nimble.data.BaseView):
            toTest._source._absPath = 'testAbsPath'
            toTest._source._relPath = 'testRelPath'
        else:
            toTest._absPath = 'testAbsPath'
            toTest._relPath = 'testRelPath'
        toTest._name = 'testName'

        ret = toTest.points.copy(start=1, end=2)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

        assert ret.nameIsDefault()
        assert ret.absolutePath == 'testAbsPath'
        assert ret.relativePath == 'testRelPath'


    def test_points_copy_rangeIntoPEmpty(self):
        """ Test points.copy() copies all points using ranges """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy(start=0, end=2)

        assert ret.isIdentical(expRet)

        expTest = self.constructor(data)

        toTest.isIdentical(expTest)


    def test_points_copy_handmadeRangeWithFeatureNames(self):
        """ Test points.copy() against handmade output for range copying with featureNames """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy(start=1, end=2)

        expectedRet = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=['4', '7'], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)


    def test_points_copy_handmadeRangeRand_FM(self):
        """ Test points.copy() for correct sizes when using randomized range and featureNames """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.points.copy(start=0, end=2, number=2, randomize=True)

        assert len(ret.points) == 2
        assert len(toTest.points) == 3


    def test_points_copy_handmadeRangeDefaults(self):
        """ Test points.copy() uses the correct defaults in the case of range based copy """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy(end=1)

        expectedRet = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=['1', '4'], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy(start=1)

        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expectedRet = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=['4', '7'], featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_points_copy_handmade_calling_pointNames(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy(start='4', end='7')

        expectedRet = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_points_copy_handmadeString(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('one=1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('one==1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('one<2')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('one<=1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('one>4')
        expectedRet = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('one>=7')
        expectedRet = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('one!=4')
        expectedRet = self.constructor([[1, 2, 3], [7, 8, 9]], pointNames=[pointNames[0], pointNames[-1]],
                                       featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('one<1')
        expectedRet = self.constructor([], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('one>0')
        expectedRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_points_copy_handmadeStringWithOperatorWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('one = 1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('one == 1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('one < 2')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('one <= 1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('one > 4')
        expectedRet = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('one >= 7')
        expectedRet = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('one != 4')
        expectedRet = self.constructor([[1, 2, 3], [7, 8, 9]], pointNames=[pointNames[0], pointNames[-1]],
                                       featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('one < 1')
        expectedRet = self.constructor([], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('one > 0')
        expectedRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_points_copy_handmadeStringWithFeatureWhitespace(self):
        featureNames = ["feature one", "feature two", "feature three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('feature one=1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName=value with operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('feature one = 1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_points_copy_list_mixed(self):
        """ Test points.copy() list input with mixed names and indices """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        names = ['1', '4', '7', '10']
        toTest = self.constructor(data, pointNames=names)
        ret = toTest.points.copy(['1',1,-1])
        expRet = self.constructor([[1, 2, 3], [4, 5, 6], [10, 11, 12]], pointNames=['1','4','10'])
        expTest = self.constructor(data, pointNames=names)
        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(expTest)

    @raises(InvalidArgumentValue)
    def test_points_copy_handmadeString_featureNotExist(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.copy('four=1')

    def test_points_copy_numberOnly(self):
        self.back_copy_numberOnly('point')

    def test_points_copy_functionAndNumber(self):
        self.back_copy_functionAndNumber('point')

    def test_points_copy_numberAndRandomizeAllData(self):
        self.back_copy_numberAndRandomizeAllData('point')

    def test_points_copy_numberAndRandomizeSelectedData(self):
        self.back_copy_numberAndRandomizeSelectedData('point')

    @raises(InvalidArgumentValueCombination)
    def test_points_copy_randomizeNoNumber(self):
        self.back_structural_randomizeNoNumber('copy', 'point')

    @raises(InvalidArgumentValue)
    def test_points_copy_list_numberGreaterThanTargeted(self):
        self.back_structural_list_numberGreaterThanTargeted('copy', 'point')

    @raises(InvalidArgumentValue)
    def test_points_copy_function_numberGreaterThanTargeted(self):
        self.back_structural_function_numberGreaterThanTargeted('copy', 'point')

    @raises(InvalidArgumentValue)
    def test_points_copy_range_numberGreaterThanTargeted(self):
        self.back_structural_range_numberGreaterThanTargeted('copy', 'point')

    ### using match module ###

    def test_points_copy_match_missing(self):
        data = [[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=['a', 'b', 'c'])
        expTest = self.constructor(data, featureNames=['a', 'b', 'c'])
        ret = toTest.points.copy(match.anyMissing)
        expRet = self.constructor([[None, 11, None], [7, 11, None]], featureNames=['a', 'b', 'c'])
        assert toTest == expTest
        assert ret == expRet

        data = [[None, None, None], [None, 11, None], [7, 11, None], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=['a', 'b', 'c'])
        expTest = self.constructor(data, featureNames=['a', 'b', 'c'])
        ret = toTest.points.copy(match.allMissing)
        expRet = self.constructor([[None, None, None]], featureNames=['a', 'b', 'c'])
        assert toTest == expTest
        assert ret == expRet

    def test_points_copy_match_nonNumeric(self):
        data = [[1, 2, 3], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=['a', 'b', 'c'])
        expTest = self.constructor(data, featureNames=['a', 'b', 'c'])
        ret = toTest.points.copy(match.anyNonNumeric)
        expRet = self.constructor([['a', 11, 'c'], [7, 11, 'c']], featureNames=['a', 'b', 'c'])
        assert toTest == expTest
        assert ret == expRet

        data = [['a', 'x', 'c'], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=['a', 'b', 'c'])
        expTest = self.constructor(data, featureNames=['a', 'b', 'c'])
        ret = toTest.points.copy(match.allNonNumeric)
        expRet = self.constructor([['a', 'x', 'c']], featureNames=['a', 'b', 'c'])
        assert toTest == expTest
        assert ret == expRet

    def test_points_copy_match_list(self):
        data = [[1, 2, 3], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=['a', 'b', 'c'])
        expTest = self.constructor(data, featureNames=['a', 'b', 'c'])
        ret = toTest.points.copy(match.anyValues(['a', 'c', 'x']))
        expRet = self.constructor([['a', 11, 'c'], [7, 11, 'c']], featureNames=['a', 'b', 'c'])
        assert toTest == expTest
        assert ret == expRet

        data = [['a', 'x', 'c'], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=['a', 'b', 'c'])
        expTest = self.constructor(data, featureNames=['a', 'b', 'c'])
        ret = toTest.points.copy(match.allValues(['a', 'c', 'x']))
        expRet = self.constructor([['a', 'x', 'c']], featureNames=['a', 'b', 'c'])
        assert toTest == expTest
        assert ret == expRet

    def test_points_copy_match_function(self):
        data = [[1, 2, 3], [-1, 11, -3], [7, 11, -3], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=['a', 'b', 'c'])
        expTest = self.constructor(data, featureNames=['a', 'b', 'c'])
        ret = toTest.points.copy(match.anyValues(lambda x: x < 0))
        expRet = self.constructor([[-1, 11, -3], [7, 11, -3]], featureNames=['a', 'b', 'c'])
        assert toTest == expTest
        assert ret == expRet

        data = [[-1, -2, -3], [-1, 11, -3], [7, 11, -3], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=['a', 'b', 'c'])
        expTest = self.constructor(data, featureNames=['a', 'b', 'c'])
        ret = toTest.points.copy(match.allValues(lambda x: x < 0))
        expRet = self.constructor([[-1, -2, -3]], featureNames=['a', 'b', 'c'])
        assert toTest == expTest
        assert ret == expRet

    #######################
    # copy common backend #
    #######################

    def back_copy_numberOnly(self, axis):
        if axis == 'point':
            toCall = "points"
        else:
            toCall = "features"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        ret = getattr(toTest, toCall).copy(number=3)
        if axis == 'point':
            exp = self.constructor(data[:3], pointNames=pnames[:3], featureNames=fnames)
            rem = self.constructor(data, pointNames=pnames, featureNames=fnames)
        else:
            exp = self.constructor([p[:3] for p in data], pointNames=pnames, featureNames=fnames[:3])
            rem = self.constructor(data, pointNames=pnames, featureNames=fnames)

        assert exp.isIdentical(ret)
        assert rem.isIdentical(toTest)

    def back_copy_functionAndNumber(self, axis):
        if axis == 'point':
            toCall = "points"
        else:
            toCall = "features"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        ret = getattr(toTest, toCall).copy(allTrue, number=2)
        if axis == 'point':
            exp = self.constructor(data[:2], pointNames=pnames[:2], featureNames=fnames)
            rem = self.constructor(data, pointNames=pnames, featureNames=fnames)
        else:
            exp = self.constructor([p[:2] for p in data], pointNames=pnames, featureNames=fnames[:2])
            rem = self.constructor(data, pointNames=pnames, featureNames=fnames)

        assert exp.isIdentical(ret)
        assert rem.isIdentical(toTest)

    def back_copy_numberAndRandomizeAllData(self, axis):
        """test that randomizing (with same randomly chosen seed) and limiting to a
        given number provides the same result for all input types if using all the data
        """
        if axis == 'point':
            toCall = "points"
        else:
            toCall = "features"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest1 = self.constructor(data, pointNames=pnames, featureNames=fnames)
        toTest2 = toTest1.copy()
        toTest3 = toTest1.copy()
        toTest4 = toTest1.copy()
        expTest = toTest1.copy()

        seed = nimble.randomness.generateSubsidiarySeed()
        nimble.randomness.startAlternateControl(seed)
        ret = getattr(toTest1, toCall).copy(number=3, randomize=True)
        nimble.randomness.endAlternateControl()

        nimble.randomness.startAlternateControl(seed)
        retList = getattr(toTest2, toCall).copy([0, 1, 2, 3], number=3, randomize=True)
        nimble.randomness.endAlternateControl()

        nimble.randomness.startAlternateControl(seed)
        retRange = getattr(toTest3, toCall).copy(start=0, end=3, number=3, randomize=True)
        nimble.randomness.endAlternateControl()

        nimble.randomness.startAlternateControl(seed)
        retFunc = getattr(toTest4, toCall).copy(allTrue, number=3, randomize=True)
        nimble.randomness.endAlternateControl()

        if axis == 'point':
            assert len(ret.points) == 3
        else:
            assert len(ret.features) == 3

        assert ret.isIdentical(retList)
        assert ret.isIdentical(retRange)
        assert ret.isIdentical(retFunc)

        assert toTest1.isIdentical(expTest)
        assert toTest2.isIdentical(expTest)
        assert toTest3.isIdentical(expTest)
        assert toTest4.isIdentical(expTest)

    def back_copy_numberAndRandomizeSelectedData(self, axis):
        """test that randomization occurs after the data has been selected from the user inputs """
        if axis == 'point':
            toCall = "points"
        else:
            toCall = "features"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest1 = self.constructor(data, pointNames=pnames, featureNames=fnames)
        toTest2 = toTest1.copy()
        toTest3 = toTest1.copy()
        if axis == 'point':
            exp1 = toTest1[1, :]
            exp2 = toTest1[2, :]
        else:
            exp1 = toTest1[:, 1]
            exp2 = toTest1[:, 2]

        seed = nimble.randomness.generateSubsidiarySeed()
        nimble.randomness.startAlternateControl(seed)
        retList = getattr(toTest1, toCall).copy([1, 2], number=1, randomize=True)
        nimble.randomness.endAlternateControl()

        nimble.randomness.startAlternateControl(seed)
        retRange = getattr(toTest2, toCall).copy(start=1, end=2, number=1, randomize=True)
        nimble.randomness.endAlternateControl()

        def middleRowsOrCols(value):
            return value[0] in [2, 4, 5, 7]

        nimble.randomness.startAlternateControl(seed)
        retFunc = getattr(toTest3, toCall).copy(middleRowsOrCols, number=1, randomize=True)
        nimble.randomness.endAlternateControl()

        assert retList.isIdentical(exp1) or retList.isIdentical(exp2)
        assert retRange.isIdentical(exp1) or retList.isIdentical(exp2)
        assert retFunc.isIdentical(exp1) or retList.isIdentical(exp2)

    #####################
    # features_copy #
    #####################

    @raises(CalledFunctionException)
    @mock.patch('nimble.data.axis.constructIndicesList', calledException)
    def test_features_copy_calls_constructIndicesList(self):
        toTest = self.constructor([[1,2,],[3,4]], featureNames=['a', 'b'])

        ret = toTest.features.copy(['a', 'b'])

    @oneLogEntryExpected
    def test_features_copy_handmadeSingle(self):
        """ Test features.copy() against handmade output when copying one feature """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        copy1 = toTest.features.copy(0)
        exp1 = self.constructor([[1], [4], [7]])

        assert copy1.isIdentical(exp1)
        expEnd = self.constructor(data)
        assert toTest.isIdentical(expEnd)

        # Check that names have not been generated unnecessarily
        assertNoNamesGenerated(toTest)
        assertNoNamesGenerated(copy1)

    def test_features_copy_List_NamePath_Preserve(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        # need to set source paths for view objects
        if isinstance(toTest, nimble.data.BaseView):
            toTest._source._absPath = 'testAbsPath'
            toTest._source._relPath = 'testRelPath'
        else:
            toTest._absPath = 'testAbsPath'
            toTest._relPath = 'testRelPath'
        toTest._name = 'testName'

        ext1 = toTest.features.copy(0)

        assert toTest.path == 'testAbsPath'
        assert toTest.absolutePath == 'testAbsPath'
        assert toTest.relativePath == 'testRelPath'

        assert ext1.nameIsDefault()
        assert ext1.absolutePath == 'testAbsPath'
        assert ext1.relativePath == 'testRelPath'

    def test_features_copy_ListIntoFEmpty(self):
        """ Test features.copy() by copying a list of all features """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        expRet = self.constructor(data)
        ret = toTest.features.copy([0, 1, 2])

        assert ret.isIdentical(expRet)
        expEnd = self.constructor(data)
        assert toTest.isIdentical(expEnd)

    def test_features_copy_ListIntoFEmptyOutOfOrder(self):
        """ Test features.copy() by copying a list of all features """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        expData = [[3, 1, 2], [6, 4, 5], [9, 7, 8], [12, 10, 11]]
        expRet = self.constructor(expData)
        ret = toTest.features.copy([2, 0, 1])

        assert ret.isIdentical(expRet)
        expEnd = self.constructor(data)
        assert toTest.isIdentical(expEnd)

    @twoLogEntriesExpected
    def test_features_copy_handmadeListSequence(self):
        """ Test features.copy() against handmade output for several copies by list """
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data, pointNames=pointNames)
        ext1 = toTest.features.copy([0])
        exp1 = self.constructor([[1], [4], [7]], pointNames=pointNames)
        assert ext1.isIdentical(exp1)
        ext2 = toTest.features.copy([3, 2])
        exp2 = self.constructor([[-1, 3], [-2, 6], [-3, 9]], pointNames=pointNames)
        assert ext2.isIdentical(exp2)
        expEnd = self.constructor(data, pointNames=pointNames)
        assert toTest.isIdentical(expEnd)

    def test_features_copy_handmadeListWithFeatureName(self):
        """ Test features.copy() against handmade output for list copies when specifying featureNames """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        toTest = self.constructor(data, featureNames=featureNames)
        ext1 = toTest.features.copy(["one"])
        exp1 = self.constructor([[1], [4], [7]], featureNames=["one"])
        assert ext1.isIdentical(exp1)
        ext2 = toTest.features.copy(["three", "neg"])
        exp2 = self.constructor([[3, -1], [6, -2], [9, -3]], featureNames=["three", "neg"])
        assert ext2.isIdentical(exp2)
        expEnd = self.constructor(data, featureNames=featureNames)
        assert toTest.isIdentical(expEnd)


    def test_features_copy_List_trickyOrdering(self):
        data = [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
        toCopy = [6, 5, 3, 9]
        #		toCopy = [3,5,6,9]

        toTest = self.constructor(data)

        ret = toTest.features.copy(toCopy)

        expRaw = [0, 0, 1, 0]
        expRet = self.constructor(expRaw)

        expRem = self.constructor(data)

        assert ret == expRet
        assert toTest == expRem

    def test_features_copy_List_reorderingWithFeatureNames(self):
        data = [[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12]]
        fnames = ['a', 'b', 'c', 'd']
        test = self.constructor(data, featureNames=fnames)

        expRetRaw = [[1, 3, 2], [4, 6, 5], [7, 9, 8]]
        expRetNames = ['a', 'c', 'b']
        expRet = self.constructor(expRetRaw, featureNames=expRetNames)

        expTestRaw = [[10], [11], [12]]
        expTestNames = ['d']
        expTest = self.constructor(data, featureNames=fnames)

        ret = test.features.copy(expRetNames)
        assert ret == expRet
        assert test == expTest


    def test_features_copy_function_selectionGap(self):
        data = [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
        fnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        copyIndices = [3, 5, 6, 9]

        def sel(feature):
            if int(feature.features.getName(0)) in copyIndices:
                return True
            else:
                return False

        toTest = self.constructor(data, featureNames=fnames)

        ret = toTest.features.copy(sel)

        expRaw = [1, 0, 0, 0]
        expNames = ['3', '5', '6', '9']
        expRet = self.constructor(expRaw, featureNames=expNames)

        expRaw = [0, 1, 1, 0, 0, 1]
        expNames = ['0', '1', '2', '4', '7', '8']
        expRem = self.constructor(data, featureNames=fnames)

        assert ret == expRet
        assert toTest == expRem


    def test_features_copy_functionIntoFEmpty(self):
        """ Test features.copy() by copying all featuress using a function """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        expRet = self.constructor(data)

        ret = toTest.features.copy(allTrue)
        assert ret.isIdentical(expRet)
        expEnd = self.constructor(data)
        assert toTest.isIdentical(expEnd)

    def test_features_copy_function_returnPointEmpty(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        exp = self.constructor(data)

        ret = toTest.features.copy(allFalse)
        expRet = self.constructor([[],[],[]])
        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(exp)


    def test_features_copy_handmadeFunction(self):
        """ Test features.copy() against handmade output for function copies """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data)

        ext = toTest.features.copy(absoluteOne)
        exp = self.constructor([[1, -1], [4, -2], [7, -3]])
        assert ext.isIdentical(exp)
        expEnd = self.constructor(data)
        assert toTest.isIdentical(expEnd)


    def test_features_copy_func_NamePath_preservation(self):
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data)

        # need to set source paths for view objects
        if isinstance(toTest, nimble.data.BaseView):
            toTest._source._absPath = 'testAbsPath'
            toTest._source._relPath = 'testRelPath'
        else:
            toTest._absPath = 'testAbsPath'
            toTest._relPath = 'testRelPath'
        toTest._name = 'testName'

        ext = toTest.features.copy(absoluteOne)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

        assert ext.nameIsDefault()
        assert ext.absolutePath == 'testAbsPath'
        assert ext.relativePath == 'testRelPath'

    def test_features_copy_handmadeFunctionWithFeatureName(self):
        """ Test features.copy() against handmade output for function copies with featureNames """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        pointNames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        ext = toTest.features.copy(absoluteOne)
        exp = self.constructor([[1, -1], [4, -2], [7, -3]], pointNames=pointNames, featureNames=['one', 'neg'])
        assert ext.isIdentical(exp)
        expEnd = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert toTest.isIdentical(expEnd)

    @raises(InvalidArgumentType)
    def test_features_copy_exceptionStartInvalidType(self):
        """ Test features.copy() for InvalidArgumentType when start is not a valid ID type"""
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.copy(start=1.1, end=2)

    @raises(KeyError)
    def test_features_copy_exceptionStartInvalidFeatureName(self):
        """ Test features.copy() for KeyError when start is not a valid feature FeatureName """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.copy(start="wrong", end=2)

    @raises(IndexError)
    def test_features_copy_exceptionEndInvalid(self):
        """ Test features.copy() for IndexError when end is not a valid feature index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.copy(start=0, end=5)

    @raises(KeyError)
    def test_features_copy_exceptionEndInvalidFeatureName(self):
        """ Test features.copy() for KeyError when end is not a valid featureName """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.copy(start="two", end="five")

    @raises(InvalidArgumentValueCombination)
    def test_features_copy_exceptionInversion(self):
        """ Test features.copy() for InvalidArgumentValueCombination when start comes after end """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.copy(start=2, end=0)

    @raises(InvalidArgumentValueCombination)
    def test_features_copy_exceptionInversionFeatureName(self):
        """ Test features.copy() for InvalidArgumentValueCombination when start comes after end as FeatureNames"""
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.copy(start="two", end="one")

    @raises(InvalidArgumentValue)
    def test_features_copy_exceptionDuplicates(self):
        """ Test points.copy() for InvalidArgumentValueCombination when toCopy contains duplicates """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.features.copy([0, 1, 0])

    def test_features_copy_rangeIntoFEmpty(self):
        """ Test features.copy() copies all Featuress using ranges """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        expRet = self.constructor(data, featureNames=featureNames)
        ret = toTest.features.copy(start=0, end=2)

        assert ret.isIdentical(expRet)
        exp = self.constructor(data, featureNames=featureNames)
        toTest.isIdentical(exp)

    def test_features_copy_handmadeRange(self):
        """ Test features.copy() against handmade output for range copies """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.features.copy(start=1, end=2)

        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]])
        expectedTest = self.constructor(data)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_features_copy_range_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        # need to set source paths for view objects
        if isinstance(toTest, nimble.data.BaseView):
            toTest._source._absPath = 'testAbsPath'
            toTest._source._relPath = 'testRelPath'
        else:
            toTest._absPath = 'testAbsPath'
            toTest._relPath = 'testRelPath'
        toTest._name = 'testName'

        ret = toTest.features.copy(start=1, end=2)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

        assert ret.nameIsDefault()
        assert ret.absolutePath == 'testAbsPath'
        assert ret.relativePath == 'testRelPath'


    def test_features_copy_handmadeWithFeatureNames(self):
        """ Test features.copy() against handmade output for range copies with FeatureNames """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy(start=1, end=2)

        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=["two", "three"])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_features_copy_handmade_calling_featureNames(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy(start="two", end="three")

        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=["two", "three"])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_features_copy_handmadeString(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['p1', 'p2', 'p3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('p2=5')
        expectedRet = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('p1==1')
        expectedRet = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('p3<9')
        expectedRet = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('p3<=8')
        expectedRet = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('p3>8')
        expectedRet = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('p3>8.5')
        expectedRet = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('p1!=1.0')
        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('p1<1')
        expectedRet = self.constructor([], pointNames=pointNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('p1>0')
        expectedRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_features_copy_handmadeStringWithOperatorWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['p1', 'p2', 'p3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('p2 = 5')
        expectedRet = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('p1 == 1')
        expectedRet = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('p3 < 9')
        expectedRet = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('p3 <= 8')
        expectedRet = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('p3 > 8')
        expectedRet = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('p3 > 8.5')
        expectedRet = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('p1 != 1.0')
        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('p1 < 1')
        expectedRet = self.constructor([], pointNames=pointNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('p1 > 0')
        expectedRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_features_copy_handmadeStringWithPointWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['pt 1', 'pt 2', 'pt 3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value with no operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('pt 2=5')
        expectedRet = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test pointName=value with operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('pt 2 = 5')
        expectedRet = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_features_copy_list_mixed(self):
        """ Test features.copy() list input with mixed names and indices """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.features.copy([1, "three", -1])
        expRet = self.constructor([[2, 3, -1], [5, 6, -2], [8, 9, -3]], featureNames=["two", "three", "neg"])
        expTest = self.constructor(data, featureNames=featureNames)
        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(expTest)

    @raises(InvalidArgumentValue)
    def test_features_copy_handmadeString_pointNotExist(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.copy('5=1')

    def test_features_copy_numberOnly(self):
        self.back_copy_numberOnly('feature')

    def test_features_copy_functionAndNumber(self):
        self.back_copy_functionAndNumber('feature')

    def test_features_copy_numberAndRandomizeAllData(self):
        self.back_copy_numberAndRandomizeAllData('feature')

    def test_features_copy_numberAndRandomizeSelectedData(self):
        self.back_copy_numberAndRandomizeSelectedData('feature')

    @raises(InvalidArgumentValueCombination)
    def test_features_copy_randomizeNoNumber(self):
        self.back_structural_randomizeNoNumber('copy', 'feature')

    @raises(InvalidArgumentValue)
    def test_features_copy_list_numberGreaterThanTargeted(self):
        self.back_structural_list_numberGreaterThanTargeted('copy', 'feature')

    @raises(InvalidArgumentValue)
    def test_features_copy_function_numberGreaterThanTargeted(self):
        self.back_structural_function_numberGreaterThanTargeted('copy', 'feature')

    @raises(InvalidArgumentValue)
    def test_features_copy_range_numberGreaterThanTargeted(self):
        self.back_structural_range_numberGreaterThanTargeted('copy', 'feature')

    ### using match module ###

    def test_features_copy_match_missing(self):
        toTest = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        expTest = toTest.copy()
        ret = toTest.features.copy(match.anyMissing)
        expRet = self.constructor([[1, 3], [None, None], [7, None], [7, 9]], featureNames=['a', 'c'])
        assert toTest == expTest
        assert ret == expRet

        toTest = self.constructor([[1, 2, None], [None, 11, None], [7, 11, None], [7, 8, None]], featureNames=['a', 'b', 'c'])
        expTest = toTest.copy()
        ret = toTest.features.copy(match.allMissing)
        expRet = self.constructor([[None], [None], [None], [None]], featureNames=['c'])
        assert toTest == expTest
        assert ret == expRet

    def test_features_copy_match_nonNumeric(self):
        toTest = self.constructor([[1, 2, 3], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        expTest = toTest.copy()
        ret = toTest.features.copy(match.anyNonNumeric)
        expRet = self.constructor([[1, 3], ['a', 'c'], [7, 'c'], [7, 9]], featureNames=['a', 'c'])
        assert toTest == expTest
        assert ret == expRet

        toTest = self.constructor([[1, 2, 'c'], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 'c']], featureNames=['a', 'b', 'c'])
        expTest = toTest.copy()
        ret = toTest.features.copy(match.allNonNumeric)
        expRet = self.constructor([['c'], ['c'], ['c'], ['c']], featureNames=['c'])
        assert toTest == expTest
        assert ret == expRet

    def test_features_copy_match_list(self):
        toTest = self.constructor([[1, 2, 3], ['a', 11, 'c'], ['x', 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        expTest = toTest.copy()
        ret = toTest.features.copy(match.anyValues(['a', 'c', 'x']))
        expRet = self.constructor([[1, 3], ['a', 'c'], ['x', 'c'], [7, 9]], featureNames=['a', 'c'])
        assert toTest == expTest
        assert ret == expRet

        toTest = self.constructor([[1, 2, 'c'], ['a', 11, 'c'], ['x', 11, 'c'], [7, 8, 'c']], featureNames=['a', 'b', 'c'])
        expTest = toTest.copy()
        ret = toTest.features.copy(match.allValues(['a', 'c', 'x']))
        expRet = self.constructor([['c'], ['c'], ['c'], ['c']], featureNames=['c'])
        assert toTest == expTest
        assert ret == expRet

    def test_features_copy_match_function(self):
        toTest = self.constructor([[1, 2, 3], [-1, 11, -3], [-1, 11, -1], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        expTest = toTest.copy()
        ret = toTest.features.copy(match.anyValues(lambda x: x < 0))
        expRet = self.constructor([[1, 3], [-1, -3], [-1, -1], [7, 9]], featureNames=['a', 'c'])
        assert toTest == expTest
        assert ret == expRet

        toTest = self.constructor([[1, 2, -3], [-1, 11, -3], [-1, 11, -3], [7, 8, -3]], featureNames=['a', 'b', 'c'])
        expTest = toTest.copy()
        ret = toTest.features.copy(match.allValues(lambda x: x < 0))
        expRet = self.constructor([[-3], [-3], [-3], [-3]], featureNames=['c'])
        assert toTest == expTest
        assert ret == expRet

class StructureModifying(StructureShared):
    ##############
    # create data
    ##############

    def test_createEmptyData1(self):
        """
        create data object using tuple, list,
        dict, numpy.ndarray, numpy.matrix, pd.DataFrame,
        pd.Series, pd.SparseDataFrame, scipy sparse matrix
        as input type.
        """
        orig1 = self.constructor([])
        orig2 = self.constructor(())
        orig3 = self.constructor({})
        orig4 = self.constructor(numpy.empty([0, 0]))
        orig5 = self.constructor(numpy.matrix(numpy.empty([0, 0])))
        if pd:
            orig6 = self.constructor(pd.DataFrame())
            orig7 = self.constructor(pd.Series())
            orig8 = self.constructor(pd.SparseDataFrame())

        assert orig1.isIdentical(orig2)
        assert orig1.isIdentical(orig3)
        assert orig1.isIdentical(orig4)
        assert orig1.isIdentical(orig5)
        assert orig1.isIdentical(orig6)
        assert orig1.isIdentical(orig7)
        assert orig1.isIdentical(orig8)


    def test_createEmptyData2(self):
        """
        create data object using tuple, list,
        dict, numpy.ndarray, numpy.matrix, pd.DataFrame,
        pd.Series, pd.SparseDataFrame, scipy sparse matrix
        as input type.
        """
        orig1 = self.constructor([[]])
        orig3 = self.constructor([{}])
        orig4 = self.constructor(numpy.empty([1, 0]))
        orig5 = self.constructor(numpy.matrix(numpy.empty([1, 0])))
        if pd:
            orig6 = self.constructor(pd.DataFrame([[]]))
            orig8 = self.constructor(pd.SparseDataFrame([[]]))
        if scipy:
            orig9 = self.constructor(scipy.sparse.coo_matrix([[]]))

        assert orig1.isIdentical(orig3)
        assert orig1.isIdentical(orig4)
        assert orig1.isIdentical(orig5)
        assert orig1.isIdentical(orig6)
        assert orig1.isIdentical(orig8)
        assert orig1.isIdentical(orig9)

    def test_createEmptyData3(self):
        """
        create data object using tuple, list,
        dict, numpy.ndarray, numpy.matrix, pd.DataFrame,
        pd.Series, pd.SparseDataFrame, scipy sparse matrix
        as input type.
        """
        orig1 = self.constructor([[], []])
        orig3 = self.constructor([{}, {}])
        orig4 = self.constructor(numpy.empty([2, 0]))
        orig5 = self.constructor(numpy.matrix(numpy.empty([2, 0])))
        if pd:
            orig6 = self.constructor(pd.DataFrame([[], []]))
            orig8 = self.constructor(pd.SparseDataFrame([[], []]))
        if scipy:
            orig9 = self.constructor(scipy.sparse.coo_matrix([[], []]))

        assert orig1.isIdentical(orig3)
        assert orig1.isIdentical(orig4)
        assert orig1.isIdentical(orig5)
        assert orig1.isIdentical(orig6)
        assert orig1.isIdentical(orig8)
        assert orig1.isIdentical(orig9)

    def test_create1DData(self):
        """
        create data object using tuple, list,
        dict, numpy.ndarray, numpy.matrix, pd.DataFrame,
        pd.Series, pd.SparseDataFrame, scipy sparse matrix
        as input type.
        """
        orig1 = self.constructor([1,2,3], featureNames=['a', 'b', 'c'])
        orig2 = self.constructor((1,2,3), featureNames=['a', 'b', 'c'])
        orig3 = self.constructor({'a':1, 'b':2, 'c':3})
        orig3.features.sort(sortBy=orig3.points.getName(0))
        orig10 = self.constructor([{'a':1, 'b':2, 'c':3}])
        orig10.features.sort(sortBy=orig10.points.getName(0))
        orig4 = self.constructor(numpy.array([1,2,3]), featureNames=['a', 'b', 'c'])
        orig5 = self.constructor(numpy.matrix([1,2,3]), featureNames=['a', 'b', 'c'])
        if pd:
            orig6 = self.constructor(pd.DataFrame([[1,2,3]]), featureNames=['a', 'b', 'c'])
            orig7 = self.constructor(pd.Series([1,2,3]), featureNames=['a', 'b', 'c'])
            orig8 = self.constructor(pd.SparseDataFrame([[1,2,3]]), featureNames=['a', 'b', 'c'])
        if scipy:
            orig9 = self.constructor(scipy.sparse.coo_matrix([1,2,3]), featureNames=['a', 'b', 'c'])

        assert orig1.isIdentical(orig2)
        assert orig1.isIdentical(orig3)
        assert orig1.isIdentical(orig10)
        assert orig1.isIdentical(orig4)
        assert orig1.isIdentical(orig5)
        assert orig1.isIdentical(orig6)
        assert orig1.isIdentical(orig7)
        assert orig1.isIdentical(orig8)
        assert orig1.isIdentical(orig9)

    def test_create2DData(self):
        """
        create data object using tuple, list,
        dict, numpy.ndarray, numpy.matrix, pd.DataFrame,
        pd.Series, pd.SparseDataFrame, scipy sparse matrix
        as input type.
        """
        orig1 = self.constructor([[1,2,'a'], [3,4,'b']], featureNames=['a', 'b', 'c'])
        orig2 = self.constructor(((1,2,'a'), (3,4,'b')), featureNames=['a', 'b', 'c'])
        orig3 = self.constructor({'a':[1,3], 'b':[2,4], 'c':['a', 'b']}, elementType=object)
        orig3.features.sort(sortBy=orig3.points.getName(0))
        orig7 = self.constructor([{'a':1, 'b':2, 'c':'a'}, {'a':3, 'b':4, 'c':'b'}], elementType=object)
        orig7.features.sort(sortBy=orig7.points.getName(0))
        orig4 = self.constructor(numpy.array([[1,2,'a'], [3,4,'b']], dtype=object), featureNames=['a', 'b', 'c'])
        orig5 = self.constructor(numpy.matrix([[1,2,'a'], [3,4,'b']], dtype=object), featureNames=['a', 'b', 'c'])
        if pd:
            orig6 = self.constructor(pd.DataFrame([[1,2,'a'], [3,4,'b']]), featureNames=['a', 'b', 'c'])
        if scipy:
            orig9 = self.constructor(scipy.sparse.coo_matrix(numpy.matrix([[1,2,'a'], [3,4,'b']], dtype=object)), featureNames=['a', 'b', 'c'])

        assert orig1.isIdentical(orig2)
        assert orig1.isIdentical(orig3)
        assert orig1.isIdentical(orig7)
        assert orig1.isIdentical(orig4)
        assert orig1.isIdentical(orig5)
        assert orig1.isIdentical(orig6)
        assert orig1.isIdentical(orig9)

    ##############
    # __init__() #
    ##############

    def test_init_allEqual(self):
        """ Test __init__() that every way to instantiate produces equal objects """
        # instantiate from list of lists
        fromList = self.constructor(data=[[1, 2, 3]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv") as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.flush()
            fromCSV = self.constructor(data=tmpCSV.name)

        # instantiate from mtx array file
        with tempfile.NamedTemporaryFile(mode='w', suffix=".mtx") as tmpMTXArr:
            tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
            tmpMTXArr.write("1 3\n")
            tmpMTXArr.write("1\n")
            tmpMTXArr.write("2\n")
            tmpMTXArr.write("3\n")
            tmpMTXArr.flush()
            fromMTXArr = self.constructor(data=tmpMTXArr.name)

        # instantiate from mtx coordinate file
        with tempfile.NamedTemporaryFile(mode='w', suffix=".mtx") as tmpMTXCoo:
            tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.write("1 1 1\n")
            tmpMTXCoo.write("1 2 2\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.flush()
            fromMTXCoo = self.constructor(data=tmpMTXCoo.name)

        # check equality between all pairs
        assert fromList.isIdentical(fromCSV)
        assert fromMTXArr.isIdentical(fromList)
        assert fromMTXArr.isIdentical(fromCSV)
        assert fromMTXCoo.isIdentical(fromList)
        assert fromMTXCoo.isIdentical(fromCSV)
        assert fromMTXCoo.isIdentical(fromMTXArr)

    def test_init_allEqualWithNames(self):
        """ Test __init__() that every way to instantiate produces equal objects, with names """
        # instantiate from list of lists
        fromList = self.constructor(data=[[1, 2, 3]], pointNames=['1P'], featureNames=['one', 'two', 'three'])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv") as tmpCSV:
            tmpCSV.write("\n")
            tmpCSV.write("\n")
            tmpCSV.write("pointNames,one,two,three\n")
            tmpCSV.write("1P,1,2,3\n")
            tmpCSV.flush()
            fromCSV = self.constructor(data=tmpCSV.name)

        # instantiate from mtx file
        with tempfile.NamedTemporaryFile(mode='w', suffix=".mtx") as tmpMTXArr:
            tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
            tmpMTXArr.write("%#1P\n")
            tmpMTXArr.write("%#one,two,three\n")
            tmpMTXArr.write("1 3\n")
            tmpMTXArr.write("1\n")
            tmpMTXArr.write("2\n")
            tmpMTXArr.write("3\n")
            tmpMTXArr.flush()
            fromMTXArr = self.constructor(data=tmpMTXArr.name)

        # instantiate from mtx coordinate file
        with tempfile.NamedTemporaryFile(mode='w', suffix=".mtx") as tmpMTXCoo:
            tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
            tmpMTXCoo.write("%#1P\n")
            tmpMTXCoo.write("%#one,two,three\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.write("1 1 1\n")
            tmpMTXCoo.write("1 2 2\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.flush()
            fromMTXCoo = self.constructor(data=tmpMTXCoo.name)

        # check equality between all pairs
        assert fromList.isIdentical(fromCSV)
        assert fromMTXArr.isIdentical(fromList)
        assert fromMTXArr.isIdentical(fromCSV)
        assert fromMTXCoo.isIdentical(fromList)
        assert fromMTXCoo.isIdentical(fromCSV)
        assert fromMTXCoo.isIdentical(fromMTXArr)


    @raises(TypeError)
    def test_init_noThriceNestedListInputs(self):
        self.constructor([[[1, 2, 3]]])


    def test_init_coo_matrix_duplicates(self):
        # Constructing a matrix with duplicate indices
        row  = numpy.array([0, 0, 1, 3, 1, 0, 0])
        col  = numpy.array([0, 2, 1, 3, 1, 0, 0])
        data = numpy.array([1, 7, 1, 6, 4, 2, 1])
        coo = scipy.sparse.coo_matrix((data, (row, col)),shape=(4,4))
        ret = self.constructor(coo)
        # Expected coo_matrix duplicates sum
        row  = numpy.array([0, 0, 1, 3])
        col  = numpy.array([0, 2, 1, 3])
        data = numpy.array([4, 7, 5, 6])
        coo = scipy.sparse.coo_matrix((data, (row, col)),shape=(4,4))
        exp = self.constructor(coo)

        assert ret.isIdentical(exp)
        assert ret[0,0] == exp[0,0]
        assert ret[3,3] == exp[3,3]
        assert ret[1,1] == exp[1,1]

    def test_init_coo_matrix_duplicates_introduces_zero(self):
        # Constructing a matrix with duplicate indices
        row  = numpy.array([0, 0, 1, 3, 1, 0, 0])
        col  = numpy.array([0, 2, 1, 3, 1, 0, 0])
        data = numpy.array([1, 7, 1, 6, -1, 2, 1])
        coo = scipy.sparse.coo_matrix((data, (row, col)),shape=(4,4))
        ret = self.constructor(coo)
        # Expected coo_matrix duplicates sum
        row  = numpy.array([0, 0, 3])
        col  = numpy.array([0, 2, 3])
        data = numpy.array([4, 7, 6])
        coo = scipy.sparse.coo_matrix((data, (row, col)),shape=(4,4))
        exp = self.constructor(coo)

        assert ret.isIdentical(exp)
        assert ret[0,0] == exp[0,0]
        assert ret[3,3] == exp[3,3]
        assert ret[0,2] == exp[0,2]


    def test_init_coo_matrix_duplicateswithNoDupStrings(self):
        # Constructing a matrix with duplicate indices
        # with String, but not in duplicate entry
        row  = numpy.array([0, 0, 1, 3, 1, 0, 0])
        col  = numpy.array([0, 2, 1, 3, 1, 0, 0])
        # need to specify object dtype, otherwise it will generate a all string object
        data = numpy.array([1, 7, 1, 'AAA', 4, 2, 1], dtype='O')
        coo_str = scipy.sparse.coo_matrix((data, (row, col)),shape=(4,4))
        ret = self.constructor(coo_str)
        # Expected coo_matrix duplicates sum
        # with String, but not in duplicate entry
        row  = numpy.array([0, 0, 1, 3])
        col  = numpy.array([0, 2, 1, 3])
        data = numpy.array([4, 7, 5, 'AAA'], dtype='O')
        coo = scipy.sparse.coo_matrix((data, (row, col)),shape=(4,4))
        exp = self.constructor(coo_str)

        assert ret.isIdentical(exp)
        assert ret[0,0] == exp[0,0]
        assert ret[3,3] == exp[3,3]
        assert ret[1,1] == exp[1,1]

    @raises(ValueError)
    def test_init_coo_matrix_duplicateswithDupStrings(self):
        # Constructing a matrix with duplicate indices
        # # with String, in a duplicate entry
        row  = numpy.array([0, 0, 1, 3, 1, 0, 0])
        col  = numpy.array([0, 2, 1, 3, 1, 0, 0])
        data = numpy.array([1, 7, 1, 'AAA', 4, 2, 'BBB'], dtype='O')
        coo_str = scipy.sparse.coo_matrix((data, (row, col)),shape=(4,4))
        ret = self.constructor(coo_str)

    @raises(CalledFunctionException)
    @mock.patch('nimble.data.base.valuesToPythonList', calledException)
    def test_init_pointNames_calls_valuesToPythonList(self):
        self.constructor([1,2,3], pointNames=['one'])

    @raises(CalledFunctionException)
    @mock.patch('nimble.data.base.valuesToPythonList', calledException)
    def test_init_featureNames_calls_valuesToPythonList(self):
        self.constructor([1,2,3], featureNames=['a', 'b', 'c'])

    ###############
    # transpose() #
    ###############

    def test_transpose_empty(self):
        """ Test transpose() on different kinds of emptiness """
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)

        toTest.transpose()

        exp1 = [[], []]
        exp1 = numpy.array(exp1)
        ret1 = self.constructor(exp1)
        assert ret1.isIdentical(toTest)

        toTest.transpose()

        exp2 = [[], []]
        exp2 = numpy.array(exp2).T
        ret2 = self.constructor(exp2)
        assert ret2.isIdentical(toTest)

    @logCountAssertionFactory(3)
    def test_transpose_handmade(self):
        """ Test transpose() function against handmade output """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        dataTrans = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

        dataObj1 = self.constructor(copy.deepcopy(data))
        dataObj2 = self.constructor(copy.deepcopy(data))
        dataObjT = self.constructor(copy.deepcopy(dataTrans))

        ret1 = dataObj1.transpose() # RET CHECK
        assert dataObj1.isIdentical(dataObjT)
        assert ret1 is None
        dataObj1.transpose()
        dataObjT.transpose()
        assert dataObj1.isIdentical(dataObj2)
        assert dataObj2.isIdentical(dataObjT)
        assertNoNamesGenerated(dataObj1)
        assertNoNamesGenerated(dataObj2)

    def test_transpose_handmadeWithZeros(self):
        """ Test transpose() function against handmade output """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0], [11, 12, 13]]
        dataTrans = [[1, 4, 7, 0, 11], [2, 5, 8, 0, 12], [3, 6, 9, 0, 13]]

        dataObj1 = self.constructor(copy.deepcopy(data))
        dataObj2 = self.constructor(copy.deepcopy(data))
        dataObjT = self.constructor(copy.deepcopy(dataTrans))

        ret1 = dataObj1.transpose() # RET CHECK

        assert dataObj1.isIdentical(dataObjT)
        assert ret1 is None

        dataObj1.transpose()
        dataObjT.transpose()
        assert dataObj1.isIdentical(dataObj2)
        assert dataObj2.isIdentical(dataObjT)

    def test_transpose_handmadeWithAxisNames(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]]
        dataTrans = [[1, 4, 7, 0], [2, 5, 8, 0], [3, 6, 9, 0]]

        origPointNames = ['1','2','3','4']
        origFeatureNames = ['a','b','c']
        transPointNames = origFeatureNames
        transFeatureNames = origPointNames

        dataObj1 = self.constructor(copy.deepcopy(data), pointNames=origPointNames,
                                                    featureNames=origFeatureNames)
        dataObj2 = self.constructor(copy.deepcopy(data), pointNames=origPointNames,
                                                    featureNames=origFeatureNames)
        dataObjT = self.constructor(copy.deepcopy(dataTrans), pointNames=transPointNames,
                                                         featureNames=transFeatureNames)
        dataObj1.transpose()
        assert dataObj1.points.getNames() == transPointNames
        assert dataObj1.features.getNames() == transFeatureNames
        assert dataObj1.isIdentical(dataObjT)

        dataObj1.transpose()
        dataObjT.transpose()
        assert dataObj1.points.getNames() == dataObj2.points.getNames()
        assert dataObj1.features.getNames() == dataObj2.features.getNames()
        assert dataObj1.isIdentical(dataObj2)

        assert dataObj2.points.getNames() == dataObjT.points.getNames()
        assert dataObj2.features.getNames() == dataObjT.features.getNames()
        assert dataObj2.isIdentical(dataObjT)

    def test_transpose_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0], [11, 12, 13]]

        dataObj1 = self.constructor(copy.deepcopy(data))

        dataObj1._name = "TestName"
        dataObj1._absPath = "TestAbsPath"
        dataObj1._relPath = "testRelPath"

        dataObj1.transpose()

        assert dataObj1.name == "TestName"
        assert dataObj1.absolutePath == "TestAbsPath"
        assert dataObj1.relativePath == 'testRelPath'

    ##################################
    # common backends insert/append #
    #################################

    @oneLogEntryExpected
    def backend_insert_emptyObject(self, axis, funcName):
        empty = [[], []]

        if axis == 'point':
            empty = numpy.array(empty).T
            data = [[1, 2]]
        else:
            empty = numpy.array(empty)
            data = [[1], [2]]

        toTest = self.constructor(empty)
        toInsert = self.constructor(data)
        toExp = self.constructor(data)

        if axis == 'point':
            if funcName == 'insert':
                toTest.points.insert(0, toInsert)
            else:
                toTest.points.append(toInsert)
        else:
            if funcName == 'insert':
                toTest.features.insert(0, toInsert)
            else:
                toTest.features.append(toInsert)

        assert toTest.isIdentical(toExp)

    @oneLogEntryExpected
    def backend_insert_handmadeSingle(self, axis, insertBefore):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        offNames = ['o1', 'o2', 'o3']
        names = ['one', 'two', 'three']
        addName = ['new']

        if axis == 'point':
            if insertBefore in [None, 3]:
                dataExpected = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]]
                namesExp = ['one', 'two', 'three', 'new']
            elif insertBefore == 0:
                dataExpected = [[-1, -2, -3], [1, 2, 3], [4, 5, 6], [7, 8, 9]]
                namesExp = ['new', 'one', 'two', 'three']
            elif insertBefore == 1:
                dataExpected = [[1, 2, 3], [-1, -2, -3], [4, 5, 6], [7, 8, 9]]
                namesExp = ['one', 'new', 'two', 'three']
            toTest = self.constructor(data, pointNames=names, featureNames=offNames)
            toInsert = self.constructor([[-1, -2, -3]], pointNames=addName, featureNames=offNames)
            expected = self.constructor(dataExpected, pointNames=namesExp, featureNames=offNames)
            if insertBefore is None:
                ret = toTest.points.append(toInsert) # RET CHECK
            else:
                ret = toTest.points.insert(insertBefore, toInsert)  # RET CHECK
        else:
            if insertBefore in [None, 3]:
                dataExpected = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
                namesExp = ['one', 'two', 'three', 'new']
            elif insertBefore == 0:
                dataExpected = [[-1, 1, 2, 3], [-2, 4, 5, 6], [-3, 7, 8, 9]]
                namesExp = ['new', 'one', 'two', 'three']
            elif insertBefore == 1:
                dataExpected = [[1, -1, 2, 3], [4, -2, 5, 6], [7, -3, 8, 9]]
                namesExp = ['one', 'new', 'two', 'three']
            toTest = self.constructor(data, pointNames=offNames, featureNames=names)
            toInsert = self.constructor([[-1], [-2], [-3]], pointNames=offNames, featureNames=addName)
            expected = self.constructor(dataExpected, pointNames=offNames, featureNames=namesExp)
            if insertBefore is None:
                ret = toTest.features.append(toInsert) # RET CHECK
            else:
                ret = toTest.features.insert(insertBefore, toInsert)  # RET CHECK

        assert toTest.isIdentical(expected)
        assert ret is None

    @logCountAssertionFactory(4)
    def backend_insert_handmadeSequence(self, axis, insertBefore):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        offNames = ['o1', 'o2', 'o3']
        names = ['one', 'two', 'three']
        newNames = ['a1', 'b1', 'b2', 'c1']
        toInsert = [[0.1, 0.2, 0.3], [0.01, 0.02, 0.03], [0, 0, 0], [10, 11, 12]]
        toInsert = self.constructor(toInsert, pointNames=newNames, featureNames=offNames)

        if axis == 'point':
            if insertBefore is None:
                dataExpected = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0.1, 0.2, 0.3],
                                [0.01, 0.02, 0.03], [0, 0, 0], [10, 11, 12]]
                namesExp = names + newNames
            elif insertBefore == 0:
                dataExpected = [[10, 11, 12], [0, 0, 0], [0.01, 0.02, 0.03], [0.1, 0.2, 0.3],
                                [1, 2, 3], [4, 5, 6], [7, 8, 9]]
                namesExp = list(reversed(newNames)) + names
            elif insertBefore == 1:
                dataExpected = [[1, 2, 3], [10, 11, 12], [0, 0, 0], [0.01, 0.02, 0.03],
                                [0.1, 0.2, 0.3], [4, 5, 6], [7, 8, 9]]
                namesExp = names[:1] + list(reversed(newNames)) + names[1:]
            elif insertBefore == 3:
                dataExpected = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [0, 0, 0], [0.01, 0.02, 0.03],
                                [0.1, 0.2, 0.3]]
                namesExp = names + list(reversed(newNames))
            toTest = self.constructor(data, pointNames=names, featureNames=offNames)
            for nextInsert in toInsert.points:
                if insertBefore is None:
                    toTest.points.append(nextInsert)
                else:
                    toTest.points.insert(insertBefore, nextInsert)
            expected = self.constructor(dataExpected, pointNames=namesExp, featureNames=offNames)
        else:
            if insertBefore is None:
                dataExpected = [[1, 2, 3, 0.1, 0.01, 0, 10], [4, 5, 6, 0.2, 0.02, 0, 11], [7, 8, 9, 0.3, 0.03, 0, 12]]
                namesExp = names + newNames
            elif insertBefore == 0:
                dataExpected = [[10, 0, 0.01, 0.1, 1, 2, 3], [11, 0, 0.02, 0.2, 4, 5, 6], [12, 0, 0.03, 0.3, 7, 8, 9]]
                namesExp = list(reversed(newNames)) + names
            elif insertBefore == 1:
                dataExpected = [[1, 10, 0, 0.01, 0.1, 2, 3], [4, 11, 0, 0.02, 0.2, 5, 6], [7, 12, 0, 0.03, 0.3, 8, 9]]
                namesExp = names[:1] + list(reversed(newNames)) + names[1:]
            elif insertBefore == 3:
                dataExpected = [[1, 2, 3, 10, 0, 0.01, 0.1], [4, 5, 6, 11, 0, 0.02, 0.2], [7, 8, 9, 12, 0, 0.03, 0.3]]
                namesExp = names + list(reversed(newNames))
            toTest = self.constructor(data, pointNames=offNames, featureNames=names)
            toInsert.transpose(useLog=False)
            for nextInsert in toInsert.features:
                if insertBefore is None:
                    toTest.features.append(nextInsert)
                else:
                    toTest.features.insert(insertBefore, nextInsert)
            expected = self.constructor(dataExpected, pointNames=offNames, featureNames=namesExp)

        assert toTest.isIdentical(expected)

    @oneLogEntryExpected
    def backend_insert_selfInsert(self, axis, insertBefore):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        names = ['one', 'two', 'three']

        if axis == 'point':
            orig = self.constructor(data, featureNames=names)
        else:
            orig = self.constructor(data, pointNames=names)

        dup = orig.copy()

        if axis == 'point':
            dupNames = dup.points.getNames()
            assert orig.points.getNames() == dupNames

            if insertBefore is None:
                orig.points.append(orig)
            else:
                orig.points.insert(insertBefore, orig)

            if insertBefore in [3, 0, None]:
                dataExp = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]]
            elif insertBefore == 1:
                dataExp = [[1, 2, 3], [1, 2, 3], [4, 5, 6], [7, 8, 9], [4, 5, 6], [7, 8, 9]]
            expected = self.constructor(dataExp, featureNames=names)
        else:
            dupNames = dup.features.getNames()
            assert orig.features.getNames() == dupNames

            if insertBefore is None:
                orig.features.append(orig)
            else:
                orig.features.insert(insertBefore, orig)

            if insertBefore in [3, 0, None]:
                dataExp = [[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6], [7, 8, 9, 7, 8, 9]]
            elif insertBefore == 1:
                dataExp = [[1, 1, 2, 3, 2, 3], [4, 4, 5, 6, 5, 6], [7, 7, 8, 9, 8, 9]]
            expected = self.constructor(dataExp, pointNames=names)

        assert orig == expected

        checkNames = orig.points.getNames() if axis == 'point' else orig.features.getNames()
        lastDefIndex = int(dupNames[2][-1])
        if insertBefore in [3, None]:
            assert checkNames[:3] == dupNames
            # indexes of inserted data
            idx1, idx2, idx3 = 3, 4, 5
        elif insertBefore == 0:
            assert checkNames[3:] == dupNames
            # indexes of inserted data
            idx1, idx2, idx3 = 0, 1, 2
        elif insertBefore == 1:
            assert [checkNames[0]] + checkNames[4:] == dupNames
            # indexes of inserted data
            idx1, idx2, idx3 = 1, 2, 3

        assert checkNames[idx1] == DEFAULT_PREFIX + str(lastDefIndex + 1)
        assert checkNames[idx2] == DEFAULT_PREFIX + str(lastDefIndex + 2)
        assert checkNames[idx3] == DEFAULT_PREFIX + str(lastDefIndex + 3)

    def backend_insert_automaticReorder(self, axis, defPrimaryNames, insertBefore):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        offNames = ['off1', 'off2', 'off3']
        addOffName = ['off3', 'off2', 'off1']
        if defPrimaryNames:
            names = [None] * 3
            addName = [None]
            namesExp = [None] * 4
        else:
            names = ['one', 'two', 'three']
            addName = ['new']
            if insertBefore in [3, None]:
                namesExp = ['one', 'two', 'three', 'new']
            elif insertBefore == 0:
                namesExp = ['new', 'one', 'two', 'three']
            elif insertBefore == 1:
                namesExp = ['one', 'new', 'two', 'three']

        if axis == 'point':
            toInsertData = [[-3, -2, -1]]
            if insertBefore in [3, None]:
                dataExpected = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]]
            elif insertBefore == 0:
                dataExpected = [[-1, -2, -3], [1, 2, 3], [4, 5, 6], [7, 8, 9]]
            elif insertBefore == 1:
                dataExpected = [[1, 2, 3], [-1, -2, -3], [4, 5, 6], [7, 8, 9]]
            toTest = self.constructor(data, pointNames=names, featureNames=offNames)
            toInsert = self.constructor(toInsertData, pointNames=addName, featureNames=addOffName)
            expInsert = toInsert.copy()
            expected = self.constructor(dataExpected, pointNames=namesExp, featureNames=offNames)
            if insertBefore is None:
                toTest.points.append(toInsert)
            else:
                toTest.points.insert(insertBefore, toInsert)
        else:
            toInsertData = [[-3], [-2], [-1]]
            if insertBefore in [3, None]:
                dataExpected = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
            elif insertBefore == 0:
                dataExpected = [[-1, 1, 2, 3], [-2, 4, 5, 6], [-3, 7, 8, 9]]
            elif insertBefore == 1:
                dataExpected = [[1, -1, 2, 3], [4, -2, 5, 6], [7, -3, 8, 9]]
            toTest = self.constructor(data, pointNames=offNames, featureNames=names)
            toInsert = self.constructor(toInsertData, pointNames=addOffName, featureNames=addName)
            expInsert = toInsert.copy()
            expected = self.constructor(dataExpected, pointNames=offNames, featureNames=namesExp)
            if insertBefore is None:
                toTest.features.append(toInsert)
            else:
                toTest.features.insert(insertBefore, toInsert)
        # check that toInsert object was not modified when reordering occurred
        assert toInsert.isIdentical(expInsert)
        assert toTest.isIdentical(expected)

    #######################################
    # points.insert() / features.insert() #
    #######################################

    def backend_insert_exceptionInsertBeforeNone(self, axis):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toInsert = self.constructor([[2, 3, 4, 5, 6]])

        if axis == 'point':
            toTest.points.insert(None, toInsert)
        else:
            toInsert.transpose()
            toTest.features.insert(None, toInsert)

    @raises(InvalidArgumentType)
    def test_points_insert_exceptionInsertBeforeNone(self):
        """ Test points.insert() for InvalidArgumentType when insertBefore is None """
        self.backend_insert_exceptionInsertBeforeNone('point')

    @raises(InvalidArgumentType)
    def test_features_insert_exceptionInsertBeforeNone(self):
        """ Test features.insert() for InvalidArgumentType when insertBefore is None """
        self.backend_insert_exceptionInsertBeforeNone('feature')

    def backend_insert_exceptionWrongSize(self, axis):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toInsert = self.constructor([[2, 3, 4, 5, 6]])

        if axis == 'point':
            toTest.points.insert(len(toTest.points), toInsert)
        else:
            toInsert.transpose()
            toTest.features.insert(len(toTest.features), toInsert)

    @raises(InvalidArgumentValue)
    def test_points_insert_exceptionWrongSize(self):
        """ Test points.insert() for InvalidArgumentValue when toInsert has too many features """
        self.backend_insert_exceptionWrongSize('point')

    @raises(InvalidArgumentValue)
    def test_features_insert_exceptionWrongSize(self):
        """ Test features.insert() for InvalidArgumentValue when toInsert has too many points """
        self.backend_insert_exceptionWrongSize('feature')


    def backend_insert_exception_extendAxis_SameName(self, axis):
        toTest1 = self.constructor([[1, 2]], pointNames=["hello"])
        toTest2 = self.constructor([[1, 2], [5, 6]], pointNames=["hello", "goodbye"])

        if axis == 'point':
            toTest2.points.insert(len(toTest2.points), toTest1)
        else:
            toTest1.transpose()
            toTest2.transpose()
            toTest2.features.insert(len(toTest2.features), toTest1)

    @raises(InvalidArgumentValue)
    def test_points_insert_exceptionSamePointName(self):
        """ Test points.insert() for InvalidArgumentValue when toInsert and self have a pointName in common """
        self.backend_insert_exception_extendAxis_SameName('point')

    @raises(InvalidArgumentValue)
    def test_features_insert_exceptionSameFeatureName(self):
        """ Test features.insert() for InvalidArgumentValue when toInsert and self have a featureName in common """
        self.backend_insert_exception_extendAxis_SameName('feature')


    def backend_insert_exception_sharedAxis_unsharedName(self, axis):
        toTest1 = self.constructor([[1, 2]], featureNames=['1', '2'])
        toTest2 = self.constructor([[2, 1], [6, 5]], featureNames=['6', '1'])

        if axis == 'point':
            toTest2.points.insert(len(toTest2.points), toTest1)
        else:
            toTest1.transpose()
            toTest2.transpose()
            toTest2.features.insert(len(toTest2.features), toTest1)

    @raises(InvalidArgumentValue)
    def test_points_insert_exception_unsharedFeatureName(self):
        """ Test points.insert() for InvalidArgumentValue when toInsert and self have a featureName not in common """
        self.backend_insert_exception_sharedAxis_unsharedName('point')

    @raises(InvalidArgumentValue)
    def test_features_insert_exception_unsharedPointName(self):
        """ Test features.insert() for InvalidArgumentValue when toInsert and self have a pointName not in common """
        self.backend_insert_exception_sharedAxis_unsharedName('feature')


    def backend_insert_exceptionNonNimbleDataType(self, axis):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        if axis == 'point':
            toTest.points.insert(len(toTest.points), [[1, 1, 1]])
        else:
            toTest.features.insert(len(toTest.features), [[1], [1], [1]])

    @raises(InvalidArgumentType)
    def test_points_insert_exceptionNonNimbleDataType(self):
        self.backend_insert_exceptionNonNimbleDataType('point')

    @raises(InvalidArgumentType)
    def test_features_insert_exceptionNonNimbleDataType(self):
        self.backend_insert_exceptionNonNimbleDataType('feature')


    def backend_insert_exception_outOfOrder_with_defaults(self, axis):
        toTest1 = self.constructor([[1, 2, 3]])
        toTest2 = self.constructor([[1, 3, 2]])

        toTest1.features.setName(1, '2')
        toTest1.features.setName(2, '3')
        toTest2.features.setName(1, '3')
        toTest2.features.setName(2, '2')

        if axis == 'point':
            toTest1.points.insert(len(toTest1.points), toTest2)
        else:
            toTest1.transpose()
            toTest2.transpose()
            toTest1.features.insert(len(toTest1.features), toTest2)

    @raises(ImproperObjectAction)
    def test_points_insert_exception_outOfOrder_with_defaults(self):
        """ Test points.insert() for ImproperObjectAction when toInsert and self contain a mix of set names and default names not in the same order"""
        self.backend_insert_exception_outOfOrder_with_defaults('point')

    @raises(ImproperObjectAction)
    def test_features_insert_exception_outOfOrder_with_defaults(self):
        """ Test features.insert() for ImproperObjectAction when toInsert and self contain a mix of set names and default names not in the same order"""
        self.backend_insert_exception_outOfOrder_with_defaults('feature')

    @raises(IndexError)
    def test_points_insert_fromEmpty_top(self):
        """ Test points.insert() with an insertBefore ID when the calling object is point empty raises exception """
        self.backend_insert_emptyObject('point', 'insert')

    @raises(IndexError)
    def test_features_insert_fromEmpty_left(self):
        """ Test features.insert() with an insertBefore ID when the calling object is feature empty raises exception """
        self.backend_insert_emptyObject('feature', 'insert')

    def test_points_insert_handmadeSingle_bottom(self):
        """ Test points.insert() against handmade output for a single added point to the bottom"""
        self.backend_insert_handmadeSingle('point', 3)

    def test_features_insert_handmadeSingle_right(self):
        """ Test features.insert() against handmade output for a single added feature to the right"""
        self.backend_insert_handmadeSingle('feature', 3)

    def test_points_insert_handmadeSingle_top(self):
        """ Test points.insert() against handmade output for a single added point the the top"""
        self.backend_insert_handmadeSingle('point', 0)

    def test_features_insert_handmadeSingle_left(self):
        """ Test features.insert() against handmade output for a single added feature to the left"""
        self.backend_insert_handmadeSingle('feature', 0)

    def test_points_insert_handmadeSingle_mid(self):
        """ Test points.insert() against handmade output for a single added point in the middle"""
        self.backend_insert_handmadeSingle('point', 1)

    def test_features_insert_handmadeSingle_mid(self):
        """ Test features.insert() against handmade output for a single added feature in the middle"""
        self.backend_insert_handmadeSingle('feature', 1)

    def test_points_insert_handmadeSequence_bottom(self):
        """ Test points.insert() against handmade output for a sequence of additions to the bottom"""
        self.backend_insert_handmadeSequence('point', 3)

    def test_features_insert_handmadeSequence_right(self):
        """ Test features.insert() against handmade output for a sequence of additions to the right"""
        self.backend_insert_handmadeSequence('feature', 3)

    def test_points_insert_handmadeSequence_top(self):
        """ Test points.insert() against handmade output for a sequence of additions to the top"""
        self.backend_insert_handmadeSequence('point', 0)

    def test_features_insert_handmadeSequence_left(self):
        """ Test features.insert() against handmade output for a sequence of additions to the left"""
        self.backend_insert_handmadeSequence('feature', 0)

    def test_points_insert_handmadeSequence_mid(self):
        """ Test points.insert() against handmade output for a sequence of additions to the middle"""
        self.backend_insert_handmadeSequence('point', 1)

    def test_features_insert_handmadeSequence_mid(self):
        """ Test features.insert() against handmade output for a sequence of additions to the middle"""
        self.backend_insert_handmadeSequence('feature', 1)

    def test_points_insert_selfInsert_bottom(self):
        self.backend_insert_selfInsert('point', 3)

    def test_features_insert_selfInsert_right(self):
        self.backend_insert_selfInsert('feature', 3)

    def test_points_insert_selfInsert_top(self):
        self.backend_insert_selfInsert('point', 0)

    def test_features_insert_selfInsert_left(self):
        self.backend_insert_selfInsert('feature', 0)

    def test_points_insert_selfInsert_mid(self):
        self.backend_insert_selfInsert('point', 1)

    def test_features_insert_selfInsert_mid(self):
        self.backend_insert_selfInsert('feature', 1)

    def test_points_insert_automaticReorder_fullySpecifiedNames_bottom(self):
        self.backend_insert_automaticReorder('point', False, 3)

    def test_features_insert_automaticReorder_fullySpecifiedNames_right(self):
        self.backend_insert_automaticReorder('feature', False, 3)

    def test_points_insert_automaticReorder_defaultPointNames_bottom(self):
        self.backend_insert_automaticReorder('point', True, 3)

    def test_features_insert_automaticReorder_defaultFeatureNames_right(self):
        self.backend_insert_automaticReorder('feature', True, 3)

    def test_points_insert_automaticReorder_fullySpecifiedNames_top(self):
        self.backend_insert_automaticReorder('point', False, 0)

    def test_features_insert_automaticReorder_fullySpecifiedNames_left(self):
        self.backend_insert_automaticReorder('feature', False, 0)

    def test_points_insert_automaticReorder_defaultPointNames_top(self):
        self.backend_insert_automaticReorder('point', True, 0)

    def test_features_insert_automaticReorder_defaultFeatureNames_left(self):
        self.backend_insert_automaticReorder('feature', True, 0)

    def test_points_insert_automaticReorder_fullySpecifiedNames_mid(self):
        self.backend_insert_automaticReorder('point', False, 1)

    def test_features_insert_automaticReorder_fullySpecifiedNames_mid(self):
        self.backend_insert_automaticReorder('feature', False, 1)

    def test_points_insert_automaticReorder_defaultPointNames_mid(self):
        self.backend_insert_automaticReorder('point', True, 1)

    def test_features_insert_automaticReorder_defaultFeatureNames_mid(self):
        self.backend_insert_automaticReorder('feature', True, 1)

    def backend_insert_allPossibleNimbleDataType(self, axis):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        currType = self.constructor([]).getTypeString()
        availableTypes = nimble.data.available
        otherTypes = [retType for retType in availableTypes if retType != currType]
        inserted = []
        for other in otherTypes:
            toTest = self.constructor(data)
            if axis == 'point':
                insertData = [[-1, -2, -3]]
                otherTest = nimble.createData(other, insertData)
                exp = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
                toTest.points.insert(len(toTest.points), otherTest)
                inserted.append(toTest)
            else:
                insertData = [[-1], [-2], [-3]]
                otherTest = nimble.createData(other, insertData)
                exp = self.constructor([[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]])
                toTest.features.insert(len(toTest.features), otherTest)
                inserted.append(toTest)

        assert all(exp == obj for obj in inserted)

    def test_points_insert_allPossibleNimbleDataType(self):
        self.backend_insert_allPossibleNimbleDataType('point')

    def test_features_insert_allPossibleNimbleDataType(self):
        self.backend_insert_allPossibleNimbleDataType('feature')


    def backend_insert_noReorderWithAllDefaultNames(self, axis):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        if axis == 'point':
            insertData = [[-1, -2, -3]]
            # assign names to be the reverse of toTest's default names
            fNames = list(reversed(toTest.features.getNames()))
            toInsert = self.constructor(insertData, featureNames=fNames)
            assert toTest.features.getNames() != toInsert.features.getNames()

            exp = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
            toTest.points.insert(len(toTest.points), toInsert)

        else:
            insertData = [[-1], [-2], [-3]]
            # assign names to be the reverse of toTest's default names
            pNames = list(reversed(toTest.points.getNames()))
            toInsert = self.constructor(insertData, pointNames=pNames)
            assert toTest.points.getNames() != toInsert.points.getNames()

            exp = self.constructor([[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]])
            toTest.features.insert(len(toTest.features), toInsert)

        assert toTest == exp

    def test_points_insert_noReorderWithAllDefaultNames(self):
        self.backend_insert_noReorderWithAllDefaultNames('point')

    def test_features_insert_noReorderWithAllDefaultNames(self):
        self.backend_insert_noReorderWithAllDefaultNames('feature')

    def backend_insert_noReorderInsertedHasDefaultNames(self, axis):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        pNames = ['1', '4', '7']
        fNames = ['a', 'b', 'c']
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        if axis == 'point':
            insertData = [[-1, -2, -3]]
            # toInsert has default names
            toInsert = self.constructor(insertData)
            assert toTest.features.getNames() != toInsert.features.getNames()

            exp = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
            exp.features.setNames(fNames)
            exp.points.setName(0, '1')
            exp.points.setName(1, '4')
            exp.points.setName(2, '7')
            toTest.points.insert(len(toTest.points), toInsert)

        else:
            insertData = [[-1], [-2], [-3]]
            # toInsert has default names
            toInsert = self.constructor(insertData)
            assert toTest.points.getNames() != toInsert.points.getNames()

            exp = self.constructor([[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]])
            exp.points.setNames(pNames)
            exp.features.setName(0, 'a')
            exp.features.setName(1, 'b')
            exp.features.setName(2, 'c')
            toTest.features.insert(len(toTest.features), toInsert)

        assert toTest == exp

    def test_addPoints_noReorderInsertedHasDefaultNames(self):
        self.backend_insert_noReorderInsertedHasDefaultNames('point')

    def test_addFeatures_noReorderInsertedHasDefaultNames(self):
        self.backend_insert_noReorderInsertedHasDefaultNames('feature')

    def backend_insert_lazyNameGeneration(self, axis):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        if axis == 'point':
            insertData = [[-1, -2, -3]]
            # toInsert has default names
            toInsert = self.constructor(insertData)
            toTest.points.insert(len(toTest.points), toInsert)

        else:
            insertData = [[-1], [-2], [-3]]
            # toInsert has default names
            toInsert = self.constructor(insertData)
            toTest.features.insert(len(toTest.features), toInsert)

        assertNoNamesGenerated(toTest)
        assertNoNamesGenerated(toInsert)

    def test_addPoints_lazyNameGeneration(self):
        self.backend_insert_lazyNameGeneration('point')

    def test_addFeatures_lazyNameGeneration(self):
        self.backend_insert_lazyNameGeneration('feature')

    def backend_insert_NamePath_preservation(self, axis):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        names = ['one', 'two', 'three']

        if axis == 'point':
            toTest = self.constructor(data, pointNames=names)
            toInsert = self.constructor([[-1, -2, -3]], pointNames=['new'])
        else:
            toTest = self.constructor(data, featureNames=names)
            toInsert = self.constructor([[-1], [-2], [-3]], featureNames=['new'])

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toInsert._name = "TestNameOther"
        toInsert._absPath = "TestAbsPathOther"
        toInsert._relPath = "testRelPathOther"

        if axis == 'point':
            toTest.points.insert(len(toTest.points), toInsert)
        else:
            toTest.features.insert(len(toTest.features), toInsert)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'

    def test_points_insert_NamePath_preservation(self):
        self.backend_insert_NamePath_preservation('point')

    def test_features_insert_NamePath_preservation(self):
        self.backend_insert_NamePath_preservation('feature')

    def test_points_insert_noNamesCreated(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toInsert = self.constructor([[-1, -2, -3]])
        toTest.points.insert(len(toTest.points), toInsert)

        assert not toTest._pointNamesCreated()
        assert not toTest._featureNamesCreated()

    def test_features_insert_noNamesCreated(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toInsert = self.constructor([[-1], [-2], [-3]])
        toTest.features.insert(len(toTest.features), toInsert)

        assert not toTest._featureNamesCreated()
        assert not toTest._pointNamesCreated()


    #######################################
    # points.append() / features.append() #
    #######################################

    @mock.patch('nimble.data.axis.Axis._insert', calledException)
    def test_append_callsInsertBackend(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toAppend = self.constructor([[-1], [-2], [-3]])
        try:
            toTest.points.append(toAppend)
            assert False
        except CalledFunctionException:
            pass

        try:
            toTest.features.append(toAppend)
            assert False
        except CalledFunctionException:
            pass

    def test_points_append_fromEmpty(self):
        """ Test points.append() to bottom when the calling object is point empty """
        self.backend_insert_emptyObject('point', 'append')

    def test_features_append_fromEmpty(self):
        """ Test features.append() to right when the calling object is feature empty """
        self.backend_insert_emptyObject('feature', 'append')

    def test_points_append_handmadeSingle(self):
        """ Test points.append() against handmade output for a single appended point"""
        self.backend_insert_handmadeSingle('point', None)

    def test_features_append_handmadeSingle(self):
        """ Test features.append() against handmade output for a single appended feature"""
        self.backend_insert_handmadeSingle('feature', None)

    def test_points_append_handmadeSequence(self):
        """ Test points.append() against handmade output for a sequence of additions"""
        self.backend_insert_handmadeSequence('point', None)

    def test_features_append_handmadeSequence(self):
        """ Test features.append() against handmade output for a sequence of additions"""
        self.backend_insert_handmadeSequence('feature', None)

    def test_points_append_selfAppend(self):
        self.backend_insert_selfInsert('point', None)

    def test_features_append_selfAppend(self):
        self.backend_insert_selfInsert('feature', None)

    def test_points_append_automaticReorder_fullySpecifiedNames(self):
        self.backend_insert_automaticReorder('point', False, None)

    def test_features_append_automaticReorder_fullySpecifiedNames(self):
        self.backend_insert_automaticReorder('feature', False, None)

    def test_points_append_automaticReorder_defaultPointNames(self):
        self.backend_insert_automaticReorder('point', True, None)

    def test_features_append_automaticReorder_defaultFeatureNames(self):
        self.backend_insert_automaticReorder('feature', True, None)

    #################
    # points.sort() #
    #################

    @raises(InvalidArgumentTypeCombination)
    def test_points_sort_exceptionAtLeastOne(self):
        """ Test points.sort() has at least one parameter """
        data = [[7, 8, 9], [1, 2, 3], [4, 5, 6]]
        toTest = self.constructor(data)

        toTest.points.sort()

    @raises(InvalidArgumentTypeCombination)
    def test_points_sort_exceptionBothNotNone(self):
        """ Test points.sort() has only one parameter """
        data = [[7, 8, 9], [1, 2, 3], [4, 5, 6]]
        toTest = self.constructor(data)

        toTest.points.sort(sortBy=1, sortHelper=[1,2,0])

    @oneLogEntryExpected
    def test_points_sort_naturalByFeature(self):
        """ Test points.sort() when we specify a feature to sort by """
        data = [[1, 2, 3], [7, 1, 9], [4, 5, 6]]
        names = ['1', '7', '4']
        toTest = self.constructor(data, pointNames=names)

        ret = toTest.points.sort(sortBy=1) # RET CHECK

        dataExpected = [[7, 1, 9], [1, 2, 3], [4, 5, 6]]
        namesExp = ['7', '1', '4']
        objExp = self.constructor(dataExpected, pointNames=namesExp)

        assert toTest.isIdentical(objExp)
        assert ret is None

    def test_points_sort_naturalByFeatureName(self):
        """ Test points.sort() when we specify a feature name to sort by """
        data = [[1, 2, 3], [7, 1, 9], [4, 5, 6]]
        pnames = ['1', '7', '4']
        fnames = ['1', '2', '3']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)

        ret = toTest.points.sort(sortBy='2') # RET CHECK

        dataExpected = [[7, 1, 9], [1, 2, 3], [4, 5, 6]]
        namesExp = ['7', '1', '4']
        objExp = self.constructor(dataExpected, pointNames=namesExp, featureNames=fnames)

        assert toTest.isIdentical(objExp)
        assert ret is None

    @oneLogEntryExpected
    def test_points_sort_scorer(self):
        """ Test points.sort() when we specify a scoring function """
        data = [[1, 2, 3], [4, 5, 6], [7, 1, 9], [0, 0, 0]]
        toTest = self.constructor(data)

        def numOdds(point):
            assert isinstance(point, BaseView)
            ret = 0
            for val in point:
                if val % 2 != 0:
                    ret += 1
            return ret

        toTest.points.sort(sortHelper=numOdds)

        dataExpected = [[0, 0, 0], [4, 5, 6], [1, 2, 3], [7, 1, 9]]
        objExp = self.constructor(dataExpected)

        assert toTest.isIdentical(objExp)
        assertNoNamesGenerated(toTest)

    @oneLogEntryExpected
    def test_points_sort_comparator(self):
        """ Test points.sort() when we specify a comparator function """
        data = [[1, 2, 3], [4, 5, 6], [7, 1, 9], [0, 0, 0]]
        toTest = self.constructor(data)

        def compOdds(point1, point2):
            odds1 = 0
            odds2 = 0
            for val in point1:
                if val % 2 != 0:
                    odds1 += 1
            for val in point2:
                if val % 2 != 0:
                    odds2 += 1
            return odds1 - odds2

        toTest.points.sort(sortHelper=compOdds)

        dataExpected = [[0, 0, 0], [4, 5, 6], [1, 2, 3], [7, 1, 9]]
        objExp = self.constructor(dataExpected)

        assert toTest.isIdentical(objExp)
        assertNoNamesGenerated(toTest)

    def test_points_sort_dataTypeRetainedFromList(self):
        """ Test points.sort() data not converted when sorting by list"""
        data = [['a', 2, 3.0], ['b', 5, 6.0], ['c', 8, 9.0]]
        toTest = self.constructor(data)

        toTest.points.sort(sortHelper=[2, 1, 0])

        expData = [['c', 8, 9.0], ['b', 5, 6.0], ['a', 2, 3.0]]
        exp = self.constructor(expData)

        assert toTest == exp
        assertNoNamesGenerated(toTest)

    def test_points_sort_indicesList(self):
        """ Test points.sort() when we specify a list of indices """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        toTest = self.constructor(data)

        toTest.points.sort(sortHelper=[2, 1, 0])

        expData = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        exp = self.constructor(expData)

        assert toTest == exp

    def test_points_sort_namesList(self):
        """ Test points.sort() when we specify a list of point names """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        pnames = ['3', '6', '9']
        toTest = self.constructor(data, pointNames=pnames)

        toTest.points.sort(sortHelper=['9', '6', '3'])

        expData = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        expNames = ['9', '6', '3']
        exp = self.constructor(expData, pointNames=expNames)

        assert toTest == exp

    @oneLogEntryExpected
    def test_points_sort_mixedList(self):
        """ Test points.sort() when we specify a mixed list (names/indices) """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        pnames = ['3', '6', '9']
        toTest = self.constructor(data, pointNames=pnames)

        toTest.points.sort(sortHelper=['9', '6', 0])

        expData = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        expNames = ['9', '6', '3']
        exp = self.constructor(expData, pointNames=expNames)

        assert toTest == exp

    @raises(IndexError)
    def test_points_sort_exceptionIndicesPEmpty(self):
        """ tests points.sort() throws an IndexError when given invalid indices """
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)
        toTest.points.sort(sortHelper=[1, 3])

    @raises(InvalidArgumentValue)
    def test_points_sort_exceptionIndicesSmall(self):
        """ tests points.sort() throws an InvalidArgumentValue when given an incorrectly sized indices list """
        data = [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
        toTest = self.constructor(data)

        toTest.points.sort(sortHelper=[1, 0])

    @raises(InvalidArgumentValue)
    def test_points_sort_exceptionNotUniqueIds(self):
        """ tests points.sort() throws an InvalidArgumentValue when given duplicate indices """
        data = [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
        toTest = self.constructor(data)
        toTest.points.sort(sortHelper=[1, 1, 0])

    #################
    # features.sort() #
    #################

    @raises(InvalidArgumentTypeCombination)
    def test_features_sort_exceptionAtLeastOne(self):
        """ Test features.sort() has at least one paramater """
        data = [[7, 8, 9], [1, 2, 3], [4, 5, 6]]
        toTest = self.constructor(data)

        toTest.features.sort()

    @raises(InvalidArgumentTypeCombination)
    def test_features_sort_exceptionBothNotNone(self):
        """ Test points.sort() has only one parameter """
        data = [[7, 8, 9], [1, 2, 3], [4, 5, 6]]
        toTest = self.constructor(data)

        toTest.features.sort(sortBy=1, sortHelper=[1,2,0])

    @oneLogEntryExpected
    def test_features_sort_naturalByPointWithNames(self):
        """ Test features.sort() when we specify a point to sort by; includes featureNames """
        data = [[1, 2, 3], [7, 1, 9], [4, 5, 6]]
        names = ["1", "2", "3"]
        toTest = self.constructor(data, featureNames=names)

        ret = toTest.features.sort(sortBy=1) # RET CHECK

        dataExpected = [[2, 1, 3], [1, 7, 9], [5, 4, 6]]
        namesExp = ["2", "1", "3"]
        objExp = self.constructor(dataExpected, featureNames=namesExp)

        assert toTest.isIdentical(objExp)
        assert ret is None

    def test_features_sort_naturalByPointNameWithFNames(self):
        """ Test features.sort() when we specify a point name to sort by; includes featureNames """
        data = [[1, 2, 3], [7, 1, 9], [4, 5, 6]]
        pnames = ['1', '7', '4']
        fnames = ["1", "2", "3"]
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)

        ret = toTest.features.sort(sortBy='7') # RET CHECK

        dataExpected = [[2, 1, 3], [1, 7, 9], [5, 4, 6]]
        namesExp = ["2", "1", "3"]
        objExp = self.constructor(dataExpected, pointNames=pnames, featureNames=namesExp)

        assert toTest.isIdentical(objExp)
        assert ret is None

    @oneLogEntryExpected
    def test_features_sort_scorer(self):
        """ Test features.sort() when we specify a scoring function """
        data = [[7, 1, 9, 0], [1, 2, 3, 0], [4, 2, 9, 0]]
        names = ["2", "1", "3", "0"]
        toTest = self.constructor(data, featureNames=names)

        def numOdds(feature):
            ret = 0
            for val in feature:
                if val % 2 != 0:
                    ret += 1
            return ret

        toTest.features.sort(sortHelper=numOdds)

        dataExpected = [[0, 1, 7, 9], [0, 2, 1, 3], [0, 2, 4, 9]]
        namesExp = ['0', '1', '2', '3']
        objExp = self.constructor(dataExpected, featureNames=namesExp)

        assert toTest.isIdentical(objExp)

    @oneLogEntryExpected
    def test_features_sort_comparator(self):
        """ Test features.sort() when we specify a comparator function """
        data = [[7, 1, 9, 0], [1, 2, 3, 0], [4, 2, 9, 0]]
        toTest = self.constructor(data)

        def compOdds(point1, point2):
            odds1 = 0
            odds2 = 0
            for val in point1:
                if val % 2 != 0:
                    odds1 += 1
            for val in point2:
                if val % 2 != 0:
                    odds2 += 1
            return odds1 - odds2

        toTest.features.sort(sortHelper=compOdds)

        dataExpected = [[0, 1, 7, 9], [0, 2, 1, 3], [0, 2, 4, 9]]
        objExp = self.constructor(dataExpected)

        assert toTest.isIdentical(objExp)
        assertNoNamesGenerated(toTest)

    def test_features_sort_dataTypeRetainedFromList(self):
        """ Test features.sort() data not converted when sorting by list"""
        data = [['a', 2, 3.0], ['b', 5, 6.0], ['c', 8, 9.0]]
        toTest = self.constructor(data)

        toTest.features.sort(sortHelper=[2, 1, 0])

        expData = [[3.0, 2, 'a'], [6.0, 5, 'b'], [9.0, 8, 'c']]
        exp = self.constructor(expData)

        assert toTest == exp
        assertNoNamesGenerated(toTest)

    def test_features_sort_indicesList(self):
        """ Test features.sort() when we specify a list of indices """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        toTest = self.constructor(data)

        toTest.features.sort(sortHelper=[2, 1, 0])

        expData = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        exp = self.constructor(expData)

        assert toTest == exp

    def test_features_sort_namesList(self):
        """ Test features.sort() when we specify a list of feature names """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        fnames = ['third', 'second', 'first']
        toTest = self.constructor(data, featureNames=fnames)

        toTest.features.sort(sortHelper=['first', 'second', 'third'])

        expData = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expNames = ['first', 'second', 'third']
        exp = self.constructor(expData, featureNames=expNames)

        assert toTest == exp

    @oneLogEntryExpected
    def test_features_sort_mixedList(self):
        """ Test features.sort() when we specify a mixed list (names/indices) """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        fnames = ['third', 'second', 'first']
        toTest = self.constructor(data, featureNames=fnames)

        toTest.features.sort(sortHelper=['first', 'second', 0])

        expData = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expNames = ['first', 'second', 'third']
        exp = self.constructor(expData, featureNames=expNames)

        assert toTest == exp

    @raises(IndexError)
    def test_features_sort_exceptionIndicesFEmpty(self):
        """ tests features.sort() throws an IndexError when given invalid indices """
        data = [[], []]
        data = numpy.array(data)
        toTest = self.constructor(data)
        toTest.features.sort(sortHelper=[1, 3])


    @raises(InvalidArgumentValue)
    def test_features_sort_exceptionIndicesSmall(self):
        """ tests features.sort() throws an InvalidArgumentValue when given an incorrectly sized indices list """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        toTest = self.constructor(data)

        toTest.features.sort(sortHelper=[1, 0])


    @raises(InvalidArgumentValue)
    def test_features_sort_exceptionNotUniqueIds(self):
        """ tests features.sort() throws an InvalidArgumentValue when given duplicate indices """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        data = numpy.array(data)
        toTest = self.constructor(data)
        toTest.features.sort(sortHelper=[1, 1, 0])

    ##################
    # points.extract #
    ##################

    @raises(CalledFunctionException)
    @mock.patch('nimble.data.axis.constructIndicesList', calledException)
    def test_points_extract_calls_constructIndicesList(self):
        toTest = self.constructor([[1,2,],[3,4]], pointNames=['a', 'b'])

        ret = toTest.points.extract(['a', 'b'])

    @oneLogEntryExpected
    def test_points_extract_handmadeSingle(self):
        """ Test points.extract() against handmade output when extracting one point """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ext1 = toTest.points.extract(0)
        exp1 = self.constructor([[1, 2, 3]])
        assert ext1.isIdentical(exp1)
        expEnd = self.constructor([[4, 5, 6], [7, 8, 9]])
        assert toTest.isIdentical(expEnd)

        # Check that names have not been generated unnecessarily
        assertNoNamesGenerated(toTest)
        assertNoNamesGenerated(ext1)

    def test_points_extract_index_NamePath_Preserve(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = 'testName'
        toTest._absPath = 'testAbsPath'
        toTest._relPath = 'testRelPath'

        ext1 = toTest.points.extract(0)

        assert ext1.nameIsDefault()
        assert ext1.path == 'testAbsPath'
        assert ext1.absolutePath == 'testAbsPath'
        assert ext1.relativePath == 'testRelPath'

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

    def test_points_extract_ListIntoPEmpty(self):
        """ Test points.extract() by removing a list of all points """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        expRet = self.constructor(data)
        ret = toTest.points.extract([0, 1, 2, 3])

        assert ret.isIdentical(expRet)

        data = [[], [], []]
        data = numpy.array(data).T
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    @twoLogEntriesExpected
    def test_points_extract_handmadeListSequence(self):
        """ Test points.extract() against handmade output for several list extractions """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        names = ['1', '4', '7', '10']
        toTest = self.constructor(data, pointNames=names)
        ext1 = toTest.points.extract('1')
        exp1 = self.constructor([[1, 2, 3]], pointNames=['1'])
        assert ext1.isIdentical(exp1)
        ext2 = toTest.points.extract([1, 2])
        exp2 = self.constructor([[7, 8, 9], [10, 11, 12]], pointNames=['7', '10'])
        assert ext2.isIdentical(exp2)
        expEnd = self.constructor([[4, 5, 6]], pointNames=['4'])
        assert toTest.isIdentical(expEnd)

    def test_points_extract_handmadeListOrdering(self):
        """ Test points.extract() against handmade output for out of order extraction """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
        names = ['1', '4', '7', '10', '13']
        toTest = self.constructor(data, pointNames=names)
        ext1 = toTest.points.extract([3, 4, 1])
        exp1 = self.constructor([[10, 11, 12], [13, 14, 15], [4, 5, 6]], pointNames=['10', '13', '4'])
        assert ext1.isIdentical(exp1)
        expEnd = self.constructor([[1, 2, 3], [7, 8, 9]], pointNames=['1', '7'])
        assert toTest.isIdentical(expEnd)

    def test_points_extract_List_trickyOrdering(self):
        data = [[0], [2], [2], [2], [0], [0], [0], [0], [2], [0]]
        toExtract = [6, 5, 3, 9]

        toTest = self.constructor(data)

        ret = toTest.points.extract(toExtract)

        expRaw = [[0], [0], [2], [0]]
        expRet = self.constructor(expRaw)

        expRaw = [[0], [2], [2], [0], [0], [2]]
        expRem = self.constructor(expRaw)

        assert ret == expRet
        assert toTest == expRem

    def test_points_extract_function_selectionGap(self):
        data = [[0], [2], [2], [2], [0], [0], [0], [0], [2], [0]]
        extractIndices = [3, 5, 6, 9]
        pnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        def sel(point):
            if int(point.points.getName(0)) in extractIndices:
                return True
            else:
                return False

        toTest = self.constructor(data, pointNames=pnames)

        ret = toTest.points.extract(sel)

        expRaw = [[2], [0], [0], [0]]
        expNames = ['3', '5', '6', '9']
        expRet = self.constructor(expRaw, pointNames=expNames)

        expRaw = [[0], [2], [2], [0], [0], [2]]
        expNames = ['0', '1', '2', '4', '7', '8']
        expRem = self.constructor(expRaw, pointNames=expNames)

        assert ret == expRet
        assert toTest == expRem


    def test_points_extract_functionIntoPEmpty(self):
        """ Test points.extract() by removing all points using a function """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        expRet = self.constructor(data)

        ret = toTest.points.extract(allTrue)
        assert ret.isIdentical(expRet)

        data = [[], [], []]
        data = numpy.array(data).T
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    def test_points_extract_function_returnPointEmpty(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        exp = self.constructor(data)

        ret = toTest.points.extract(allFalse)

        data = [[], [], []]
        data = numpy.array(data).T
        expRet = self.constructor(data)

        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(exp)

    def test_points_extract_handmadeFunction(self):
        """ Test points.extract() against handmade output for function extraction """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        ext = toTest.points.extract(oneOrFour)
        exp = self.constructor([[1, 2, 3], [4, 5, 6]])
        assert ext.isIdentical(exp)
        expEnd = self.constructor([[7, 8, 9]])
        assert toTest.isIdentical(expEnd)

    def test_points_extract_func_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        ext = toTest.points.extract(oneOrFour)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

        assert ext.nameIsDefault()
        assert ext.absolutePath == 'testAbsPath'
        assert ext.relativePath == 'testRelPath'

    def test_points_extract_handmadeFuncionWithFeatureNames(self):
        """ Test points.extract() against handmade output for function extraction with featureNames"""
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)

        ext = toTest.points.extract(oneOrFour)
        exp = self.constructor([[1, 2, 3], [4, 5, 6]], featureNames=featureNames)
        assert ext.isIdentical(exp)
        expEnd = self.constructor([[7, 8, 9]], featureNames=featureNames)
        assert toTest.isIdentical(expEnd)


    @raises(InvalidArgumentType)
    def test_points_extract_exceptionStartInvalidType(self):
        """ Test points.extract() for InvalidArgumentType when start is not a valid ID type """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.extract(start=1.1, end=2)

    @raises(IndexError)
    def test_points_extract_exceptionEndInvalid(self):
        """ Test points.extract() for IndexError when end is not a valid Point index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.extract(start=1, end=5)

    @raises(InvalidArgumentValueCombination)
    def test_points_extract_exceptionInversion(self):
        """ Test points.extract() for InvalidArgumentValueCombination when start comes after end """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.extract(start=2, end=0)

    @raises(InvalidArgumentValueCombination)
    def test_points_extract_exceptionInversionPointName(self):
        """ Test points.extract() for InvalidArgumentValueCombination when start comes after end as FeatureNames"""
        pointNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames)
        toTest.points.extract(start="two", end="one")

    @raises(InvalidArgumentValue)
    def test_points_extract_exceptionDuplicates(self):
        """ Test points.extract() for InvalidArgumentValueCombination when toExtract contains duplicates """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.points.extract([0, 1, 0])

    def test_points_extract_handmadeRange(self):
        """ Test points.extract() against handmade output for range extraction """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.points.extract(start=1, end=2)

        expectedRet = self.constructor([[4, 5, 6], [7, 8, 9]])
        expectedTest = self.constructor([[1, 2, 3]])

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_points_extract_range_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        ret = toTest.points.extract(start=1, end=2)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

        assert ret.nameIsDefault()
        assert ret.absolutePath == 'testAbsPath'
        assert ret.relativePath == 'testRelPath'


    def test_points_extract_rangeIntoPEmpty(self):
        """ Test points.extract() removes all points using ranges """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract(start=0, end=2)

        assert ret.isIdentical(expRet)

        data = [[], [], []]
        data = numpy.array(data).T
        exp = self.constructor(data, featureNames=featureNames)

        assert toTest.isIdentical(exp)


    def test_points_extract_handmadeRangeWithFeatureNames(self):
        """ Test points.extract() against handmade output for range extraction with featureNames """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract(start=1, end=2)

        expectedRet = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=['4', '7'], featureNames=featureNames)
        expectedTest = self.constructor([[1, 2, 3]], pointNames=['1'], featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_points_extract_handmadeRangeRand_FM(self):
        """ Test points.extract() for correct sizes when using randomized range extraction and featureNames """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.points.extract(start=0, end=2, number=2, randomize=True)

        assert len(ret.points) == 2
        assert len(toTest.points) == 1

    def test_points_extract_handmadeRangeDefaults(self):
        """ Test points.extract uses the correct defaults in the case of range based extraction """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract(end=1)

        expectedRet = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=['1', '4'], featureNames=featureNames)
        expectedTest = self.constructor([[7, 8, 9]], pointNames=['7'], featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract(start=1)

        expectedTest = self.constructor([[1, 2, 3]], pointNames=['1'], featureNames=featureNames)
        expectedRet = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=['4', '7'], featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_points_extract_handmade_calling_pointNames(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract(start='4', end='7')

        expectedRet = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_points_extract_handmadeString(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('one=1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('one==1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('one<2')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('one<=1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('one>4')
        expectedRet = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)
        expectedTest = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=pointNames[:-1], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('one>=7')
        expectedRet = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)
        expectedTest = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=pointNames[:-1], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('one!=4')
        expectedRet = self.constructor([[1, 2, 3], [7, 8, 9]], pointNames=[pointNames[0], pointNames[-1]],
                                       featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6]], pointNames=[pointNames[1]], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('one<1')
        expectedRet = self.constructor([], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('one>0')
        expectedRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expectedTest = self.constructor([], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_points_extract_handmadeStringWithOperatorWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('one = 1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('one == 1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('one < 2')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('one <= 1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('one > 4')
        expectedRet = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)
        expectedTest = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=pointNames[:-1], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('one >= 7')
        expectedRet = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)
        expectedTest = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=pointNames[:-1], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('one != 4')
        expectedRet = self.constructor([[1, 2, 3], [7, 8, 9]], pointNames=[pointNames[0], pointNames[-1]],
                                       featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6]], pointNames=[pointNames[1]], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('one < 1')
        expectedRet = self.constructor([], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('one > 0')
        expectedRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expectedTest = self.constructor([], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_points_extract_handmadeStringWithFeatureWhitespace(self):
        featureNames = ["feature one", "feature two", "feature three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('feature one=1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName=value with operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('feature one = 1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_points_extract_list_mixed(self):
        """ Test points.extract() list input with mixed names and indices """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        names = ['1', '4', '7', '10']
        toTest = self.constructor(data, pointNames=names)
        ret = toTest.points.extract(['1',1,-1])
        expRet = self.constructor([[1, 2, 3], [4, 5, 6], [10, 11, 12]], pointNames=['1','4','10'])
        expTest = self.constructor([[7, 8, 9]], pointNames=['7'])
        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(expTest)

    @raises(InvalidArgumentValue)
    def test_points_extract_handmadeString_featureNotExist(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.points.extract('four=1')

    def test_points_extract_numberOnly(self):
        self.back_extract_numberOnly('point')

    def test_points_extract_functionAndNumber(self):
        self.back_extract_functionAndNumber('point')

    def test_points_extract_numberAndRandomizeAllData(self):
        self.back_extract_numberAndRandomizeAllData('point')

    def test_points_extract_numberAndRandomizeSelectedData(self):
        self.back_extract_numberAndRandomizeSelectedData('point')

    @raises(InvalidArgumentValueCombination)
    def test_points_extract_randomizeNoNumber(self):
        self.back_structural_randomizeNoNumber('extract', 'point')

    @raises(InvalidArgumentValue)
    def test_points_extract_list_numberGreaterThanTargeted(self):
        self.back_structural_list_numberGreaterThanTargeted('extract', 'point')

    @raises(InvalidArgumentValue)
    def test_points_extract_function_numberGreaterThanTargeted(self):
        self.back_structural_function_numberGreaterThanTargeted('extract', 'point')

    @raises(InvalidArgumentValue)
    def test_points_extract_range_numberGreaterThanTargeted(self):
        self.back_structural_range_numberGreaterThanTargeted('extract', 'point')

    ### using match module ###

    def test_points_extract_match_missing(self):
        toTest = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.points.extract(match.anyMissing)
        expTest = self.constructor([[1, 2, 3], [7, 8, 9]])
        expRet = self.constructor([[None, 11, None], [7, 11, None]])
        expTest.features.setNames(['a', 'b', 'c'])
        expRet.features.setNames(['a', 'b', 'c'])
        assert toTest == expTest
        assert ret == expRet

        toTest = self.constructor([[None, None, None], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.points.extract(match.allMissing)
        expTest = self.constructor([[None, 11, None], [7, 11, None], [7, 8, 9]])
        expRet = self.constructor([[None, None, None]])
        expTest.features.setNames(['a', 'b', 'c'])
        expRet.features.setNames(['a', 'b', 'c'])
        assert toTest == expTest
        assert ret == expRet

    def test_points_extract_match_nonNumeric(self):
        toTest = self.constructor([[1, 2, 3], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.points.extract(match.anyNonNumeric)
        expTest = self.constructor([[1, 2, 3], [7, 8, 9]])
        expRet = self.constructor([['a', 11, 'c'], [7, 11, 'c']])
        expTest.features.setNames(['a', 'b', 'c'])
        expRet.features.setNames(['a', 'b', 'c'])
        assert toTest == expTest
        assert ret == expRet

        toTest = self.constructor([['a', 'x', 'c'], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.points.extract(match.allNonNumeric)
        expTest = self.constructor([['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]])
        expRet = self.constructor([['a', 'x', 'c']])
        expTest.features.setNames(['a', 'b', 'c'])
        expRet.features.setNames(['a', 'b', 'c'])
        assert toTest == expTest
        assert ret == expRet

    def test_points_extract_match_list(self):
        toTest = self.constructor([[1, 2, 3], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.points.extract(match.anyValues(['a', 'c', 'x']))
        expTest = self.constructor([[1, 2, 3], [7, 8, 9]])
        expRet = self.constructor([['a', 11, 'c'], [7, 11, 'c']])
        expTest.features.setNames(['a', 'b', 'c'])
        expRet.features.setNames(['a', 'b', 'c'])
        assert toTest == expTest
        assert ret == expRet

        toTest = self.constructor([['a', 'x', 'c'], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.points.extract(match.allValues(['a', 'c', 'x']))
        expTest = self.constructor([['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]])
        expRet = self.constructor([['a', 'x', 'c']])
        expTest.features.setNames(['a', 'b', 'c'])
        expRet.features.setNames(['a', 'b', 'c'])
        assert toTest == expTest
        assert ret == expRet

    def test_points_extract_match_function(self):
        toTest = self.constructor([[1, 2, 3], [-1, 11, -3], [7, 11, -3], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.points.extract(match.anyValues(lambda x: x < 0))
        expTest = self.constructor([[1, 2, 3], [7, 8, 9]])
        expRet = self.constructor([[-1, 11, -3], [7, 11, -3]])
        expTest.features.setNames(['a', 'b', 'c'])
        expRet.features.setNames(['a', 'b', 'c'])
        assert toTest == expTest
        assert ret == expRet

        toTest = self.constructor([[-1, -2, -3], [-1, 11, -3], [7, 11, -3], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.points.extract(match.allValues(lambda x: x < 0))
        expTest = self.constructor([[-1, 11, -3], [7, 11, -3], [7, 8, 9]])
        expRet = self.constructor([[-1, -2, -3]])
        expTest.features.setNames(['a', 'b', 'c'])
        expRet.features.setNames(['a', 'b', 'c'])
        assert toTest == expTest
        assert ret == expRet

    ##########################
    # extract common backend #
    ##########################

    def back_extract_numberOnly(self, axis):
        if axis == 'point':
            toCall = "points"
        else:
            toCall = "features"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        ret = getattr(toTest, toCall).extract(number=3)
        if axis == 'point':
            exp = self.constructor(data[:3], pointNames=pnames[:3], featureNames=fnames)
            rem = self.constructor(data[3:], pointNames=pnames[3:], featureNames=fnames)
        else:
            exp = self.constructor([p[:3] for p in data], pointNames=pnames, featureNames=fnames[:3])
            rem = self.constructor([p[3:] for p in data], pointNames=pnames, featureNames=fnames[3:])

        assert exp.isIdentical(ret)
        assert rem.isIdentical(toTest)

    def back_extract_functionAndNumber(self, axis):
        if axis == 'point':
            toCall = "points"
        else:
            toCall = "features"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        ret = getattr(toTest, toCall).extract(allTrue, number=2)
        if axis == 'point':
            exp = self.constructor(data[:2], pointNames=pnames[:2], featureNames=fnames)
            rem = self.constructor(data[2:], pointNames=pnames[2:], featureNames=fnames)
        else:
            exp = self.constructor([p[:2] for p in data], pointNames=pnames, featureNames=fnames[:2])
            rem = self.constructor([p[2:] for p in data], pointNames=pnames, featureNames=fnames[2:])

        assert exp.isIdentical(ret)
        assert rem.isIdentical(toTest)

    def back_extract_numberAndRandomizeAllData(self, axis):
        """test that randomizing (with same randomly chosen seed) and limiting to a
        given number provides the same result for all input types if using all the data
        """
        if axis == 'point':
            toCall = "points"
        else:
            toCall = "features"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest1 = self.constructor(data, pointNames=pnames, featureNames=fnames)
        toTest2 = toTest1.copy()
        toTest3 = toTest1.copy()
        toTest4 = toTest1.copy()

        seed = nimble.randomness.generateSubsidiarySeed()
        nimble.randomness.startAlternateControl(seed)
        ret = getattr(toTest1, toCall).extract(number=3, randomize=True)
        nimble.randomness.endAlternateControl()

        nimble.randomness.startAlternateControl(seed)
        retList = getattr(toTest2, toCall).extract([0, 1, 2, 3], number=3, randomize=True)
        nimble.randomness.endAlternateControl()

        nimble.randomness.startAlternateControl(seed)
        retRange = getattr(toTest3, toCall).extract(start=0, end=3, number=3, randomize=True)
        nimble.randomness.endAlternateControl()

        nimble.randomness.startAlternateControl(seed)
        retFunc = getattr(toTest4, toCall).extract(allTrue, number=3, randomize=True)
        nimble.randomness.endAlternateControl()

        if axis == 'point':
            assert len(ret.points) == 3
            assert len(toTest1.points) == 1
        else:
            assert len(ret.features) == 3
            assert len(toTest1.features) == 1

        assert ret.isIdentical(retList)
        assert ret.isIdentical(retRange)
        assert ret.isIdentical(retFunc)
        assert toTest1.isIdentical(toTest2)
        assert toTest1.isIdentical(toTest3)
        assert toTest1.isIdentical(toTest4)

    def back_extract_numberAndRandomizeSelectedData(self, axis):
        """test that randomization occurs after the data has been selected from the user inputs """
        if axis == 'point':
            toCall = "points"
        else:
            toCall = "features"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest1 = self.constructor(data, pointNames=pnames, featureNames=fnames)
        toTest2 = toTest1.copy()
        toTest3 = toTest1.copy()
        if axis == 'point':
            expRet1 = toTest1[1, :]
            expRet2 = toTest1[2, :]
            expTest1 = toTest1[[0, 1, 3], :]
            expTest2 = toTest1[[0, 2, 3], :]
        else:
            expRet1 = toTest1[:, 1]
            expRet2 = toTest1[:, 2]
            expTest1 = toTest1[:, [0, 1, 3]]
            expTest2 = toTest1[:, [0, 2, 3]]

        seed = nimble.randomness.generateSubsidiarySeed()
        nimble.randomness.startAlternateControl(seed)
        retList = getattr(toTest1, toCall).extract([1, 2], number=1, randomize=True)
        nimble.randomness.endAlternateControl()

        nimble.randomness.startAlternateControl(seed)
        retRange = getattr(toTest2, toCall).extract(start=1, end=2, number=1, randomize=True)
        nimble.randomness.endAlternateControl()

        def middleRowsOrCols(value):
            return value[0] in [2, 4, 5, 7]

        nimble.randomness.startAlternateControl(seed)
        retFunc = getattr(toTest3, toCall).extract(middleRowsOrCols, number=1, randomize=True)
        nimble.randomness.endAlternateControl()

        assert retList.isIdentical(expRet1) or retList.isIdentical(expRet2)
        assert retRange.isIdentical(expRet1) or retList.isIdentical(expRet2)
        assert retFunc.isIdentical(expRet1) or retList.isIdentical(expRet2)

        assert toTest1.isIdentical(expTest1) or toTest1.isIdentical(expTest2)
        assert toTest2.isIdentical(expTest1) or toTest2.isIdentical(expTest2)
        assert toTest3.isIdentical(expTest1) or toTest3.isIdentical(expTest2)

    ######################
    # features.extract() #
    ######################

    @raises(CalledFunctionException)
    @mock.patch('nimble.data.axis.constructIndicesList', calledException)
    def test_features_extract_calls_constructIndicesList(self):
        toTest = self.constructor([[1,2,],[3,4]], featureNames=['a', 'b'])

        ret = toTest.features.extract(['a', 'b'])

    @oneLogEntryExpected
    def test_features_extract_handmadeSingle(self):
        """ Test features.extract() against handmade output when extracting one feature """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ext1 = toTest.features.extract(0)
        exp1 = self.constructor([[1], [4], [7]])

        assert ext1.isIdentical(exp1)
        expEnd = self.constructor([[2, 3], [5, 6], [8, 9]])
        assert toTest.isIdentical(expEnd)

        # Check that names have not been generated unnecessarily
        assertNoNamesGenerated(toTest)
        assertNoNamesGenerated(ext1)

    def test_features_extract_List_NamePath_Preserve(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        ext1 = toTest.features.extract(0)

        assert toTest.path == 'testAbsPath'
        assert toTest.absolutePath == 'testAbsPath'
        assert toTest.relativePath == 'testRelPath'

        assert ext1.nameIsDefault()
        assert ext1.absolutePath == 'testAbsPath'
        assert ext1.relativePath == 'testRelPath'

    def test_features_extract_ListIntoFEmpty(self):
        """ Test features.extract() by removing a list of all features """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        expRet = self.constructor(data)
        ret = toTest.features.extract([0, 1, 2])

        assert ret.isIdentical(expRet)

        data = [[], [], [], []]
        data = numpy.array(data)
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    def test_features_extract_ListIntoFEmptyOutOfOrder(self):
        """ Test features.extract() by removing a list of all features """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        expData = [[3, 1, 2], [6, 4, 5], [9, 7, 8], [12, 10, 11]]
        expRet = self.constructor(expData)
        ret = toTest.features.extract([2, 0, 1])

        assert ret.isIdentical(expRet)

        data = [[], [], [], []]
        data = numpy.array(data)
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    @twoLogEntriesExpected
    def test_features_extract_handmadeListSequence(self):
        """ Test features.extract() against handmade output for several extractions by list """
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data, pointNames=pointNames)
        ext1 = toTest.features.extract([0])
        exp1 = self.constructor([[1], [4], [7]], pointNames=pointNames)
        assert ext1.isIdentical(exp1)
        ext2 = toTest.features.extract([2, 1])
        exp2 = self.constructor([[-1, 3], [-2, 6], [-3, 9]], pointNames=pointNames)
        assert ext2.isIdentical(exp2)
        expEndData = [[2], [5], [8]]
        expEnd = self.constructor(expEndData, pointNames=pointNames)
        assert toTest.isIdentical(expEnd)

    def test_features_extract_handmadeListWithFeatureName(self):
        """ Test features.extract() against handmade output for list extraction when specifying featureNames """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        toTest = self.constructor(data, featureNames=featureNames)
        ext1 = toTest.features.extract(["one"])
        exp1 = self.constructor([[1], [4], [7]], featureNames=["one"])
        assert ext1.isIdentical(exp1)
        ext2 = toTest.features.extract(["three", "neg"])
        exp2 = self.constructor([[3, -1], [6, -2], [9, -3]], featureNames=["three", "neg"])
        assert ext2.isIdentical(exp2)
        expEnd = self.constructor([[2], [5], [8]], featureNames=["two"])
        assert toTest.isIdentical(expEnd)


    def test_features_extract_List_trickyOrdering(self):
        data = [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
        toExtract = [6, 5, 3, 9]
        #		toExtract = [3,5,6,9]

        toTest = self.constructor(data)

        ret = toTest.features.extract(toExtract)

        expRaw = [0, 0, 1, 0]
        expRet = self.constructor(expRaw)

        expRaw = [0, 1, 1, 0, 0, 1]
        expRem = self.constructor(expRaw)

        assert ret == expRet
        assert toTest == expRem

    def test_features_extract_List_reorderingWithFeatureNames(self):
        data = [[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12]]
        fnames = ['a', 'b', 'c', 'd']
        test = self.constructor(data, featureNames=fnames)

        expRetRaw = [[1, 3, 2], [4, 6, 5], [7, 9, 8]]
        expRetNames = ['a', 'c', 'b']
        expRet = self.constructor(expRetRaw, featureNames=expRetNames)

        expTestRaw = [[10], [11], [12]]
        expTestNames = ['d']
        expTest = self.constructor(expTestRaw, featureNames=expTestNames)

        ret = test.features.extract(expRetNames)
        assert ret == expRet
        assert test == expTest


    def test_features_extract_function_selectionGap(self):
        data = [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
        fnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        extractIndices = [3, 5, 6, 9]

        def sel(feature):
            if int(feature.features.getName(0)) in extractIndices:
                return True
            else:
                return False

        toTest = self.constructor(data, featureNames=fnames)

        ret = toTest.features.extract(sel)

        expRaw = [1, 0, 0, 0]
        expNames = ['3', '5', '6', '9']
        expRet = self.constructor(expRaw, featureNames=expNames)

        expRaw = [0, 1, 1, 0, 0, 1]
        expNames = ['0', '1', '2', '4', '7', '8']
        expRem = self.constructor(expRaw, featureNames=expNames)

        assert ret == expRet
        assert toTest == expRem


    def test_features_extract_functionIntoFEmpty(self):
        """ Test features.extract() by removing all featuress using a function """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        expRet = self.constructor(data)

        ret = toTest.features.extract(allTrue)
        assert ret.isIdentical(expRet)

        data = [[], [], []]
        data = numpy.array(data)
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    def test_features_extract_function_returnPointEmpty(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        exp = self.constructor(data)

        ret = toTest.features.extract(allFalse)

        data = [[], [], []]
        data = numpy.array(data)
        expRet = self.constructor(data)

        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(exp)


    def test_features_extract_handmadeFunction(self):
        """ Test features.extract() against handmade output for function extraction """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data)

        ext = toTest.features.extract(absoluteOne)
        exp = self.constructor([[1, -1], [4, -2], [7, -3]])
        assert ext.isIdentical(exp)
        expEnd = self.constructor([[2, 3], [5, 6], [8, 9]])
        assert toTest.isIdentical(expEnd)


    def test_features_extract_func_NamePath_preservation(self):
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        ext = toTest.features.extract(absoluteOne)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

        assert ext.nameIsDefault()
        assert ext.absolutePath == 'testAbsPath'
        assert ext.relativePath == 'testRelPath'

    def test_features_extract_handmadeFunctionWithFeatureName(self):
        """ Test features.extract() against handmade output for function extraction with featureNames """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        pointNames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        ext = toTest.features.extract(absoluteOne)
        exp = self.constructor([[1, -1], [4, -2], [7, -3]], pointNames=pointNames, featureNames=['one', 'neg'])
        assert ext.isIdentical(exp)
        expEnd = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=["two", "three"])
        assert toTest.isIdentical(expEnd)

    @raises(InvalidArgumentType)
    def test_features_extract_exceptionStartInvalidType(self):
        """ Test features.extract() for InvalidArgumentType when start is not a valid ID type """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.extract(start=1.1, end=2)

    @raises(KeyError)
    def test_features_extract_exceptionStartInvalidFeatureName(self):
        """ Test features.extract() for KeyError when start is not a valid feature FeatureName """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.extract(start="wrong", end=2)

    @raises(IndexError)
    def test_features_extract_exceptionEndInvalid(self):
        """ Test features.extract() for IndexError when end is not a valid feature index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.extract(start=0, end=5)

    @raises(KeyError)
    def test_features_extract_exceptionEndInvalidFeatureName(self):
        """ Test features.extract() for KeyError when end is not a valid featureName """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.extract(start="two", end="five")

    @raises(InvalidArgumentValueCombination)
    def test_features_extract_exceptionInversion(self):
        """ Test features.extract() for InvalidArgumentValueCombination when start comes after end """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.extract(start=2, end=0)

    @raises(InvalidArgumentValueCombination)
    def test_features_extract_exceptionInversionFeatureName(self):
        """ Test features.extract() for InvalidArgumentValueCombination when start comes after end as FeatureNames"""
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.extract(start="two", end="one")

    @raises(InvalidArgumentValue)
    def test_features_extract_exceptionDuplicates(self):
        """ Test points.extract() for InvalidArgumentValueCombination when toExtract contains duplicates """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.features.extract([0, 1, 0])

    def test_features_extract_rangeIntoFEmpty(self):
        """ Test features.extract() removes all Featuress using ranges """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        expRet = self.constructor(data, featureNames=featureNames)
        ret = toTest.features.extract(start=0, end=2)

        assert ret.isIdentical(expRet)

        data = [[], [], []]
        data = numpy.array(data)
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    def test_features_extract_handmadeRange(self):
        """ Test features.extract() against handmade output for range extraction """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.features.extract(start=1, end=2)

        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]])
        expectedTest = self.constructor([[1], [4], [7]])

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_features_extract_range_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        ret = toTest.features.extract(start=1, end=2)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

        assert ret.nameIsDefault()
        assert ret.absolutePath == 'testAbsPath'
        assert ret.relativePath == 'testRelPath'


    def test_features_extract_handmadeWithFeatureNames(self):
        """ Test features.extract() against handmade output for range extraction with FeatureNames """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract(start=1, end=2)

        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=["two", "three"])
        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=["one"])

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_features_extract_handmade_calling_featureNames(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract(start="two", end="three")

        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=["two", "three"])
        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=["one"])

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_features_extract_handmadeString(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['p1', 'p2', 'p3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('p2=5')
        expectedRet = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])
        expectedTest = self.constructor([[1, 3], [4, 6], [7, 9]], pointNames=pointNames,
                                        featureNames=[featureNames[0], featureNames[-1]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('p1==1')
        expectedRet = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])
        expectedTest = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('p3<9')
        expectedRet = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('p3<=8')
        expectedRet = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('p3>8')
        expectedRet = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('p3>8.5')
        expectedRet = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('p1!=1.0')
        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])
        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('p1<1')
        expectedRet = self.constructor([], pointNames=pointNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('p1>0')
        expectedRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expectedTest = self.constructor([], pointNames=pointNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_features_extract_handmadeStringWithOperatorWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['p1', 'p2', 'p3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('p2 = 5')
        expectedRet = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])
        expectedTest = self.constructor([[1, 3], [4, 6], [7, 9]], pointNames=pointNames,
                                        featureNames=[featureNames[0], featureNames[-1]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('p1 == 1')
        expectedRet = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])
        expectedTest = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('p3 < 9')
        expectedRet = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('p3 <= 8')
        expectedRet = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('p3 > 8')
        expectedRet = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('p3 > 8.5')
        expectedRet = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('p1 != 1.0')
        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])
        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('p1 < 1')
        expectedRet = self.constructor([], pointNames=pointNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('p1 > 0')
        expectedRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expectedTest = self.constructor([], pointNames=pointNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_features_extract_handmadeStringWithPointWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['pt 1', 'pt 2', 'pt 3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value with no operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('pt 2=5')
        expectedRet = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])
        expectedTest = self.constructor([[1, 3], [4, 6], [7, 9]], pointNames=pointNames,
                                        featureNames=[featureNames[0], featureNames[-1]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test pointName=value with operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('pt 2 = 5')
        expectedRet = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])
        expectedTest = self.constructor([[1, 3], [4, 6], [7, 9]], pointNames=pointNames,
                                        featureNames=[featureNames[0], featureNames[-1]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_features_extract_list_mixed(self):
        """ Test features.extract() list input with mixed names and indices """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.features.extract([1, "three", -1])
        expRet = self.constructor([[2, 3, -1], [5, 6, -2], [8, 9, -3]], featureNames=["two", "three", "neg"])
        expTest = self.constructor([[1], [4], [7]], featureNames=["one"])
        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(expTest)

    @raises(InvalidArgumentValue)
    def test_features_extract_handmadeString_pointNotExist(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.features.extract('5=1')

    def test_features_extract_numberOnly(self):
        self.back_extract_numberOnly('feature')

    def test_features_extract_functionAndNumber(self):
        self.back_extract_functionAndNumber('feature')

    def test_features_extract_numberAndRandomizeAllData(self):
        self.back_extract_numberAndRandomizeAllData('feature')

    def test_features_extract_numberAndRandomizeSelectedData(self):
        self.back_extract_numberAndRandomizeSelectedData('feature')

    @raises(InvalidArgumentValueCombination)
    def test_features_extract_randomizeNoNumber(self):
        self.back_structural_randomizeNoNumber('extract', 'feature')

    @raises(InvalidArgumentValue)
    def test_features_extract_list_numberGreaterThanTargeted(self):
        self.back_structural_list_numberGreaterThanTargeted('extract', 'feature')

    @raises(InvalidArgumentValue)
    def test_features_extract_function_numberGreaterThanTargeted(self):
        self.back_structural_function_numberGreaterThanTargeted('extract', 'feature')

    @raises(InvalidArgumentValue)
    def test_features_extract_range_numberGreaterThanTargeted(self):
        self.back_structural_range_numberGreaterThanTargeted('extract', 'feature')

    ### using match module ###

    def test_features_extract_match_missing(self):
        toTest = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.features.extract(match.anyMissing)
        expTest = self.constructor([[2], [11], [11], [8]])
        expRet = self.constructor([[1, 3], [None, None], [7, None], [7, 9]])
        expTest.features.setNames(['b'])
        expRet.features.setNames(['a', 'c'])
        assert toTest == expTest
        assert ret == expRet

        toTest = self.constructor([[1, 2, None], [None, 11, None], [7, 11, None], [7, 8, None]], featureNames=['a', 'b', 'c'])
        ret = toTest.features.extract(match.allMissing)
        expTest = self.constructor([[1, 2], [None, 11], [7, 11], [7, 8]])
        expRet = self.constructor([[None], [None], [None], [None]])
        expTest.features.setNames(['a', 'b'])
        expRet.features.setNames(['c'])
        assert toTest == expTest
        assert ret == expRet

    def test_features_extract_match_nonNumeric(self):
        toTest = self.constructor([[1, 2, 3], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.features.extract(match.anyNonNumeric)
        expTest = self.constructor([[2], [11], [11], [8]])
        expRet = self.constructor([[1, 3], ['a', 'c'], [7, 'c'], [7, 9]])
        expTest.features.setNames(['b'])
        expRet.features.setNames(['a', 'c'])
        assert toTest == expTest
        assert ret == expRet

        toTest = self.constructor([[1, 2, 'c'], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 'c']], featureNames=['a', 'b', 'c'])
        ret = toTest.features.extract(match.allNonNumeric)
        expTest = self.constructor([[1, 2], ['a', 11], [7, 11], [7, 8]])
        expRet = self.constructor([['c'], ['c'], ['c'], ['c']])
        expTest.features.setNames(['a', 'b'])
        expRet.features.setNames(['c'])
        assert toTest == expTest
        assert ret == expRet

    def test_features_extract_match_list(self):
        toTest = self.constructor([[1, 2, 3], ['a', 11, 'c'], ['x', 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.features.extract(match.anyValues(['a', 'c', 'x']))
        expTest = self.constructor([[2], [11], [11], [8]])
        expRet = self.constructor([[1, 3], ['a', 'c'], ['x', 'c'], [7, 9]])
        expTest.features.setNames(['b'])
        expRet.features.setNames(['a', 'c'])
        assert toTest == expTest
        assert ret == expRet

        toTest = self.constructor([[1, 2, 'c'], ['a', 11, 'c'], ['x', 11, 'c'], [7, 8, 'c']], featureNames=['a', 'b', 'c'])
        ret = toTest.features.extract(match.allValues(['a', 'c', 'x']))
        expTest = self.constructor([[1, 2], ['a', 11], ['x', 11], [7, 8]])
        expRet = self.constructor([['c'], ['c'], ['c'], ['c']])
        expTest.features.setNames(['a', 'b'])
        expRet.features.setNames(['c'])
        assert toTest == expTest
        assert ret == expRet

    def test_features_extract_match_function(self):
        toTest = self.constructor([[1, 2, 3], [-1, 11, -3], [-1, 11, -1], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.features.extract(match.anyValues(lambda x: x < 0))
        expTest = self.constructor([[2], [11], [11], [8]])
        expRet = self.constructor([[1, 3], [-1, -3], [-1, -1], [7, 9]])
        expTest.features.setNames(['b'])
        expRet.features.setNames(['a', 'c'])
        assert toTest == expTest
        assert ret == expRet

        toTest = self.constructor([[1, 2, -3], [-1, 11, -3], [-1, 11, -3], [7, 8, -3]], featureNames=['a', 'b', 'c'])
        ret = toTest.features.extract(match.allValues(lambda x: x < 0))
        expTest = self.constructor([[1, 2], [-1, 11], [-1, 11], [7, 8]])
        expRet = self.constructor([[-3], [-3], [-3], [-3]])
        expTest.features.setNames(['a', 'b'])
        expRet.features.setNames(['c'])
        assert toTest == expTest
        assert ret == expRet

    #################
    # points.delete #
    #################

    @raises(CalledFunctionException)
    @mock.patch('nimble.data.axis.constructIndicesList', calledException)
    def test_points_delete_calls_constructIndicesList(self):
        toTest = self.constructor([[1,2,],[3,4]], pointNames=['a', 'b'])

        toTest.points.delete(['a', 'b'])

    @oneLogEntryExpected
    def test_points_delete_handmadeSingle(self):
        """ Test points.delete() against handmade output when deleting one point """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.points.delete(0)
        expEnd = self.constructor([[4, 5, 6], [7, 8, 9]])
        assert toTest.isIdentical(expEnd)

        # Check that names have not been generated unnecessarily
        assertNoNamesGenerated(toTest)

    def test_points_delete_index_NamePath_Preserve(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = 'testName'
        toTest._absPath = 'testAbsPath'
        toTest._relPath = 'testRelPath'

        toTest.points.delete(0)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_points_delete_ListIntoPEmpty(self):
        """ Test points.delete() by deleting a list of all points """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        toTest.points.delete([0, 1, 2, 3])

        data = [[], [], []]
        data = numpy.array(data).T
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    @twoLogEntriesExpected
    def test_points_delete_handmadeListSequence(self):
        """ Test points.delete() against handmade output for several list deletions """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        names = ['1', '4', '7', '10']
        toTest = self.constructor(data, pointNames=names)
        toTest.points.delete('1')
        exp1 = self.constructor([[4, 5, 6], [7, 8, 9], [10, 11, 12]], pointNames=['4', '7', '10'])
        assert toTest.isIdentical(exp1)
        toTest.points.delete([1, 2])
        exp2 = self.constructor([[4, 5, 6]], pointNames=['4'])
        assert toTest.isIdentical(exp2)

    def test_points_delete_handmadeListOrdering(self):
        """ Test points.delete() against handmade output for out of order deletion """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
        names = ['1', '4', '7', '10', '13']
        toTest = self.constructor(data, pointNames=names)
        toTest.points.delete([3, 4, 1])
        expEnd = self.constructor([[1, 2, 3], [7, 8, 9]], pointNames=['1', '7'])
        assert toTest.isIdentical(expEnd)

    def test_points_delete_List_trickyOrdering(self):
        data = [[0], [2], [2], [2], [0], [0], [0], [0], [2], [0]]
        toDelete = [6, 5, 3, 9]

        toTest = self.constructor(data)

        toTest.points.delete(toDelete)

        expRaw = [[0], [2], [2], [0], [0], [2]]
        expRem = self.constructor(expRaw)

        assert toTest == expRem

    def test_points_delete_function_selectionGap(self):
        data = [[0], [2], [2], [2], [0], [0], [0], [0], [2], [0]]
        deleteIndices = [3, 5, 6, 9]
        pnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        def sel(point):
            if int(point.points.getName(0)) in deleteIndices:
                return True
            else:
                return False

        toTest = self.constructor(data, pointNames=pnames)

        toTest.points.delete(sel)

        expRaw = [[0], [2], [2], [0], [0], [2]]
        expNames = ['0', '1', '2', '4', '7', '8']
        expRem = self.constructor(expRaw, pointNames=expNames)

        assert toTest == expRem


    def test_points_delete_functionIntoPEmpty(self):
        """ Test points.delete() by removing all points using a function """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest.points.delete(allTrue)

        data = [[], [], []]
        data = numpy.array(data).T
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    def test_points_delete_function_returnPointEmpty(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        exp = self.constructor(data)

        toTest.points.delete(allFalse)

        assert toTest.isIdentical(exp)

    def test_points_delete_handmadeFunction(self):
        """ Test points.delete() against handmade output for function deletion """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest.points.delete(oneOrFour)
        expEnd = self.constructor([[7, 8, 9]])
        assert toTest.isIdentical(expEnd)

    def test_points_delete_func_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.points.delete(oneOrFour)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_points_delete_handmadeFuncionWithFeatureNames(self):
        """ Test points.delete() against handmade output for function deletion with featureNames"""
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)

        toTest.points.delete(oneOrFour)
        expEnd = self.constructor([[7, 8, 9]], featureNames=featureNames)
        assert toTest.isIdentical(expEnd)


    @raises(InvalidArgumentType)
    def test_points_delete_exceptionStartInvalidType(self):
        """ Test points.delete() for InvalidArgumentType when start is not a valid ID type """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.delete(start=1.1, end=2)

    @raises(IndexError)
    def test_points_delete_exceptionEndInvalid(self):
        """ Test points.delete() for IndexError when end is not a valid Point index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.delete(start=1, end=5)

    @raises(InvalidArgumentValueCombination)
    def test_points_delete_exceptionInversion(self):
        """ Test points.delete() for InvalidArgumentValueCombination when start comes after end """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.delete(start=2, end=0)

    @raises(InvalidArgumentValueCombination)
    def test_points_delete_exceptionInversionPointName(self):
        """ Test points.delete() for InvalidArgumentValueCombination when start comes after end as FeatureNames"""
        pointNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames)
        toTest.points.delete(start="two", end="one")

    @raises(InvalidArgumentValue)
    def test_points_delete_exceptionDuplicates(self):
        """ Test points.delete() for InvalidArgumentValueCombination when toDelete contains duplicates """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.points.delete([0, 1, 0])

    def test_points_delete_handmadeRange(self):
        """ Test points.delete() against handmade output for range deletion """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.points.delete(start=1, end=2)

        expectedTest = self.constructor([[1, 2, 3]])

        assert expectedTest.isIdentical(toTest)

    def test_points_delete_range_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.points.delete(start=1, end=2)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_points_delete_rangeIntoPEmpty(self):
        """ Test points.delete() removes all points using ranges """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete(start=0, end=2)

        data = [[], [], []]
        data = numpy.array(data).T
        exp = self.constructor(data, featureNames=featureNames)

        assert toTest.isIdentical(exp)


    def test_points_delete_handmadeRangeWithFeatureNames(self):
        """ Test points.delete() against handmade output for range deletion with featureNames """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete(start=1, end=2)

        expectedTest = self.constructor([[1, 2, 3]], pointNames=['1'], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

    def test_points_delete_handmadeRangeRand_FM(self):
        """ Test points.delete() for correct sizes when using randomized range deletion and featureNames """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.delete(start=0, end=2, number=2, randomize=True)

        assert len(toTest.points) == 1

    def test_points_delete_handmadeRangeDefaults(self):
        """ Test points.delete uses the correct defaults in the case of range based deletion """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete(end=1)

        expectedTest = self.constructor([[7, 8, 9]], pointNames=['7'], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete(start=1)

        expectedTest = self.constructor([[1, 2, 3]], pointNames=['1'], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

    def test_points_delete_handmade_calling_pointNames(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete(start='4', end='7')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

    def test_points_delete_handmadeString(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('one=1')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('one==1')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('one<2')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('one<=1')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('one>4')
        expectedTest = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=pointNames[:-1], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('one>=7')
        expectedTest = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=pointNames[:-1], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('one!=4')
        expectedTest = self.constructor([[4, 5, 6]], pointNames=[pointNames[1]], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('one<1')
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('one>0')
        expectedTest = self.constructor([], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

    def test_points_delete_handmadeStringWithOperatorWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('one = 1')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('one == 1')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('one < 2')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('one <= 1')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('one > 4')
        expectedTest = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=pointNames[:-1], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('one >= 7')
        expectedTest = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=pointNames[:-1], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('one != 4')
        expectedTest = self.constructor([[4, 5, 6]], pointNames=[pointNames[1]], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('one < 1')
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('one > 0')
        expectedTest = self.constructor([], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

    def test_points_delete_handmadeStringWithFeatureWhitespace(self):
        featureNames = ["feature one", "feature two", "feature three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('feature one=1')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName=value with operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('feature one = 1')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

    def test_points_delete_list_mixed(self):
        """ Test points.delete() list input with mixed names and indices """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        names = ['1', '4', '7', '10']
        toTest = self.constructor(data, pointNames=names)
        toTest.points.delete(['1',1,-1])
        exp1 = self.constructor([[7, 8, 9]], pointNames=['7'])
        assert toTest.isIdentical(exp1)

    @raises(InvalidArgumentValue)
    def test_points_delete_handmadeString_featureNotExist(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.delete('four=1')

    def test_points_delete_numberOnly(self):
        self.back_delete_numberOnly('point')

    def test_points_delete_functionAndNumber(self):
        self.back_delete_functionAndNumber('point')

    def test_points_delete_numberAndRandomizeAllData(self):
        self.back_delete_numberAndRandomizeAllData('point')

    def test_points_delete_numberAndRandomizeSelectedData(self):
        self.back_delete_numberAndRandomizeSelectedData('point')

    @raises(InvalidArgumentValueCombination)
    def test_points_delete_randomizeNoNumber(self):
        self.back_structural_randomizeNoNumber('delete', 'point')

    @raises(InvalidArgumentValue)
    def test_points_delete_list_numberGreaterThanTargeted(self):
        self.back_structural_list_numberGreaterThanTargeted('delete', 'point')

    @raises(InvalidArgumentValue)
    def test_points_delete_function_numberGreaterThanTargeted(self):
        self.back_structural_function_numberGreaterThanTargeted('delete', 'point')

    @raises(InvalidArgumentValue)
    def test_points_delete_range_numberGreaterThanTargeted(self):
        self.back_structural_range_numberGreaterThanTargeted('delete', 'point')

    ### using match module ###

    def test_points_delete_match_missing(self):
        toTest = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        toTest.points.delete(match.anyMissing)
        exp = self.constructor([[1, 2, 3], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert toTest == exp

        toTest = self.constructor([[None, None, None], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        toTest.points.delete(match.allMissing)
        exp = self.constructor([[None, 11, None], [7, 11, None], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert toTest == exp

    def test_points_delete_match_nonNumeric(self):
        toTest = self.constructor([[1, 2, 3], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        toTest.points.delete(match.anyNonNumeric)
        exp = self.constructor([[1, 2, 3], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert toTest == exp

        toTest = self.constructor([['a', 'x', 'c'], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        toTest.points.delete(match.allNonNumeric)
        exp = self.constructor([['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert toTest == exp

    def test_points_delete_match_list(self):
        toTest = self.constructor([[1, 2, 3], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        toTest.points.delete(match.anyValues(['a', 'c', 'x']))
        exp = self.constructor([[1, 2, 3], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert toTest == exp

        toTest = self.constructor([['a', 'x', 'c'], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        toTest.points.delete(match.allValues(['a', 'c', 'x']))
        exp = self.constructor([['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert toTest == exp

    def test_points_delete_match_function(self):
        toTest = self.constructor([[1, 2, 3], [-1, 11, -3], [7, 11, -3], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        toTest.points.delete(match.anyValues(lambda x: x < 0))
        exp = self.constructor([[1, 2, 3], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert toTest == exp

        toTest = self.constructor([[-1, -2, -3], [-1, 11, -3], [7, 11, -3], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        toTest.points.delete(match.allValues(lambda x: x < 0))
        exp = self.constructor([[-1, 11, -3], [7, 11, -3], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert toTest == exp

    #########################
    # delete common backend #
    #########################

    def back_delete_numberOnly(self, axis):
        if axis == 'point':
            toCall = "points"
        else:
            toCall = "features"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        getattr(toTest, toCall).delete(number=3)
        if axis == 'point':
            rem = self.constructor(data[3:], pointNames=pnames[3:], featureNames=fnames)
        else:
            rem = self.constructor([p[3:] for p in data], pointNames=pnames, featureNames=fnames[3:])

        assert rem.isIdentical(toTest)

    def back_delete_functionAndNumber(self, axis):
        if axis == 'point':
            toCall = "points"
        else:
            toCall = "features"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        getattr(toTest, toCall).delete(allTrue, number=2)
        if axis == 'point':
            rem = self.constructor(data[2:], pointNames=pnames[2:], featureNames=fnames)
        else:
            rem = self.constructor([p[2:] for p in data], pointNames=pnames, featureNames=fnames[2:])

        assert rem.isIdentical(toTest)

    def back_delete_numberAndRandomizeAllData(self, axis):
        """test that randomizing (with same randomly chosen seed) and limiting to a
        given number provides the same result for all input types if using all the data
        """
        if axis == 'point':
            toCall = "points"
        else:
            toCall = "features"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest1 = self.constructor(data, pointNames=pnames, featureNames=fnames)
        toTest2 = toTest1.copy()
        toTest3 = toTest1.copy()
        toTest4 = toTest1.copy()

        seed = nimble.randomness.generateSubsidiarySeed()
        nimble.randomness.startAlternateControl(seed)
        getattr(toTest1, toCall).delete(number=3, randomize=True)
        nimble.randomness.endAlternateControl()

        nimble.randomness.startAlternateControl(seed)
        getattr(toTest2, toCall).delete([0, 1, 2, 3], number=3, randomize=True)
        nimble.randomness.endAlternateControl()

        nimble.randomness.startAlternateControl(seed)
        getattr(toTest3, toCall).delete(start=0, end=3, number=3, randomize=True)
        nimble.randomness.endAlternateControl()

        nimble.randomness.startAlternateControl(seed)
        getattr(toTest4, toCall).delete(allTrue, number=3, randomize=True)
        nimble.randomness.endAlternateControl()

        if axis == 'point':
            assert len(toTest1.points) == 1
        else:
            assert len(toTest1.features) == 1

        assert toTest1.isIdentical(toTest2)
        assert toTest1.isIdentical(toTest3)
        assert toTest1.isIdentical(toTest4)

    def back_delete_numberAndRandomizeSelectedData(self, axis):
        """test that randomization occurs after the data has been selected from the user inputs """
        if axis == 'point':
            toCall = "points"
        else:
            toCall = "features"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest1 = self.constructor(data, pointNames=pnames, featureNames=fnames)
        toTest2 = toTest1.copy()
        toTest3 = toTest1.copy()
        if axis == 'point':
            exp1 = toTest1[[0, 1, 3], :]
            exp2 = toTest1[[0, 2, 3], :]
        else:
            exp1 = toTest1[:, [0, 1, 3]]
            exp2 = toTest1[:, [0, 2, 3]]

        seed = nimble.randomness.generateSubsidiarySeed()
        nimble.randomness.startAlternateControl(seed)
        getattr(toTest1, toCall).delete([1, 2], number=1, randomize=True)
        nimble.randomness.endAlternateControl()

        nimble.randomness.startAlternateControl(seed)
        getattr(toTest2, toCall).delete(start=1, end=2, number=1, randomize=True)
        nimble.randomness.endAlternateControl()

        def middleRowsOrCols(value):
            return value[0] in [2, 4, 5, 7]

        nimble.randomness.startAlternateControl(seed)
        getattr(toTest3, toCall).delete(middleRowsOrCols, number=1, randomize=True)
        nimble.randomness.endAlternateControl()

        assert toTest1.isIdentical(exp1) or toTest1.isIdentical(exp2)
        assert toTest2.isIdentical(exp1) or toTest2.isIdentical(exp2)
        assert toTest3.isIdentical(exp1) or toTest3.isIdentical(exp2)

    ###################
    # features.delete #
    ###################

    @raises(CalledFunctionException)
    @mock.patch('nimble.data.axis.constructIndicesList', calledException)
    def test_features_delete_calls_constructIndicesList(self):
        toTest = self.constructor([[1,2,],[3,4]], featureNames=['a', 'b'])

        toTest.features.delete(['a', 'b'])

    @oneLogEntryExpected
    def test_features_delete_handmadeSingle(self):
        """ Test features.delete() against handmade output when deleting one feature """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.features.delete(0)

        expEnd = self.constructor([[2, 3], [5, 6], [8, 9]])
        assert toTest.isIdentical(expEnd)

        # Check that names have not been generated unnecessarily
        assertNoNamesGenerated(toTest)

    def test_features_delete_List_NamePath_Preserve(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.features.delete(0)

        assert toTest.path == 'testAbsPath'
        assert toTest.absolutePath == 'testAbsPath'
        assert toTest.relativePath == 'testRelPath'


    def test_features_delete_ListIntoFEmpty(self):
        """ Test features.delete() by removing a list of all features """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        toTest.features.delete([0, 1, 2])

        data = [[], [], [], []]
        data = numpy.array(data)
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    def test_features_delete_ListIntoFEmptyOutOfOrder(self):
        """ Test features.delete() by removing a list of all features """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        toTest.features.delete([2, 0, 1])

        data = [[], [], [], []]
        data = numpy.array(data)
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    @twoLogEntriesExpected
    def test_features_delete_handmadeListSequence(self):
        """ Test features.delete() against handmade output for several deletions by list """
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data, pointNames=pointNames)
        toTest.features.delete([0])
        exp1 = self.constructor([[2, 3, -1], [5, 6, -2], [8, 9, -3]], pointNames=pointNames)
        assert toTest.isIdentical(exp1)
        toTest.features.delete([2, 1])
        expEndData = [[2], [5], [8]]
        exp2 = self.constructor(expEndData, pointNames=pointNames)
        assert toTest.isIdentical(exp2)

    def test_features_delete_handmadeListWithFeatureName(self):
        """ Test features.delete() against handmade output for list deletion when specifying featureNames """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.delete(["one"])
        exp1 = self.constructor([[2, 3, -1], [5, 6, -2], [8, 9, -3]], featureNames=["two", "three", "neg"])
        assert toTest.isIdentical(exp1)
        toTest.features.delete(["three", "neg"])
        exp2 = self.constructor([[2], [5], [8]], featureNames=["two"])
        assert toTest.isIdentical(exp2)


    def test_features_delete_List_trickyOrdering(self):
        data = [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
        toDelete = [6, 5, 3, 9]

        toTest = self.constructor(data)
        toTest.features.delete(toDelete)

        expRaw = [0, 1, 1, 0, 0, 1]
        expRem = self.constructor(expRaw)

        assert toTest == expRem

    def test_features_delete_List_reorderingWithFeatureNames(self):
        data = [[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12]]
        fnames = ['a', 'b', 'c', 'd']
        toTest = self.constructor(data, featureNames=fnames)

        toDelete = ['a', 'c', 'b']
        toTest.features.delete(toDelete)
        expTestRaw = [[10], [11], [12]]
        expTestNames = ['d']
        expTest = self.constructor(expTestRaw, featureNames=expTestNames)

        assert toTest == expTest


    def test_features_delete_function_selectionGap(self):
        data = [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
        fnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        deleteIndices = [3, 5, 6, 9]

        def sel(feature):
            if int(feature.features.getName(0)) in deleteIndices:
                return True
            else:
                return False

        toTest = self.constructor(data, featureNames=fnames)
        toTest.features.delete(sel)

        expRaw = [0, 1, 1, 0, 0, 1]
        expNames = ['0', '1', '2', '4', '7', '8']
        expRem = self.constructor(expRaw, featureNames=expNames)

        assert toTest == expRem


    def test_features_delete_functionIntoFEmpty(self):
        """ Test features.delete() by removing all featuress using a function """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest.features.delete(allTrue)

        data = [[], [], []]
        data = numpy.array(data)
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    def test_features_delete_function_returnPointEmpty(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        exp = self.constructor(data)

        toTest.features.delete(allFalse)

        assert toTest.isIdentical(exp)


    def test_features_delete_handmadeFunction(self):
        """ Test features.delete() against handmade output for function deletion """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data)

        toTest.features.delete(absoluteOne)

        expEnd = self.constructor([[2, 3], [5, 6], [8, 9]])
        assert toTest.isIdentical(expEnd)


    def test_features_delete_func_NamePath_preservation(self):
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.features.delete(absoluteOne)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_features_delete_handmadeFunctionWithFeatureName(self):
        """ Test features.delete() against handmade output for function deletion with featureNames """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        pointNames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        ext = toTest.features.delete(absoluteOne)
        expEnd = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=["two", "three"])
        assert toTest.isIdentical(expEnd)

    @raises(InvalidArgumentType)
    def test_features_delete_exceptionStartInvalidType(self):
        """ Test features.delete() for InvalidArgumentType when start is not a valid ID type """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.delete(start=1.1, end=2)

    @raises(KeyError)
    def test_features_delete_exceptionStartInvalidFeatureName(self):
        """ Test features.delete() for KeyError when start is not a valid feature FeatureName """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.delete(start="wrong", end=2)

    @raises(IndexError)
    def test_features_delete_exceptionEndInvalid(self):
        """ Test features.delete() for IndexError when end is not a valid feature index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.delete(start=0, end=5)

    @raises(KeyError)
    def test_features_delete_exceptionEndInvalidFeatureName(self):
        """ Test features.delete() for KeyError when end is not a valid featureName """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.delete(start="two", end="five")

    @raises(InvalidArgumentValueCombination)
    def test_features_delete_exceptionInversion(self):
        """ Test features.delete() for InvalidArgumentValueCombination when start comes after end """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.delete(start=2, end=0)

    @raises(InvalidArgumentValueCombination)
    def test_features_delete_exceptionInversionFeatureName(self):
        """ Test features.delete() for InvalidArgumentValueCombination when start comes after end as FeatureNames"""
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.delete(start="two", end="one")

    @raises(InvalidArgumentValue)
    def test_features_delete_exceptionDuplicates(self):
        """ Test points.delete() for InvalidArgumentValueCombination when toDelete contains duplicates """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.features.delete([0, 1, 0])

    def test_features_delete_rangeIntoFEmpty(self):
        """ Test features.delete() removes all Featuress using ranges """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.delete(start=0, end=2)

        data = [[], [], []]
        data = numpy.array(data)
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    def test_features_delete_handmadeRange(self):
        """ Test features.delete() against handmade output for range deletion """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.features.delete(start=1, end=2)

        expectedTest = self.constructor([[1], [4], [7]])

        assert expectedTest.isIdentical(toTest)

    def test_features_delete_range_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.features.delete(start=1, end=2)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_features_delete_handmadeWithFeatureNames(self):
        """ Test features.delete() against handmade output for range deletion with FeatureNames """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete(start=1, end=2)

        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=["one"])

        assert expectedTest.isIdentical(toTest)

    def test_features_delete_handmade_calling_featureNames(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete(start="two", end="three")

        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=["one"])

        assert expectedTest.isIdentical(toTest)

    def test_features_delete_handmadeString(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['p1', 'p2', 'p3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('p2=5')
        expectedTest = self.constructor([[1, 3], [4, 6], [7, 9]], pointNames=pointNames,
                                        featureNames=[featureNames[0], featureNames[-1]])
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('p1==1')
        expectedTest = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('p3<9')
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('p3<=8')
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('p3>8')
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('p3>8.5')
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('p1!=1.0')
        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('p1<1')
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('p1>0')
        expectedTest = self.constructor([[], [], []], pointNames=pointNames)
        assert expectedTest.isIdentical(toTest)

    def test_features_delete_handmadeStringWithOperatorWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['p1', 'p2', 'p3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('p2 = 5')
        expectedTest = self.constructor([[1, 3], [4, 6], [7, 9]], pointNames=pointNames,
                                        featureNames=[featureNames[0], featureNames[-1]])
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('p1 == 1')
        expectedTest = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('p3 < 9')
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('p3 <= 8')
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('p3 > 8')
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('p3 > 8.5')
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('p1 != 1.0')
        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('p1 < 1')
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('p1 > 0')
        expectedTest = self.constructor([[],[],[]], pointNames=pointNames)
        assert expectedTest.isIdentical(toTest)

    def test_features_delete_handmadeStringWithPointWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['pt 1', 'pt 2', 'pt 3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value with no operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('pt 2=5')
        expectedTest = self.constructor([[1, 3], [4, 6], [7, 9]], pointNames=pointNames,
                                        featureNames=[featureNames[0], featureNames[-1]])
        assert expectedTest.isIdentical(toTest)

        #test pointName=value with operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('pt 2 = 5')
        expectedTest = self.constructor([[1, 3], [4, 6], [7, 9]], pointNames=pointNames,
                                        featureNames=[featureNames[0], featureNames[-1]])
        assert expectedTest.isIdentical(toTest)

    def test_features_delete_list_mixed(self):
        """ Test features.delete() list input with mixed names and indices """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.delete([1, "three", -1])
        exp1 = self.constructor([[1], [4], [7]], featureNames=["one"])
        assert toTest.isIdentical(exp1)

    @raises(InvalidArgumentValue)
    def test_features_delete_handmadeString_pointNotExist(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.delete('5=1')

    def test_features_delete_numberOnly(self):
        self.back_delete_numberOnly('feature')

    def test_features_delete_functionAndNumber(self):
        self.back_delete_functionAndNumber('feature')

    def test_features_delete_numberAndRandomizeAllData(self):
        self.back_delete_numberAndRandomizeAllData('feature')

    def test_features_delete_numberAndRandomizeSelectedData(self):
        self.back_delete_numberAndRandomizeSelectedData('feature')

    @raises(InvalidArgumentValueCombination)
    def test_features_delete_randomizeNoNumber(self):
        self.back_structural_randomizeNoNumber('delete', 'feature')

    @raises(InvalidArgumentValue)
    def test_features_delete_list_numberGreaterThanTargeted(self):
        self.back_structural_list_numberGreaterThanTargeted('delete', 'feature')

    @raises(InvalidArgumentValue)
    def test_features_delete_function_numberGreaterThanTargeted(self):
        self.back_structural_function_numberGreaterThanTargeted('delete', 'feature')

    @raises(InvalidArgumentValue)
    def test_features_delete_range_numberGreaterThanTargeted(self):
        self.back_structural_range_numberGreaterThanTargeted('delete', 'feature')

    ### using match module ###

    def test_features_delete_match_missing(self):
        toTest = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        toTest.features.delete(match.anyMissing)
        exp = self.constructor([[2], [11], [11], [8]])
        exp.features.setNames(['b'])
        assert toTest == exp

        toTest = self.constructor([[1, 2, None], [None, 11, None], [7, 11, None], [7, 8, None]], featureNames=['a', 'b', 'c'])
        toTest.features.delete(match.allMissing)
        exp = self.constructor([[1, 2], [None, 11], [7, 11], [7, 8]])
        exp.features.setNames(['a', 'b'])
        assert toTest == exp

    def test_features_delete_match_nonNumeric(self):
        toTest = self.constructor([[1, 2, 3], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        toTest.features.delete(match.anyNonNumeric)
        exp = self.constructor([[2], [11], [11], [8]])
        exp.features.setNames(['b'])
        assert toTest == exp

        toTest = self.constructor([[1, 2, 'c'], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 'c']], featureNames=['a', 'b', 'c'])
        toTest.features.delete(match.allNonNumeric)
        exp = self.constructor([[1, 2], ['a', 11], [7, 11], [7, 8]])
        exp.features.setNames(['a', 'b'])
        assert toTest == exp

    def test_features_delete_match_list(self):
        toTest = self.constructor([[1, 2, 3], ['a', 11, 'c'], ['x', 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        toTest.features.delete(match.anyValues(['a', 'c', 'x']))
        exp = self.constructor([[2], [11], [11], [8]])
        exp.features.setNames(['b'])
        assert toTest == exp

        toTest = self.constructor([[1, 2, 'c'], ['a', 11, 'c'], ['x', 11, 'c'], [7, 8, 'c']], featureNames=['a', 'b', 'c'])
        toTest.features.delete(match.allValues(['a', 'c', 'x']))
        exp = self.constructor([[1, 2], ['a', 11], ['x', 11], [7, 8]])
        exp.features.setNames(['a', 'b'])
        assert toTest == exp

    def test_features_delete_match_function(self):
        toTest = self.constructor([[1, 2, 3], [-1, 11, -3], [-1, 11, -1], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        toTest.features.delete(match.anyValues(lambda x: x < 0))
        exp = self.constructor([[2], [11], [11], [8]])
        exp.features.setNames(['b'])
        assert toTest == exp

        toTest = self.constructor([[1, 2, -3], [-1, 11, -3], [-1, 11, -3], [7, 8, -3]], featureNames=['a', 'b', 'c'])
        toTest.features.delete(match.allValues(lambda x: x < 0))
        exp = self.constructor([[1, 2], [-1, 11], [-1, 11], [7, 8]])
        exp.features.setNames(['a', 'b'])
        assert toTest == exp

    #################
    # points.retain #
    #################

    @raises(CalledFunctionException)
    @mock.patch('nimble.data.axis.constructIndicesList', calledException)
    def test_points_retain_calls_constructIndicesList(self):
        """ Test points.retain calls constructIndicesList before calling _genericStructuralFrontend"""
        toTest = self.constructor([[1,2,],[3,4]], pointNames=['a', 'b'])
        toTest.points.retain(['a', 'b'])

    @oneLogEntryExpected
    def test_points_retain_handmadeSingle(self):
        """ Test points.retain() against handmade output when retaining one point """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.points.retain(0)
        exp1 = self.constructor([[1, 2, 3]])
        assert toTest.isIdentical(exp1)

        # Check that names have not been generated unnecessarily
        assertNoNamesGenerated(toTest)

    def test_points_retain_index_NamePath_Preserve(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = 'testName'
        toTest._absPath = 'testAbsPath'
        toTest._relPath = 'testRelPath'

        toTest.points.retain(0)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_points_retain_list_retain_all(self):
        """ Test points.retain() by retaining a list of all points """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        exp = self.constructor(data)
        toTest.points.retain([0, 1, 2, 3])

        assert toTest.isIdentical(exp)

    def test_points_retain_list_retain_nothing(self):
        """ Test points.retain() by retaining an empty list """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        toTest.points.retain([])

        expData = [[], [], []]
        expData = numpy.array(expData).T
        expTest = self.constructor(expData)
        assert toTest.isIdentical(expTest)

    def test_points_retain_pythonRange(self):
        """ Test points.retain() by retaining a python range of points """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        exp = self.constructor([[4, 5, 6], [7, 8, 9]])
        toTest.points.retain(range(1,3))

        assert toTest.isIdentical(exp)

    @twoLogEntriesExpected
    def test_points_retain_handmadeListSequence(self):
        """ Test points.retain() against handmade output for several list retentions """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        names = ['1', '4', '7', '10']
        toTest = self.constructor(data, pointNames=names)
        toTest.points.retain(['1','4','10'])
        exp1 = self.constructor([[1, 2, 3], [4, 5, 6], [10, 11, 12]], pointNames=['1','4','10'])
        assert toTest.isIdentical(exp1)
        toTest.points.retain(1)
        exp2 = self.constructor([4, 5, 6], pointNames=['4'])
        assert toTest.isIdentical(exp2)


    def test_points_retain_list_mixed(self):
        """ Test points.retain() list input with mixed names and indices """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        names = ['1', '4', '7', '10']
        toTest = self.constructor(data, pointNames=names)
        toTest.points.retain(['1',1,-1])
        exp1 = self.constructor([[1, 2, 3], [4, 5, 6], [10, 11, 12]], pointNames=['1','4','10'])
        assert toTest.isIdentical(exp1)


    def test_points_retain_handmadeListOrdering(self):
        """ Test points.retain() against handmade output for out of order retention """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
        names = ['1', '4', '7', '10', '13']
        toTest = self.constructor(data, pointNames=names)
        toTest.points.retain([3, 4, 1])
        exp1 = self.constructor([[10, 11, 12], [13, 14, 15], [4, 5, 6]], pointNames=['10', '13', '4'])
        assert toTest.isIdentical(exp1)


    def test_points_retain_List_trickyOrdering(self):
        data = [[0], [2], [2], [2], [0], [0], [0], [0], [2], [0]]
        toRetain = [6, 5, 3, 9]

        toTest = self.constructor(data)

        toTest.points.retain(toRetain)

        expRaw = [[0], [0], [2], [0]]
        expTest = self.constructor(expRaw)

        assert toTest == expTest

    def test_points_retain_function_selectionGap(self):
        data = [[0], [2], [2], [2], [0], [0], [0], [0], [2], [0]]
        retainIndices = [3, 5, 6, 9]
        pnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        def sel(point):
            if int(point.points.getName(0)) in retainIndices:
                return True
            else:
                return False

        toTest = self.constructor(data, pointNames=pnames)

        toTest.points.retain(sel)

        expRaw = [[2], [0], [0], [0]]
        expNames = ['3', '5', '6', '9']
        expTest = self.constructor(expRaw, pointNames=expNames)

        assert toTest == expTest


    def test_points_retain_functionIntoPEmpty(self):
        """ Test points.retain() by retaining all points using a function """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        expTest = self.constructor(data)

        toTest.points.retain(allTrue)
        assert toTest.isIdentical(expTest)


    def test_points_retain_function_returnPointEmpty(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest.points.retain(allFalse)

        expData = numpy.array([[], [], []])
        expData = expData.T
        expTest = self.constructor(expData)

        assert toTest.isIdentical(expTest)

    def test_points_retain_function_NumberAndRandomize(self):
        data = [[1], [2], [3], [4], [5], [6], [7], [8]]
        toTest = self.constructor(data)

        toTest.points.retain(evenOnly, number=3, randomize=True)
        assert len(toTest.points) == 3

    def test_points_retain_handmadeFunction(self):
        """ Test points.retain() against handmade output for function retention """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest.points.retain(oneOrFour)
        exp = self.constructor([[1, 2, 3], [4, 5, 6]])
        assert toTest.isIdentical(exp)


    def test_points_retain_func_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.points.retain(oneOrFour)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_points_retain_handmadeFunctionWithFeatureNames(self):
        """ Test points.retain() against handmade output for function retention with featureNames"""
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)

        toTest.points.retain(oneOrFour)
        exp = self.constructor([[1, 2, 3], [4, 5, 6]], featureNames=featureNames)
        assert toTest.isIdentical(exp)


    @raises(InvalidArgumentType)
    def test_points_retain_exceptionStartInvalidType(self):
        """ Test points.retain() for InvalidArgumentType when start is not a valid ID type """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.retain(start=1.1, end=2)

    @raises(IndexError)
    def test_points_retain_exceptionEndInvalid(self):
        """ Test points.retain() for IndexError when end is not a valid Point index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.retain(start=1, end=5)

    @raises(InvalidArgumentValueCombination)
    def test_points_retain_exceptionInversion(self):
        """ Test points.retain() for InvalidArgumentValueCombination when start comes after end """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.retain(start=2, end=0)

    @raises(InvalidArgumentValueCombination)
    def test_points_retain_exceptionInversionPointName(self):
        """ Test points.retain() for InvalidArgumentValueCombination when start comes after end as FeatureNames"""
        pointNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames)
        toTest.points.retain(start="two", end="one")

    @raises(InvalidArgumentValue)
    def test_points_retain_exceptionDuplicates(self):
        """ Test points.retain() for InvalidArgumentValueCombination when toRetain contains duplicates """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.points.retain([0, 1, 0])

    def test_points_retain_handmadeRange(self):
        """ Test points.retain() against handmade output for range retention """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.points.retain(start=1, end=2)

        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]])

        assert expectedTest.isIdentical(toTest)

    def test_points_retain_range_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.points.retain(start=1, end=2)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_points_retain_rangeIntoPEmpty(self):
        """ Test points.retain() retains all points using ranges """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain(start=0, end=2)

        assert toTest.isIdentical(expRet)


    def test_points_retain_handmadeRangeWithFeatureNames(self):
        """ Test points.retain() against handmade output for range retention with featureNames """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain(start=1, end=2)

        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=['4', '7'], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

    def test_points_retain_handmadeRangeRand_FM(self):
        """ Test points.retain() for correct sizes when using randomized range retention and featureNames """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.retain(start=0, end=2, number=2, randomize=True)
        assert len(toTest.points) == 2

    def test_points_retain_handmadeRangeDefaults(self):
        """ Test points.retain uses the correct defaults in the case of range based retention """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain(end=1)

        expectedTest = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=['1', '4'], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain(start=1)

        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=['4', '7'], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

    def test_points_retain_handmade_calling_pointNames(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain(start='4', end='7')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

    def test_points_retain_handmadeString(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('one=1')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('one==1')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('one<2')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('one<=1')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('one>4')
        expectedTest = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('one>=7')
        expectedTest = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('one!=4')
        expectedTest = self.constructor([[1, 2, 3], [7, 8, 9]], pointNames=[pointNames[0], pointNames[-1]],
                                       featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('one<1')
        expectedTest = self.constructor([], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('one>0')
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

    def test_points_retain_handmadeStringWithOperatorWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('one = 1')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('one == 1')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('one < 2')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('one <= 1')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('one > 4')
        expectedTest = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('one >= 7')
        expectedTest = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('one != 4')
        expectedTest = self.constructor([[1, 2, 3], [7, 8, 9]], pointNames=[pointNames[0], pointNames[-1]],
                                       featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('one < 1')
        expectedTest = self.constructor([], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('one > 0')
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

    def test_points_retain_handmadeStringWithFeatureWhitespace(self):
        featureNames = ["feature one", "feature two", "feature three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('feature one=1')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName=value with operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('feature one = 1')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

    @raises(InvalidArgumentValue)
    def test_points_retain_handmadeString_featureNotExist(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.points.retain('four=1')

    def test_points_retain_numberOnly(self):
        self.back_retain_numberOnly('point')

    def test_points_retain_functionAndNumber(self):
        self.back_retain_functionAndNumber('point')

    def test_points_retain_numberAndRandomizeAllData(self):
        self.back_retain_numberAndRandomizeAllData('point')

    def test_points_retain_numberAndRandomizeSelectedData(self):
        self.back_retain_numberAndRandomizeSelectedData('point')

    @raises(InvalidArgumentValueCombination)
    def test_points_retain_randomizeNoNumber(self):
        self.back_structural_randomizeNoNumber('retain', 'point')

    @raises(InvalidArgumentValue)
    def test_points_retain_list_numberGreaterThanTargeted(self):
        self.back_structural_list_numberGreaterThanTargeted('retain', 'point')

    @raises(InvalidArgumentValue)
    def test_points_retain_function_numberGreaterThanTargeted(self):
        self.back_structural_function_numberGreaterThanTargeted('retain', 'point')

    @raises(InvalidArgumentValue)
    def test_points_retain_range_numberGreaterThanTargeted(self):
        self.back_structural_range_numberGreaterThanTargeted('retain', 'point')

    ### using match module ###

    def test_points_retain_match_missing(self):
        toTest = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.points.retain(match.anyMissing)
        expTest = self.constructor([[None, 11, None], [7, 11, None]])
        expTest.features.setNames(['a', 'b', 'c'])
        assert toTest == expTest

        toTest = self.constructor([[None, None, None], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.points.retain(match.allMissing)
        expTest = self.constructor([[None, None, None]])
        expTest.features.setNames(['a', 'b', 'c'])
        assert toTest == expTest

    def test_points_retain_match_nonNumeric(self):
        toTest = self.constructor([[1, 2, 3], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.points.retain(match.anyNonNumeric)
        expTest = self.constructor([['a', 11, 'c'], [7, 11, 'c']])
        expTest.features.setNames(['a', 'b', 'c'])
        assert toTest == expTest

        toTest = self.constructor([['a', 'x', 'c'], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.points.retain(match.allNonNumeric)
        expTest = self.constructor([['a', 'x', 'c']])
        expTest.features.setNames(['a', 'b', 'c'])
        assert toTest == expTest

    def test_points_retain_match_list(self):
        toTest = self.constructor([[1, 2, 3], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.points.retain(match.anyValues(['a', 'c', 'x']))
        expTest = self.constructor([['a', 11, 'c'], [7, 11, 'c']])
        expTest.features.setNames(['a', 'b', 'c'])
        assert toTest == expTest

        toTest = self.constructor([['a', 'x', 'c'], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.points.retain(match.allValues(['a', 'c', 'x']))
        expTest = self.constructor([['a', 'x', 'c']])
        expTest.features.setNames(['a', 'b', 'c'])
        assert toTest == expTest

    def test_points_retain_match_function(self):
        toTest = self.constructor([[1, 2, 3], [-1, 11, -3], [7, 11, -3], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.points.retain(match.anyValues(lambda x: x < 0))
        expTest = self.constructor([[-1, 11, -3], [7, 11, -3]])
        expTest.features.setNames(['a', 'b', 'c'])
        assert toTest == expTest

        toTest = self.constructor([[-1, -2, -3], [-1, 11, -3], [7, 11, -3], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.points.retain(match.allValues(lambda x: x < 0))
        expTest = self.constructor([[-1, -2, -3]])
        expTest.features.setNames(['a', 'b', 'c'])
        assert toTest == expTest

    #########################
    # retain common backend #
    #########################

    def back_retain_numberOnly(self, axis):
        if axis == 'point':
            toCall = "points"
        else:
            toCall = "features"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        getattr(toTest, toCall).retain(number=3)
        if axis == 'point':
            exp = self.constructor(data[:3], pointNames=pnames[:3], featureNames=fnames)
        else:
            exp = self.constructor([p[:3] for p in data], pointNames=pnames, featureNames=fnames[:3])

        assert exp.isIdentical(toTest)

    def back_retain_functionAndNumber(self, axis):
        if axis == 'point':
            toCall = "points"
        else:
            toCall = "features"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        getattr(toTest, toCall).retain([0,1,2], number=2)
        if axis == 'point':
            exp = self.constructor(data[:2], pointNames=pnames[:2], featureNames=fnames)
        else:
            exp = self.constructor([p[:2] for p in data], pointNames=pnames, featureNames=fnames[:2])

        assert toTest.isIdentical(exp)

    def back_retain_numberAndRandomizeAllData(self, axis):
        """test that randomizing (with same randomly chosen seed) and limiting to a
        given number provides the same result for all input types if using all the data
        """
        if axis == 'point':
            toCall = "points"
        else:
            toCall = "features"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest1 = self.constructor(data, pointNames=pnames, featureNames=fnames)
        toTest2 = toTest1.copy()
        toTest3 = toTest1.copy()
        toTest4 = toTest1.copy()

        seed = nimble.randomness.generateSubsidiarySeed()
        nimble.randomness.startAlternateControl(seed)
        getattr(toTest1, toCall).retain(number=3, randomize=True)
        nimble.randomness.endAlternateControl()

        nimble.randomness.startAlternateControl(seed)
        getattr(toTest2, toCall).retain([0, 1, 2, 3], number=3, randomize=True)
        nimble.randomness.endAlternateControl()

        nimble.randomness.startAlternateControl(seed)
        getattr(toTest3, toCall).retain(start=0, end=3, number=3, randomize=True)
        nimble.randomness.endAlternateControl()

        nimble.randomness.startAlternateControl(seed)
        getattr(toTest4, toCall).retain(allTrue, number=3, randomize=True)
        nimble.randomness.endAlternateControl()

        if axis == 'point':
            assert len(toTest1.points) == 3
        else:
            assert len(toTest1.features) == 3

        assert toTest1.isIdentical(toTest2)
        assert toTest1.isIdentical(toTest3)
        assert toTest1.isIdentical(toTest4)

    def back_retain_numberAndRandomizeSelectedData(self, axis):
        """test that randomization occurs after the data has been selected from the user inputs """
        if axis == 'point':
            toCall = "points"
        else:
            toCall = "features"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest1 = self.constructor(data, pointNames=pnames, featureNames=fnames)
        toTest2 = toTest1.copy()
        toTest3 = toTest1.copy()
        if axis == 'point':
            exp1 = toTest1[1, :]
            exp2 = toTest1[2, :]
        else:
            exp1 = toTest1[:, 1]
            exp2 = toTest1[:, 2]

        seed = nimble.randomness.generateSubsidiarySeed()
        nimble.randomness.startAlternateControl(seed)
        getattr(toTest1, toCall).retain([1, 2], number=1, randomize=True)
        nimble.randomness.endAlternateControl()

        nimble.randomness.startAlternateControl(seed)
        getattr(toTest2, toCall).retain(start=1, end=2, number=1, randomize=True)
        nimble.randomness.endAlternateControl()

        def middleRowsOrCols(value):
            return value[0] in [2, 4, 5, 7]

        nimble.randomness.startAlternateControl(seed)
        getattr(toTest3, toCall).retain(middleRowsOrCols, number=1, randomize=True)
        nimble.randomness.endAlternateControl()

        assert toTest1.isIdentical(exp1) or toTest1.isIdentical(exp2)
        assert toTest2.isIdentical(exp1) or toTest2.isIdentical(exp2)
        assert toTest3.isIdentical(exp1) or toTest3.isIdentical(exp2)

    ###################
    # features.retain #
    ###################

    @raises(CalledFunctionException)
    @mock.patch('nimble.data.axis.constructIndicesList', calledException)
    def test_features_retain_calls_constructIndicesList(self):
        toTest = self.constructor([[1,2,],[3,4]], featureNames=['a', 'b'])

        toTest.features.retain(['a', 'b'])

    @oneLogEntryExpected
    def test_features_retain_handmadeSingle(self):
        """ Test features.retain() against handmade output when retaining one feature """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.features.retain(0)
        exp1 = self.constructor([[1], [4], [7]])

        assert toTest.isIdentical(exp1)

        # Check that names have not been generated unnecessarily
        assertNoNamesGenerated(toTest)

    def test_features_retain_List_NamePath_Preserve(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.features.retain(0)

        assert toTest.path == 'testAbsPath'
        assert toTest.absolutePath == 'testAbsPath'
        assert toTest.relativePath == 'testRelPath'

    def test_features_retain_list_retain_all(self):
        """ Test features.retain() by retaining a list of all features """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        expTest = self.constructor(data)
        toTest.features.retain([0, 1, 2])

        assert toTest.isIdentical(expTest)

    def test_features_retain_list_retain_nothing(self):
        """ Test features.retain() by retaining an empty list """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        toTest.features.retain([])

        expData = [[], [], [], []]
        expData = numpy.array(expData)
        expTest = self.constructor(expData)
        assert toTest.isIdentical(expTest)

    def test_features_retain_pythonRange(self):
        """ Test features.retain() by retaining a python range of points """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        exp = self.constructor([[2, 3], [5, 6], [8, 9], [11, 12]])
        toTest.features.retain(range(1,3))

        assert toTest.isIdentical(exp)

    def test_features_retain_ListIntoFEmptyOutOfOrder(self):
        """ Test features.retain() by retaining a list of all features """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        expData = [[3, 1, 2], [6, 4, 5], [9, 7, 8], [12, 10, 11]]
        expTest = self.constructor(expData)
        toTest.features.retain([2, 0, 1])

        assert toTest.isIdentical(expTest)

    @twoLogEntriesExpected
    def test_features_retain_handmadeListSequence(self):
        """ Test features.retain() against handmade output for several retentions by list """
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data, pointNames=pointNames)
        toTest.features.retain([1, 2, 3])
        exp1 = self.constructor([[2, 3, -1], [5, 6, -2], [8, 9, -3]], pointNames=pointNames)
        assert toTest.isIdentical(exp1)
        toTest.features.retain([2, 1])
        exp2 = self.constructor([[-1, 3], [-2, 6], [-3, 9]], pointNames=pointNames)
        assert toTest.isIdentical(exp2)

    def test_features_retain_handmadeListWithFeatureName(self):
        """ Test features.retain() against handmade output for list retention when specifying featureNames """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.retain(["two", "three", "neg"])
        exp1 = self.constructor([[2, 3, -1], [5, 6, -2], [8, 9, -3]], featureNames=["two", "three", "neg"])
        assert toTest.isIdentical(exp1)
        toTest.features.retain(["three", "neg"])
        exp2 = self.constructor([[3, -1], [6, -2], [9, -3]], featureNames=["three", "neg"])
        assert toTest.isIdentical(exp2)


    def test_features_retain_list_mixed(self):
        """ Test features.retain() list input with mixed names and indices """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.retain([1, "three", -1])
        exp1 = self.constructor([[2, 3, -1], [5, 6, -2], [8, 9, -3]], featureNames=["two", "three", "neg"])
        assert toTest.isIdentical(exp1)


    def test_features_retain_List_trickyOrdering(self):
        data = [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
        toRetain = [6, 5, 3, 9]

        toTest = self.constructor(data)

        toTest.features.retain(toRetain)

        expRaw = [0, 0, 1, 0]
        expTest = self.constructor(expRaw)

        assert toTest == expTest

    def test_features_retain_List_reorderingWithFeatureNames(self):
        data = [[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12]]
        fnames = ['a', 'b', 'c', 'd']
        test = self.constructor(data, featureNames=fnames)

        expRetRaw = [[1, 3, 2], [4, 6, 5], [7, 9, 8]]
        expRetNames = ['a', 'c', 'b']
        exp = self.constructor(expRetRaw, featureNames=expRetNames)

        test.features.retain(expRetNames)
        assert test == exp


    def test_features_retain_function_selectionGap(self):
        data = [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
        fnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        retainIndices = [3, 5, 6, 9]

        def sel(feature):
            if int(feature.features.getName(0)) in retainIndices:
                return True
            else:
                return False

        toTest = self.constructor(data, featureNames=fnames)

        toTest.features.retain(sel)

        expRaw = [1, 0, 0, 0]
        expNames = ['3', '5', '6', '9']
        expTest = self.constructor(expRaw, featureNames=expNames)

        assert toTest == expTest


    def test_features_retain_functionIntoFEmpty(self):
        """ Test features.retain() by retaining all featuress using a function """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        expTest = self.constructor(data)

        toTest.features.retain(allTrue)
        assert toTest.isIdentical(expTest)

    def test_features_retain_function_returnPointEmpty(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest.features.retain(allFalse)

        data = [[], [], []]
        data = numpy.array(data)
        expTest = self.constructor(data)

        assert toTest.isIdentical(expTest)

    def test_features_retain_function_NumberAndRandomize(self):
        data = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        toTest = self.constructor(data)

        toTest.features.retain(evenOnly, number=2, randomize=True)
        assert len(toTest.features) == 2

    def test_features_retain_handmadeFunction(self):
        """ Test features.retain() against handmade output for function retention """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data)

        toTest.features.retain(absoluteOne)
        exp = self.constructor([[1, -1], [4, -2], [7, -3]])
        assert toTest.isIdentical(exp)

    def test_features_retain_func_NamePath_preservation(self):
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.features.retain(absoluteOne)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_features_retain_handmadeFunctionWithFeatureName(self):
        """ Test features.retain() against handmade output for function retention with featureNames """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        pointNames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        toTest.features.retain(absoluteOne)
        exp = self.constructor([[1, -1], [4, -2], [7, -3]], pointNames=pointNames, featureNames=['one', 'neg'])
        assert toTest.isIdentical(exp)

    @raises(InvalidArgumentType)
    def test_features_retain_exceptionStartInvalidType(self):
        """ Test features.retain() for InvalidArgumentType when start is not a valid ID type """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.retain(start=1.1, end=2)

    @raises(KeyError)
    def test_features_retain_exceptionStartInvalidFeatureName(self):
        """ Test features.retain() for KeyError when start is not a valid feature FeatureName """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.retain(start="wrong", end=2)

    @raises(IndexError)
    def test_features_retain_exceptionEndInvalid(self):
        """ Test features.retain() for IndexError when end is not a valid feature index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.retain(start=0, end=5)

    @raises(KeyError)
    def test_features_retain_exceptionEndInvalidFeatureName(self):
        """ Test features.retain() for KeyError when end is not a valid featureName """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.retain(start="two", end="five")

    @raises(InvalidArgumentValueCombination)
    def test_features_retain_exceptionInversion(self):
        """ Test features.retain() for InvalidArgumentValueCombination when start comes after end """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.retain(start=2, end=0)

    @raises(InvalidArgumentValueCombination)
    def test_features_retain_exceptionInversionFeatureName(self):
        """ Test features.retain() for InvalidArgumentValueCombination when start comes after end as FeatureNames"""
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.retain(start="two", end="one")

    @raises(InvalidArgumentValue)
    def test_features_retain_exceptionDuplicates(self):
        """ Test points.retain() for InvalidArgumentValueCombination when toRetain contains duplicates """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.features.retain([0, 1, 0])

    def test_features_retain_rangeIntoFEmpty(self):
        """ Test features.retain() retains all features using ranges """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        expTest = self.constructor(data, featureNames=featureNames)
        toTest.features.retain(start=0, end=2)

        assert toTest.isIdentical(expTest)


    def test_features_retain_handmadeRange(self):
        """ Test features.retain() against handmade output for range retention """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.features.retain(start=1, end=2)

        expectedTest = self.constructor([[2, 3], [5, 6], [8, 9]])

        assert expectedTest.isIdentical(toTest)

    def test_features_retain_range_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.features.retain(start=1, end=2)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_features_retain_handmadeWithFeatureNames(self):
        """ Test features.retain() against handmade output for range retention with FeatureNames """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain(start=1, end=2)

        expectedTest = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=["two", "three"])

        assert expectedTest.isIdentical(toTest)

    def test_features_retain_handmade_calling_featureNames(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain(start="two", end="three")

        expectedTest = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=["two", "three"])

        assert expectedTest.isIdentical(toTest)

    def test_features_retain_handmadeString(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['p1', 'p2', 'p3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('p2=5')
        expectedTest = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])

        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('p1==1')
        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])

        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('p3<9')
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])

        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('p3<=8')
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])

        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('p3>8')
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])

        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('p3>8.5')
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])

        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('p1!=1.0')
        expectedTest = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])

        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('p1<1')
        expectedTest = self.constructor([[], [], []], pointNames=pointNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('p1>0')
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

    def test_features_retain_handmadeStringWithOperatorWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['p1', 'p2', 'p3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('p2 = 5')
        expectedTest = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])

        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('p1 == 1')
        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])

        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('p3 < 9')
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])

        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('p3 <= 8')
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])

        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('p3 > 8')
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])

        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('p3 > 8.5')
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])

        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('p1 != 1.0')
        expectedTest = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])

        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('p1 < 1')
        expectedTest = self.constructor([[], [], []], pointNames=pointNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('p1 > 0')
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

    def test_features_retain_handmadeStringWithPointWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['pt 1', 'pt 2', 'pt 3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value with no operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('pt 2=5')
        expectedTest = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])

        assert expectedTest.isIdentical(toTest)

        #test pointName=value with operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('pt 2 = 5')
        expectedTest = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])

        assert expectedTest.isIdentical(toTest)

    @raises(InvalidArgumentValue)
    def test_features_retain_handmadeString_pointNotExist(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.features.retain('5=1')

    def test_features_retain_numberOnly(self):
        self.back_retain_numberOnly('feature')

    def test_features_retain_functionAndNumber(self):
        self.back_retain_functionAndNumber('feature')

    def test_features_retain_numberAndRandomizeAllData(self):
        self.back_retain_numberAndRandomizeAllData('feature')

    def test_features_retain_numberAndRandomizeSelectedData(self):
        self.back_retain_numberAndRandomizeSelectedData('feature')

    @raises(InvalidArgumentValueCombination)
    def test_features_retain_randomizeNoNumber(self):
        self.back_structural_randomizeNoNumber('retain', 'feature')

    @raises(InvalidArgumentValue)
    def test_features_retain_list_numberGreaterThanTargeted(self):
        self.back_structural_list_numberGreaterThanTargeted('retain', 'feature')

    @raises(InvalidArgumentValue)
    def test_features_retain_function_numberGreaterThanTargeted(self):
        self.back_structural_function_numberGreaterThanTargeted('retain', 'feature')

    @raises(InvalidArgumentValue)
    def test_features_retain_range_numberGreaterThanTargeted(self):
        self.back_structural_range_numberGreaterThanTargeted('retain', 'feature')

    ### using match module ###

    def test_features_retain_match_missing(self):
        toTest = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        toTest.features.retain(match.anyMissing)
        expTest = self.constructor([[1, 3], [None, None], [7, None], [7, 9]])
        expTest.features.setNames(['a', 'c'])
        assert toTest == expTest

        toTest = self.constructor([[1, 2, None], [None, 11, None], [7, 11, None], [7, 8, None]], featureNames=['a', 'b', 'c'])
        ret = toTest.features.retain(match.allMissing)
        expTest = self.constructor([[None], [None], [None], [None]])
        expTest.features.setNames(['c'])
        assert toTest == expTest

    def test_features_retain_match_nonNumeric(self):
        toTest = self.constructor([[1, 2, 3], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.features.retain(match.anyNonNumeric)
        expTest = self.constructor([[1, 3], ['a', 'c'], [7, 'c'], [7, 9]])
        expTest.features.setNames(['a', 'c'])
        assert toTest == expTest

        toTest = self.constructor([[1, 2, 'c'], ['a', 11, 'c'], [7, 11, 'c'], [7, 8, 'c']], featureNames=['a', 'b', 'c'])
        ret = toTest.features.retain(match.allNonNumeric)
        expTest = self.constructor([['c'], ['c'], ['c'], ['c']])
        expTest.features.setNames(['c'])
        assert toTest == expTest

    def test_features_retain_match_list(self):
        toTest = self.constructor([[1, 2, 3], ['a', 11, 'c'], ['x', 11, 'c'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.features.retain(match.anyValues(['a', 'c', 'x']))
        expTest = self.constructor([[1, 3], ['a', 'c'], ['x', 'c'], [7, 9]])
        expTest.features.setNames(['a', 'c'])
        assert toTest == expTest

        toTest = self.constructor([[1, 2, 'c'], ['a', 11, 'c'], ['x', 11, 'c'], [7, 8, 'c']], featureNames=['a', 'b', 'c'])
        ret = toTest.features.retain(match.allValues(['a', 'c', 'x']))
        expTest = self.constructor([['c'], ['c'], ['c'], ['c']])
        expTest.features.setNames(['c'])
        assert toTest == expTest

    def test_features_retain_match_function(self):
        toTest = self.constructor([[1, 2, 3], [-1, 11, -3], [-1, 11, -1], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        ret = toTest.features.retain(match.anyValues(lambda x: x < 0))
        expTest = self.constructor([[1, 3], [-1, -3], [-1, -1], [7, 9]])
        expTest.features.setNames(['a', 'c'])
        assert toTest == expTest

        toTest = self.constructor([[1, 2, -3], [-1, 11, -3], [-1, 11, -3], [7, 8, -3]], featureNames=['a', 'b', 'c'])
        ret = toTest.features.retain(match.allValues(lambda x: x < 0))
        expTest = self.constructor([[-3], [-3], [-3], [-3]])
        expTest.features.setNames(['c'])
        assert toTest == expTest

    #####################
    # referenceDataFrom #
    #####################

    @raises(InvalidArgumentType)
    def test_referenceDataFrom_exceptionWrongType(self):
        """ Test referenceDataFrom() throws exception when other is not the same type """
        data1 = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        featureNames = ['one', 'two', 'three']
        pNames = ['1', 'one', '2', '0']
        orig = self.constructor(data1, pointNames=pNames, featureNames=featureNames)

        retType0 = nimble.data.available[0]
        retType1 = nimble.data.available[1]

        objType0 = nimble.createData(retType0, data1, pointNames=pNames, featureNames=featureNames)
        objType1 = nimble.createData(retType1, data1, pointNames=pNames, featureNames=featureNames)

        # at least one of these two will be the wrong type
        orig.referenceDataFrom(objType0)
        orig.referenceDataFrom(objType1)

    @oneLogEntryExpected
    def test_referenceDataFrom_data_axisNames(self):
        data1 = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        featureNames = ['one', 'two', 'three']
        pNames = ['1', 'one', '2', '0']
        orig = self.constructor(data1, pointNames=pNames, featureNames=featureNames)

        data2 = [[-1, -2, -3, -4]]
        featureNames = ['1', '2', '3', '4']
        pNames = ['-1']
        other = self.constructor(data2, pointNames=pNames, featureNames=featureNames)

        ret = orig.referenceDataFrom(other)  # RET CHECK

        assert orig.data is other.data
        assert '-1' in orig.points.getNames()
        assert '1' in orig.features.getNames()
        assert ret is None

    @oneLogEntryExpected
    def test_referenceDataFrom_lazyNameGeneration(self):
        data1 = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        orig = self.constructor(data1)

        data2 = [[-1, -2, -3, -4]]
        other = self.constructor(data2)

        orig.referenceDataFrom(other)

        assertNoNamesGenerated(orig)
        assertNoNamesGenerated(other)

    @oneLogEntryExpected
    def test_referenceDataFrom_ObjName_Paths(self):
        data1 = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        featureNames = ['one', 'two', 'three']
        pNames = ['1', 'one', '2', '0']
        orig = self.constructor(data1, pointNames=pNames, featureNames=featureNames)

        data2 = [[-1, -2, -3, ]]
        featureNames = ['1', '2', '3']
        pNames = ['-1']
        other = self.constructor(data2, pointNames=pNames, featureNames=featureNames)

        orig._name = "testName"
        orig._absPath = "testAbsPath"
        orig._relPath = "testRelPath"

        other._name = "testNameother"
        other._absPath = "testAbsPathother"
        other._relPath = "testRelPathother"

        orig.referenceDataFrom(other)

        assert orig.name == "testName"
        assert orig.absolutePath == "testAbsPathother"
        assert orig.relativePath == 'testRelPathother'

        assert other.name == "testNameother"
        assert other.absolutePath == "testAbsPathother"
        assert other.relativePath == 'testRelPathother'

    @oneLogEntryExpected
    def test_referenceDataFrom_allMetadataAttributes(self):
        data1 = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        featureNames = ['one', 'two', 'three']
        pNames = ['1', 'one', '2', '0']
        orig = self.constructor(data1, pointNames=pNames, featureNames=featureNames)

        data2 = [[-1, -2, -3, 4, 5, 3, ], [-1, -2, -3, 4, 5, 3, ]]
        other = self.constructor(data2, )

        orig.referenceDataFrom(other)

        assert orig._pointCount == len(other.points)
        assert orig._featureCount == len(other.features)

        assert orig._nextDefaultValuePoint == other._nextDefaultValuePoint
        assert orig._nextDefaultValueFeature == other._nextDefaultValueFeature

    ######################
    # points.transform() #
    ######################

    @raises(InvalidArgumentType)
    def test_points_transform_exceptionInputNone(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(copy.deepcopy(origData), featureNames=featureNames)
        origObj.points.transform(None)

    @raises(ImproperObjectAction)
    def test_points_transform_exceptionPEmpty(self):
        data = [[], []]
        data = numpy.array(data).T
        origObj = self.constructor(data)

        def emitLower(point):
            return point[origObj.features.getIndex('deci')]

        origObj.points.transform(emitLower)

    @raises(ImproperObjectAction)
    def test_points_transform_exceptionFEmpty(self):
        data = [[], []]
        data = numpy.array(data)
        origObj = self.constructor(data)

        def emitLower(point):
            return point[origObj.features.getIndex('deci')]

        origObj.points.transform(emitLower)

    @raises(CalledFunctionException)
    @mock.patch('nimble.data.axis.constructIndicesList', calledException)
    def test_points_transform_calls_constructIndicesList(self):
        toTest = self.constructor([[1,2,],[3,4]], pointNames=['a', 'b'])

        toTest.points.transform(noChange, points=['a', 'b'])

    @oneLogEntryExpected
    def test_points_transform_Handmade(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(copy.deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitAllDeci(point):
            value = point[origObj.features.getIndex('deci')]
            return [value, value, value]

        lowerCounts = origObj.points.transform(emitAllDeci)  # RET CHECK

        expectedOut = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
        exp = self.constructor(expectedOut, pointNames=pointNames, featureNames=featureNames)

        assert lowerCounts is None
        assert origObj.isIdentical(exp)

    def test_points_transform_Handmade_lazyNameGeneration(self):
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(copy.deepcopy(origData))

        def emitAllDeci(point):
            value = point[1]
            return [value, value, value]

        lowerCounts = origObj.points.transform(emitAllDeci)  # RET CHECK

        assertNoNamesGenerated(origObj)

    def test_points_transform_NamePath_preservation(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(copy.deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitAllDeci(point):
            value = point[toTest.features.getIndex('deci')]
            return [value, value, value]

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.points.transform(emitAllDeci)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'

    @oneLogEntryExpected
    def test_points_transform_HandmadeLimited(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(copy.deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitAllDeci(point):
            value = point[origObj.features.getIndex('deci')]
            return [value, value, value]

        origObj.points.transform(emitAllDeci, points=[3, 'two'])

        expectedOut = [[1, 0.1, 0.01], [1, 0.1, 0.02], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
        exp = self.constructor(expectedOut, pointNames=pointNames, featureNames=featureNames)

        assert origObj.isIdentical(exp)


    def test_points_transform_nonZeroIterAndLen(self):
        origData = [[1, 1, 1], [1, 0, 2], [1, 1, 0], [0, 2, 0]]
        origObj = self.constructor(copy.deepcopy(origData))

        def emitNumNZ(point):
            ret = 0
            assert len(point) == 3
            for value in point.iterateElements(only=match.nonZero):
                ret += 1
            return [ret, ret, ret]

        origObj.points.transform(emitNumNZ)

        expectedOut = [[3, 3, 3], [2, 2, 2], [2, 2, 2], [1, 1, 1]]
        exp = self.constructor(expectedOut)

        assert origObj.isIdentical(exp)

    def test_points_transform_zerosReturned(self):

        def returnAllZero(pt):
            return [0 for val in pt]

        orig1 = self.constructor([[1, 2, 3], [1, 2, 3], [0, 0, 0]])
        exp1 = self.constructor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        orig1.points.transform(returnAllZero)
        assert orig1 == exp1

        def invert(pt):
            return [0 if v == 1 else 1 for v in pt]

        orig2 = self.constructor([[1, 1, 1], [0, 1, 0], [0, 0, 0]])
        exp2 = self.constructor([[0, 0, 0], [1, 0, 1], [1, 1, 1]])

        orig2.points.transform(invert)
        assert orig2 == exp2

    def test_points_transform_conversionWhenIntType(self):

        def addTenth(pt):
            return [v + 0.1 for v in pt]

        orig = self.constructor([[1, 2, 3], [4, 5, 6], [0, 0, 0]],
                                elementType=int)
        exp = self.constructor([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [0.1, 0.1, 0.1]])

        orig.points.transform(addTenth)
        assert orig == exp

    def test_points_transform_stringReturnsPreserved(self):

        def toString(pt):
            return [str(v) for v in pt]

        orig = self.constructor([[1, 2, 3], [4, 5, 6], [0, 0, 0]],
                                elementType=int)
        exp = self.constructor([['1', '2', '3'], ['4', '5', '6'], ['0', '0', '0']],
                               elementType=object)

        orig.points.transform(toString)
        assert orig == exp

    ########################
    # features.transform() #
    ########################

    @raises(ImproperObjectAction)
    def test_features_transform_exceptionPEmpty(self):
        data = [[], []]
        data = numpy.array(data).T
        origObj = self.constructor(data)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return 0
            return 1

        origObj.features.transform(emitAllEqual)

    @raises(ImproperObjectAction)
    def test_features_transform_exceptionFEmpty(self):
        data = [[], []]
        data = numpy.array(data)
        origObj = self.constructor(data)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return 0
            return 1

        origObj.features.transform(emitAllEqual)

    @raises(InvalidArgumentType)
    def test_features_transform_exceptionInputNone(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(copy.deepcopy(origData), featureNames=featureNames)
        origObj.features.transform(None)

    @raises(CalledFunctionException)
    @mock.patch('nimble.data.axis.constructIndicesList', calledException)
    def test_features_transform_calls_constructIndicesList(self):
        toTest = self.constructor([[1,2,],[3,4]], featureNames=['a', 'b'])

        toTest.features.transform(noChange, features=['a', 'b'])

    @oneLogEntryExpected
    def test_features_transform_Handmade(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(copy.deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return [0, 0, 0, 0]
            return [1, 1, 1, 1]

        lowerCounts = origObj.features.transform(emitAllEqual)  # RET CHECK
        expectedOut = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]
        exp = self.constructor(expectedOut, pointNames=pointNames, featureNames=featureNames)

        assert lowerCounts is None
        assert origObj.isIdentical(exp)

    def test_features_transform_Handmade_lazyNameGeneration(self):
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(copy.deepcopy(origData))

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return [0, 0, 0, 0]
            return [1, 1, 1, 1]

        lowerCounts = origObj.features.transform(emitAllEqual)  # RET CHECK

        assertNoNamesGenerated(origObj)

    def test_features_transform_NamePath_preservation(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(copy.deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return [0, 0, 0, 0]
            return [1, 1, 1, 1]

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.features.transform(emitAllEqual)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'

    @oneLogEntryExpected
    def test_features_transform_HandmadeLimited(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(copy.deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return [0, 0, 0, 0]
            return [1, 1, 1, 1]

        origObj.features.transform(emitAllEqual, features=[0, 'centi'])
        expectedOut = [[1, 0.1, 0], [1, 0.1, 0], [1, 0.1, 0], [1, 0.2, 0]]
        exp = self.constructor(expectedOut, pointNames=pointNames, featureNames=featureNames)

        assert origObj.isIdentical(exp)


    def test_features_transform_nonZeroIterAndLen(self):
        origData = [[1, 1, 1], [1, 0, 2], [1, 1, 0], [0, 2, 0]]
        origObj = self.constructor(copy.deepcopy(origData))

        def emitNumNZ(feature):
            ret = 0
            assert len(feature) == 4
            for value in feature.iterateElements(order='feature', only=match.nonZero):
                ret += 1
            return [ret, ret, ret, ret]

        origObj.features.transform(emitNumNZ)

        expectedOut = [[3, 3, 2], [3, 3, 2], [3, 3, 2], [3, 3, 2]]
        exp = self.constructor(expectedOut)

        assert origObj.isIdentical(exp)

    def test_features_transform_zerosReturned(self):

        def returnAllZero(ft):
            return [0 for val in ft]

        orig1 = self.constructor([[1, 2, 3], [1, 2, 3], [0, 0, 0]])
        exp1 = self.constructor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        orig1.features.transform(returnAllZero)
        assert orig1 == exp1

        def invert(ft):
            return [0 if v == 1 else 1 for v in ft]

        orig2 = self.constructor([[1, 1, 1], [0, 1, 0], [0, 0, 0]])
        exp2 = self.constructor([[0, 0, 0], [1, 0, 1], [1, 1, 1]])

        orig2.features.transform(invert)
        assert orig2 == exp2

    def test_features_transform_conversionWhenIntType(self):

        def addTenth(ft):
            return [v + 0.1 for v in ft]

        orig = self.constructor([[1, 2, 3], [4, 5, 6], [0, 0, 0]],
                                elementType=int)
        exp = self.constructor([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [0.1, 0.1, 0.1]])

        orig.features.transform(addTenth)
        assert orig == exp

    def test_features_transform_stringReturnsPreserved(self):

        def toString(ft):
            return [str(v) for v in ft]

        orig = self.constructor([[1, 2, 3], [4, 5, 6], [0, 0, 0]],
                                elementType=int)
        exp = self.constructor([['1', '2', '3'], ['4', '5', '6'], ['0', '0', '0']],
                               elementType=object)

        orig.features.transform(toString)
        assert orig == exp

    #######################
    # transformElements() #
    #######################

    @raises(CalledFunctionException)
    @mock.patch('nimble.data.base.constructIndicesList', calledException)
    def test_transformElements_calls_constructIndicesList1(self):
        toTest = self.constructor([[1,2],[3,4]], pointNames=['a', 'b'])

        def noChange(point):
            return point

        toTest.transformElements(noChange, points=['a', 'b'])

    @raises(CalledFunctionException)
    @mock.patch('nimble.data.base.constructIndicesList', calledException)
    def test_transformElements_calls_constructIndicesList2(self):
        toTest = self.constructor([[1,2],[3,4]], featureNames=['a', 'b'])

        def noChange(point):
            return point

        toTest.transformElements(noChange, features=['a', 'b'])

    @raises(InvalidArgumentValue)
    def test_transformElements_invalidElementReturned(self):
        data = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']]
        toTest = self.constructor(data)
        toTest.transformElements(lambda e: [e])

    @oneLogEntryExpected
    def test_transformElements_passthrough(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        ret = toTest.transformElements(passThrough)  # RET CHECK
        assert ret is None
        retRaw = toTest.copy(to="python list")

        assert [1, 2, 3] in retRaw
        assert [4, 5, 6] in retRaw
        assert [7, 8, 9] in retRaw
        assertNoNamesGenerated(toTest)


    def test_transformElements_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.transformElements(passThrough)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'

    @oneLogEntryExpected
    def test_transformElements_plusOnePreserve(self):
        data = [[1, 0, 3], [0, 5, 6], [7, 0, 9]]
        toTest = self.constructor(data)

        toTest.transformElements(plusOne, preserveZeros=True)
        retRaw = toTest.copy(to="python list")

        assert [2, 0, 4] in retRaw
        assert [0, 6, 7] in retRaw
        assert [8, 0, 10] in retRaw

    @oneLogEntryExpected
    def test_transformElements_plusOneExclude(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest.transformElements(plusOneOnlyEven, skipNoneReturnValues=True)
        retRaw = toTest.copy(to="python list")

        assert [1, 3, 3] in retRaw
        assert [5, 5, 7] in retRaw
        assert [7, 9, 9] in retRaw

    @oneLogEntryExpected
    def test_transformElements_plusOneLimited(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)

        toTest.transformElements(plusOneOnlyEven, points=1, features=[1, 'three'], skipNoneReturnValues=True)
        retRaw = toTest.copy(to="python list")

        assert [1, 2, 3] in retRaw
        assert [4, 5, 7] in retRaw
        assert [7, 8, 9] in retRaw

    @oneLogEntryExpected
    def test_transformElements_DictionaryAllMapped(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {1:9, 2:8, 3:7, 4:6, 5:5, 6:4, 7:3, 8:2, 9:1}
        expData = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformElements(transformMapping)

        assert toTest.isIdentical(expTest)

    @oneLogEntryExpected
    def test_transformElements_DictionaryAllMappedStrings(self):
        data = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {"a": 1, "b":2, "c":3, "d":4, "e":5, "f":6, "g":7, "h":8, "i": 9}
        expData = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformElements(transformMapping)

        assert toTest.isIdentical(expTest)


    def test_transformElements_DictionarySomeMapped(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {2:8, 8:2}
        expData = [[1, 8, 3], [4, 5, 6], [7, 2, 9]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformElements(transformMapping)

        assert toTest.isIdentical(expTest)


    def test_transformElements_DictionaryMappedNotInPoints(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {2:8, 8:2}
        expData = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformElements(transformMapping, points=1)

        assert toTest.isIdentical(expTest)


    def test_transformElements_DictionaryMappedNotInFeatures(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {2:8, 8:2}
        expData = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformElements(transformMapping, features=0)

        assert toTest.isIdentical(expTest)


    def test_transformElements_DictionaryPreserveZerosNoZeroMap(self):
        data = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {1:2}
        expData = [[0, 0, 0], [2, 2, 2], [0, 0, 0]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformElements(transformMapping, preserveZeros=True)

        assert toTest.isIdentical(expTest)


    def test_transformElements_DictionaryPreserveZerosZeroMapZero(self):
        data = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {0:0, 1:2}
        expData = [[0, 0, 0], [2, 2, 2], [0, 0, 0]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformElements(transformMapping, preserveZeros=True)

        assert toTest.isIdentical(expTest)


    def test_transformElements_DictionaryPreserveZerosZeroMapNonZero(self):
        data = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {0:100, 1:2}
        expData = [[0, 0, 0], [2, 2, 2], [0, 0, 0]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformElements(transformMapping, preserveZeros=True)

        assert toTest.isIdentical(expTest)


    def test_transformElements_DictionaryDoNotPreserveZerosZeroMapNonZero(self):
        data = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {0:100}
        expData = [[100, 100, 100], [1, 1, 1], [100, 100, 100]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformElements(transformMapping, preserveZeros=False)

        assert toTest.isIdentical(expTest)


    def test_transformElements_DictionarySkipNoneReturn(self):
        data = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {1:None}
        expData = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformElements(transformMapping, skipNoneReturnValues=True)

        assert toTest.isIdentical(expTest)


    def test_transformElements_DictionaryDoNotSkipNoneReturn(self):
        data = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {1: None}
        expData = [[0, 0, 0], [None, None, None], [0, 0, 0]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names, treatAsMissing=None)
        toTest.transformElements(transformMapping, skipNoneReturnValues=False)

        assert toTest.isIdentical(expTest)

    def test_transformElements_zerosReturned(self):

        def returnAllZero(elem):
            return 0

        orig1 = self.constructor([[1, 2, 3], [1, 2, 3], [0, 0, 0]])
        exp1 = self.constructor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        orig1.transformElements(returnAllZero)
        assert orig1 == exp1

        def invert(elem):
            return 0 if elem == 1 else 1

        orig2 = self.constructor([[1, 1, 1], [0, 1, 0], [0, 0, 0]])
        exp2 = self.constructor([[0, 0, 0], [1, 0, 1], [1, 1, 1]])

        orig2.transformElements(invert)
        assert orig2 == exp2

        orig3 = self.constructor([[1, 1, 1], [0, 1, 0], [0, 0, 0]])
        exp3 = self.constructor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        orig3.transformElements(invert, preserveZeros=True)
        assert orig3 == exp3

    def test_transformElements_conversionWhenIntType(self):

        def addTenth(elem):
            return elem + 0.1

        orig = self.constructor([[1, 2, 3], [4, 5, 6], [0, 0, 0]],
                                elementType=int)
        exp = self.constructor([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [0.1, 0.1, 0.1]])

        orig.transformElements(addTenth)
        assert orig == exp

    def test_transformElements_stringReturnsPreserved(self):

        def toString(e):
            return str(e)

        orig = self.constructor([[1, 2, 3], [4, 5, 6], [0, 0, 0]],
                                elementType=int)
        exp = self.constructor([['1', '2', '3'], ['4', '5', '6'], ['0', '0', '0']],
                               elementType=object)

        orig.transformElements(toString)
        assert orig == exp

    ######################
    # replaceRectangle() #
    ######################

    # replaceRectangle(self, values, pointStart, featureStart, pointEnd, featureEnd)

    def test_replaceRectangle_acceptableValues(self):
        raw = [[1, 2], [3, 4]]
        toTest = self.constructor(raw)

        try:
            toTest.replaceRectangle(set([1, 3]), 0, 0, 0, 1)
            assert False  # expected InvalidArgumentType
        except InvalidArgumentType as iat:
            pass

        try:
            toTest.replaceRectangle(lambda x: x * x, 0, 0, 0, 1)
            assert False  # expected InvalidArgumentType
        except InvalidArgumentType as iat:
            pass


    def test_replaceRectangle_sizeMismatch(self):
        raw = [[1, 2], [3, 4]]
        toTest = self.constructor(raw)

        raw = [[-1, -2]]
        val = self.constructor(raw)

        try:
            toTest.replaceRectangle(val, 0, 0, 1, 1)
            assert False  # expected InvalidArgumentValueCombination
        except InvalidArgumentValueCombination as ivc:
            pass

        val.transpose()

        try:
            toTest.replaceRectangle(val, 0, 0, 1, 1)
            assert False  # expected InvalidArgumentValueCombination
        except InvalidArgumentValueCombination as ivc:
            pass


    def test_replaceRectangle_invalidID(self):
        raw = [[1, 2], [3, 4]]
        toTest = self.constructor(raw)

        val = 1

        try:
            toTest.replaceRectangle(val, "hello", 0, 1, 1)
            assert False  # expected KeyError
        except KeyError:
            pass
        try:
            toTest.replaceRectangle(val, 0, "Wrong", 1, 1)
            assert False  # expected KeyError
        except KeyError:
            pass
        try:
            toTest.replaceRectangle(val, 0, 0, 2, 1)
            assert False  # expected IndexError
        except IndexError:
            pass
        try:
            toTest.replaceRectangle(val, 0, 0, 1, -12)
            assert False  # expected IndexError
        except IndexError as iav:
            pass


    def test_replaceRectangle_start_lessThan_end(self):
        raw = [[1, 2], [3, 4]]
        toTest = self.constructor(raw)

        val = 1

        try:
            toTest.replaceRectangle(val, 1, 0, 0, 1)
            assert False  # expected InvalidArgumentValueCombination
        except InvalidArgumentValueCombination as ivc:
            pass
        try:
            toTest.replaceRectangle(val, 0, 1, 1, 0)
            assert False  # expected InvalidArgumentValueCombination
        except InvalidArgumentValueCombination as ivc:
            pass

    @oneLogEntryExpected
    def test_replaceRectangle_fullObjectFill(self):
        raw = [[1, 2], [3, 4]]
        toTest = self.constructor(raw)

        arg = [[-1, -2], [-3, -4]]
        arg = self.constructor(arg)
        exp = arg.copy()

        ret = toTest.replaceRectangle(arg, 0, 0, len(toTest.points) - 1, len(toTest.features) - 1)
        assert ret is None

        arg *= 10

        assert toTest == exp
        assert toTest != arg
        assertNoNamesGenerated(toTest)

    @twoLogEntriesExpected
    def test_replaceRectangle_vectorFill(self):
        raw = [[1, 2], [3, 4]]
        toTestP = self.constructor(raw)
        toTestF = self.constructor(raw)

        rawP = [[-1, -2]]
        valP = self.constructor(rawP)

        rawF = [[-1], [-3]]
        valF = self.constructor(rawF)

        expP = [[-1, -2], [3, 4]]
        expP = self.constructor(expP)

        expF = [[-1, 2], [-3, 4]]
        expF = self.constructor(expF)

        toTestP.replaceRectangle(valP, 0, 0, 0, 1)
        assert toTestP == expP

        toTestF.replaceRectangle(valF, 0, 0, 1, 0)
        assert toTestF == expF


    def test_replaceRectangle_offsetSquare(self):
        raw = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
        base = self.constructor(raw)
        trialRaw = [[0, 0], [0, 0]]
        trial = self.constructor(trialRaw)

        leftCorner = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for p, f in leftCorner:
            toTest = base.copy()

            toTest.replaceRectangle(trial, p, f, p + 1, f + 1)
            assert toTest[p, f] == 0
            assert toTest[p + 1, f] == 0
            assert toTest[p, f + 1] == 0
            assert toTest[p + 1, f + 1] == 0

    @logCountAssertionFactory(4)
    def test_replaceRectangle_constants(self):
        toTest0 = self.constructor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        exp0 = self.constructor([[0, 1, 1], [0, 1, 1], [0, 0, 0]])
        toTest0.replaceRectangle(1, 0, 1, 1, 2)
        assert toTest0 == exp0

        toTest1 = self.constructor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        exp1 = self.constructor([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
        toTest1.replaceRectangle(0, 0, 1, 2, 1)
        assert toTest1 == exp1

        toTestI = self.constructor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        expi = self.constructor([[1, 0, 2], [0, 1, 0], [2, 0, 1]])
        toTestI.replaceRectangle(2, 0, 2, 0, 2)
        toTestI.replaceRectangle(2, 2, 0, 2, 0)
        assert toTestI == expi


    def test_fillWIth_differentType(self):
        raw = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
        fill = [[0, 0], [0, 0]]
        exp = [[0, 0, 13], [0, 0, 23], [31, 32, 33]]
        exp = self.constructor(exp)
        for t in nimble.data.available:
            toTest = self.constructor(raw)
            arg = nimble.createData(t, fill)
            toTest.replaceRectangle(arg, 0, 0, 1, 1)
            assert toTest == exp

    ###########################################
    # flattenToOnePoint | flattenToOneFeature #
    ###########################################

    # exception: either axis empty
    def test_flattenToOnePoint_empty(self):
        self.back_flatten_empty('point')

    def test_flattenToOneFeature_empty(self):
        self.back_flatten_empty('feature')

    def back_flatten_empty(self, axis):
        checkMsg = False
        target = "flattenToOnePoint" if axis == 'point' else "flattenToOneFeature"

        pempty = self.constructor(numpy.empty((0,2)))
        exceptionHelper(pempty, target, [], ImproperObjectAction, checkMsg)

        fempty = self.constructor(numpy.empty((4,0)))
        exceptionHelper(fempty, target, [], ImproperObjectAction, checkMsg)

        trueEmpty = self.constructor(numpy.empty((0,0)))
        exceptionHelper(trueEmpty, target, [], ImproperObjectAction, checkMsg)


    # flatten single p/f - see name changes
    def test_flattenToOnePoint_vector(self):
        self.back_flatten_vector('point')

    def test_flattenToOneFeature_vector(self):
        self.back_flatten_vector('feature')

    @oneLogEntryExpected
    def back_flatten_vector(self, axis):
        target = "flattenToOnePoint" if axis == 'point' else "flattenToOneFeature"
        raw = [1, -1, 2, -2, 3, -3, 4, -4]
        vecNames = ['vector']
        longNames = ['one+', 'one-', 'two+', 'two-', 'three+', 'three-', 'four+', 'four-',]

        testObj = self.constructor(raw, pointNames=vecNames, featureNames=longNames)

        expLongNames = ['one+ | vector', 'one- | vector',
                        'two+ | vector', 'two- | vector',
                        'three+ | vector', 'three- | vector',
                        'four+ | vector', 'four- | vector']
#        expLongNames = [n + ' | ' + vecNames[0] for n in longNames]
        expObj = self.constructor(raw, pointNames=['Flattened'], featureNames=expLongNames)

        if axis != 'point':
            testObj.transpose(useLog=False)
            expObj.transpose(useLog=False)

        ret = getattr(testObj, target)()

        assert testObj == expObj
        assert ret is None  # in place op, nothing returned


    def test_flattenToOnePoint_handMade_valuesOnly(self):
        dataRaw = [["p1,f1", "p1,f2"], ["p2,f1", "p2,f2"]]
        expRaw = [["p1,f1", "p1,f2", "p2,f1", "p2,f2"]]
        testObj = self.constructor(dataRaw)

        testObj.flattenToOnePoint()

        expObj = self.constructor(expRaw, pointNames=["Flattened"])
        assert testObj == expObj

    def test_flattenToOneFeature_handMade_valuesOnly(self):
        dataRaw = [["p1,f1", "p1,f2"], ["p2,f1", "p2,f2"]]
        expRaw = [["p1,f1"], ["p2,f1"], ["p1,f2"], ["p2,f2"]]
        testObj = self.constructor(dataRaw)

        testObj.flattenToOneFeature()

        expObj = self.constructor(expRaw, featureNames=["Flattened"])

        assert testObj == expObj


    # flatten rectangular object
    def test_flattenToOnePoint_rectangleRandom(self):
        self.back_flatten_rectangleRandom('point')

    def test_flattenToOneFeature_rectangleRandom(self):
        self.back_flatten_rectangleRandom('feature')

    @oneLogEntryExpected
    def back_flatten_rectangleRandom(self, axis):
        target = "flattenToOnePoint" if axis == 'point' else "flattenToOneFeature"
        order = 'C' if axis == 'point' else 'F'  # controls row or column major flattening
        discardAxisLen = 30
        keptAxisLen = 50
        shape = (discardAxisLen, keptAxisLen) if axis == 'point' else (keptAxisLen, discardAxisLen)
        endLength = discardAxisLen * keptAxisLen
        targetShape = (1, endLength) if axis == 'point' else (endLength, 1)
        origRaw = numpyRandom.randint(0, 2, shape)  # array of ones and zeroes
        expRaw = numpy.reshape(origRaw, targetShape, order)

        testObj = self.constructor(origRaw)
        copyObj = testObj.copy()  # freeze the default axis names to check against later
        if axis == 'point':
            expObj = self.constructor(expRaw, pointNames=['Flattened'])
        else:
            expObj = self.constructor(expRaw, featureNames=['Flattened'])

        getattr(testObj, target)()
        assert testObj == expObj

        # default names are ignored by ==, so we explicitly check them in this test
        keptAxisNames = copyObj.features.getNames() if axis == 'point' else copyObj.points.getNames()
        discardAxisNames = copyObj.points.getNames() if axis == 'point' else copyObj.features.getNames()
        check = testObj.features.getNames() if axis == 'point' else testObj.points.getNames()

        for i,name in enumerate(check):
            splitName = name.split(' | ')
            assert len(splitName) == 2
            # we cycle through the names from the kept axis
            assert splitName[0] == keptAxisNames[i % keptAxisLen]
            # we have to go through all of the names of the kept axis before we increment
            # the name from the discarded axis
            assert splitName[1] == discardAxisNames[i // keptAxisLen]


    ###################################################
    # unflattenFromOnePoint | unflattenFromOneFeature #
    ###################################################

    # exception: either axis empty
    def test_unflattenFromOnePoint_empty(self):
        self.back_unflatten_empty('point')

    def test_unflattenFromOneFeature_empty(self):
        self.back_unflatten_empty('feature')

    def back_unflatten_empty(self, axis):
        checkMsg = False
        target = "unflattenFromOnePoint" if axis == 'point' else "unflattenFromOneFeature"
        single = (0,2) if axis == 'point' else (2,0)

        singleEmpty = self.constructor(numpy.empty(single))
        exceptionHelper(singleEmpty, target, [2], ImproperObjectAction, checkMsg)

        trueEmpty = self.constructor(numpy.empty((0,0)))
        exceptionHelper(trueEmpty, target, [2], ImproperObjectAction, checkMsg)


    # exceptions: opposite vector, 2d data
    def test_unflattenFromOnePoint_wrongShape(self):
        self.back_unflatten_wrongShape('point')

    def test_unflattenFromOneFeature_wrongShape(self):
        self.back_unflatten_wrongShape('feature')

    def back_unflatten_wrongShape(self, axis):
        checkMsg = False
        target = "unflattenFromOnePoint" if axis == 'point' else "unflattenFromOneFeature"
        vecShape = (4,1) if axis == 'point' else (1,4)

        wrongVector = self.constructor(numpyRandom.rand(*vecShape))
        exceptionHelper(wrongVector, target, [2], ImproperObjectAction, checkMsg)

        rectangle = self.constructor(numpyRandom.rand(4,4))
        exceptionHelper(rectangle, target, [2], ImproperObjectAction, checkMsg)


    # exception: numPoints / numFeatures does not divide length of mega P/F
    def test_unflattenFromOnePoint_doesNotDivide(self):
        self.back_unflatten_doesNotDivide('point')

    def test_unflattenFromOneFeature_doesNotDivide(self):
        self.back_unflatten_doesNotDivide('feature')

    def back_unflatten_doesNotDivide(self, axis):
        checkMsg = False
        target = "unflattenFromOnePoint" if axis == 'point' else "unflattenFromOneFeature"
        primeLength = (1,7) if axis == 'point' else (7,1)
        divisableLength = (1,8) if axis == 'point' else (8,1)

        undivisable = self.constructor(numpyRandom.rand(*primeLength))
        exceptionHelper(undivisable, target, [2], InvalidArgumentValue, checkMsg)

        divisable = self.constructor(numpyRandom.rand(*divisableLength))
        exceptionHelper(divisable, target, [5], InvalidArgumentValue, checkMsg)


    # exception: unflattening would destroy an axis name
    def test_unflattenFromOnePoint_nameDestroyed(self):
        self.back_unflatten_nameDestroyed('point')

    def test_unflattenFromOneFeature_nameDestroyed(self):
        self.back_unflatten_nameDestroyed('feature')

    def back_unflatten_nameDestroyed(self, axis):
        checkMsg = False
        target = "unflattenFromOnePoint" if axis == 'point' else "unflattenFromOneFeature"
        vecShape = (1,4) if axis == 'point' else (4,1)
        data = numpyRandom.rand(*vecShape)

        # non-default name, flattened axis
        args = {"pointNames":["non-default"]} if axis == 'point' else {"featureNames":["non-default"]}
        testObj = self.constructor(data, **args)
        exceptionHelper(testObj, target, [2], ImproperObjectAction, checkMsg)

        # all non-default names, unflattened axis
        names = ["a", "b", "c", "d"]
        args = {"featureNames":names} if axis == 'point' else {"pointNames":names}
        testObj = self.constructor(data, **args)
        exceptionHelper(testObj, target, [2], ImproperObjectAction, checkMsg)

        # single non-default name, unflattened axis
        testObj = self.constructor(data)
        if axis == 'point':
            testObj.features.setName(1, "non-default")
        else:
            testObj.points.setName(2, "non-default")
        exceptionHelper(testObj, target, [2], ImproperObjectAction, checkMsg)

    # exception: unflattening would destroy an axis name
    def test_unflattenFromOnePoint_nameFormatInconsistent(self):
        self.back_unflatten_nameFormatInconsistent('point')

    def test_unflattenFromOneFeature_nameFormatInconsistent(self):
        self.back_unflatten_nameFormatInconsistent('feature')

    def back_unflatten_nameFormatInconsistent(self, axis):
        checkMsg = False
        target = "unflattenFromOnePoint" if axis == 'point' else "unflattenFromOneFeature"
        vecShape = (1,4) if axis == 'point' else (4,1)
        data = numpyRandom.rand(*vecShape)

        # unflattend axis, mix of default names and correctly formatted
        names = ["a | 1", "b | 1", "a | 2", "b | 2"]
        args = {"featureNames":names} if axis == 'point' else {"pointNames":names}
        testObj = self.constructor(data, **args)
        if axis == 'point':
            testObj.features.setName(1, None)
        else:
            testObj.points.setName(1, None)
        exceptionHelper(testObj, target, [2], ImproperObjectAction, checkMsg)

        # unflattened axis, inconsistent along original unflattened axis
        names = ["a | 1", "b | 1", "a | 2", "c | 2"]
        args = {"featureNames":names} if axis == 'point' else {"pointNames":names}
        testObj = self.constructor(data, **args)
        exceptionHelper(testObj, target, [2], ImproperObjectAction, checkMsg)

        # unflattened axis, inconsistent along original flattened axis
        names = ["a | 1", "b | 2", "a | 2", "b | 3"]
        args = {"featureNames":names} if axis == 'point' else {"pointNames":names}
        testObj = self.constructor(data, **args)
        exceptionHelper(testObj, target, [2], ImproperObjectAction, checkMsg)


    # unflatten something that was flattened - include name transformation
    @oneLogEntryExpected
    def test_unflattenFromOnePoint_handmadeWithNames(self):
        raw = [["el0", "el1", "el2", "el3", "el4", "el5"]]
        rawNames = ["1 | A", "2 | A", "3 | A", "1 | B", "2 | B", "3 | B"]
        toTest = self.constructor(raw, pointNames=["Flattened"], featureNames=rawNames)
        expData = numpy.array([["el0", "el1", "el2"], ["el3", "el4", "el5"]])
        namesP = ["A", "B"]
        namesF = ["1", "2", "3"]

        exp = self.constructor(expData, pointNames=namesP, featureNames=namesF)

        toTest.unflattenFromOnePoint(2)
        assert toTest == exp

    # unflatten something that was flattened - include name transformation
    @oneLogEntryExpected
    def test_unflattenFromOneFeature_handmadeWithNames(self):
        raw = [["el0"], ["el1"], ["el2"], ["el3"], ["el4"], ["el5"]]
        rawNames = ["1 | A", "2 | A", "3 | A", "1 | B", "2 | B", "3 | B"]
        toTest = self.constructor(raw, pointNames=rawNames, featureNames=["Flattened"])
        expData = [["el0", "el3"],["el1", "el4"], ["el2", "el5"]]
        namesP = ["1", "2", "3"]
        namesF = ["A", "B"]

        exp = self.constructor(expData, pointNames=namesP, featureNames=namesF)

        toTest.unflattenFromOneFeature(2)
        assert toTest == exp


    # unflatten something that is just a vector - default names
    def test_unflattenFromOnePoint_handmadeDefaultNames(self):
        self.back_unflatten_handmadeDefaultNames('point')

    def test_unflattenFromOneFeature_handmadeDefaultNames(self):
        self.back_unflatten_handmadeDefaultNames('feature')

    @oneLogEntryExpected
    def back_unflatten_handmadeDefaultNames(self, axis):
        target = "unflattenFromOnePoint" if axis == 'point' else "unflattenFromOneFeature"
        raw = [[1, 10, 20, 2]]
        toTest = self.constructor(raw)
        expData = numpy.array([[1,10],[20,2]])

        if axis == 'point':
            exp = self.constructor(expData)
        else:
            toTest.transpose(useLog=False)
            exp = self.constructor(expData.T)

        getattr(toTest, target)(2)
        assert toTest == exp

        # check that the name conforms to the standards of how nimble objects assign
        # default names
        def checkName(n):
            assert n.startswith(DEFAULT_PREFIX)
            assert int(n[len(DEFAULT_PREFIX):]) >= 0

        list(map(checkName, toTest.points.getNames()))
        list(map(checkName, toTest.features.getNames()))


    # random round trip
    def test_flatten_to_unflatten_point_roundTrip(self):
        self.back_flatten_to_unflatten_roundTrip('point')

    def test_flatten_to_unflatten_feature_roundTrip(self):
        self.back_flatten_to_unflatten_roundTrip('feature')

    @logCountAssertionFactory(4)
    def back_flatten_to_unflatten_roundTrip(self, axis):
        targetDown = "flattenToOnePoint" if axis == 'point' else "flattenToOneFeature"
        targetUp = "unflattenFromOnePoint" if axis == 'point' else "unflattenFromOneFeature"
        discardAxisLen = 30
        keptAxisLen = 50
        shape = (discardAxisLen, keptAxisLen) if axis == 'point' else (keptAxisLen, discardAxisLen)
        origRaw = numpyRandom.randint(0, 2, shape)  # array of ones and zeroes
        namesDiscard = list(map(str, numpyRandom.choice(100, discardAxisLen, replace=False).tolist()))
        namesKept = list(map(str, numpyRandom.choice(100, keptAxisLen, replace=False).tolist()))
        namesArgs = {"pointNames":namesDiscard, "featureNames":namesKept} if axis == 'point' else {"pointNames":namesKept, "featureNames":namesDiscard}

        testObj = self.constructor(origRaw, **namesArgs)
        expObj = testObj.copy()

        getattr(testObj, targetDown)()
        getattr(testObj, targetUp)(discardAxisLen)
        assert testObj == expObj

        # second round to see if status of hidden internal variable are still viable
        getattr(testObj, targetDown)()
        getattr(testObj, targetUp)(discardAxisLen)
        assert testObj == expObj

#     ###############################################
#     # points.flattenToOne | features.flattenToOne #
#     ###############################################
#
#     # exception: either axis empty
#     def test_points_flattenToOne_empty(self):
#         self.back_flatten_empty('point')
#
#     def test_features_flattenToOne_empty(self):
#         self.back_flatten_empty('feature')
#
#     def back_flatten_empty(self, axis):
#         checkMsg = True
#         target = (axis + 's', 'flattenToOne')
#         pempty = self.constructor(numpy.empty((0,2)))
#         exceptionHelper(pempty, target, [], ImproperObjectAction, checkMsg)
#
#         fempty = self.constructor(numpy.empty((4,0)))
#         exceptionHelper(fempty, target, [], ImproperObjectAction, checkMsg)
#
#         trueEmpty = self.constructor(numpy.empty((0,0)))
#         exceptionHelper(trueEmpty, target, [], ImproperObjectAction, checkMsg)
#
#
#     # flatten single p/f - see name changes
#     def test_points_flattenToOne_vector(self):
#         self.back_flatten_vector('point')
#
#     def test_features_flattenToOne_vector(self):
#         self.back_flatten_vector('feature')
#
#     def back_flatten_vector(self, axis):
#         raw = [1, -1, 2, -2, 3, -3, 4, -4]
#         vecNames = ['vector']
#         longNames = ['one+', 'one-', 'two+', 'two-', 'three+', 'three-', 'four+', 'four-',]
#
#         testObj = self.constructor(raw, pointNames=vecNames, featureNames=longNames)
#
#         expLongNames = ['one+ | vector', 'one- | vector',
#                         'two+ | vector', 'two- | vector',
#                         'three+ | vector', 'three- | vector',
#                         'four+ | vector', 'four- | vector']
# #        expLongNames = [n + ' | ' + vecNames[0] for n in longNames]
#         expObj = self.constructor(raw, pointNames=['Flattened'], featureNames=expLongNames)
#
#         if axis != 'point':
#             testObj.transpose()
#             expObj.transpose()
#
#         axisObj = getattr(testObj, axis + 's')
#         ret = getattr(axisObj, 'flattenToOne')()
#
#         assert testObj == expObj
#         assert ret is None  # in place op, nothing returned
#
#
#     def test_points_flattenToOne_handMade_valuesOnly(self):
#         dataRaw = [["p1,f1", "p1,f2"], ["p2,f1", "p2,f2"]]
#         expRaw = [["p1,f1", "p1,f2", "p2,f1", "p2,f2"]]
#         testObj = self.constructor(dataRaw)
#
#         testObj.points.flattenToOne()
#
#         expObj = self.constructor(expRaw, pointNames=["Flattened"])
#
#         assert testObj == expObj
#
#     def test_features_flattenToOne_handMade_valuesOnly(self):
#         dataRaw = [["p1,f1", "p1,f2"], ["p2,f1", "p2,f2"]]
#         expRaw = [["p1,f1"], ["p2,f1"], ["p1,f2"], ["p2,f2"]]
#         testObj = self.constructor(dataRaw)
#
#         testObj.features.flattenToOne()
#
#         expObj = self.constructor(expRaw, featureNames=["Flattened"])
#
#         assert testObj == expObj
#
#
#     # flatten rectangular object
#     def test_points_flattenToOne_rectangleRandom(self):
#         self.back_flatten_rectangleRandom('point')
#
#     def test_features_flattenToOne_rectangleRandom(self):
#         self.back_flatten_rectangleRandom('feature')
#
#     def back_flatten_rectangleRandom(self, axis):
#         order = 'C' if axis == 'point' else 'F'  # controls row or column major flattening
#         discardAxisLen = 30
#         keptAxisLen = 50
#         shape = (discardAxisLen, keptAxisLen) if axis == 'point' else (keptAxisLen, discardAxisLen)
#         endLength = discardAxisLen * keptAxisLen
#         targetShape = (1, endLength) if axis == 'point' else (endLength, 1)
#         origRaw = numpyRandom.randint(0, 2, shape)  # array of ones and zeroes
#         expRaw = numpy.reshape(origRaw, targetShape, order)
#
#         testObj = self.constructor(origRaw)
#         copyObj = testObj.copy()  # freeze the default axis names to check against later
#         if axis == 'point':
#             expObj = self.constructor(expRaw, pointNames=['Flattened'])
#         else:
#             expObj = self.constructor(expRaw, featureNames=['Flattened'])
#
#         axisObj = getattr(testObj, axis + 's')
#         getattr(axisObj, 'flattenToOne')()
#         assert testObj == expObj
#
#         # default names are ignored by ==, so we explicitly check them in this test
#         keptAxisNames = copyObj.features.getNames() if axis == 'point' else copyObj.points.getNames()
#         discardAxisNames = copyObj.points.getNames() if axis == 'point' else copyObj.features.getNames()
#         check = testObj.features.getNames() if axis == 'point' else testObj.points.getNames()
#
#         for i, name in enumerate(check):
#             splitName = name.split(' | ')
#             assert len(splitName) == 2
#             # we cycle through the names from the kept axis
#             assert splitName[0] == keptAxisNames[i % keptAxisLen]
#             # we have to go through all of the names of the kept axis before we increment
#             # the name from the discarded axis
#             assert splitName[1] == discardAxisNames[i // keptAxisLen]
#
#
#     ###################################################
#     # points.unflattenFromOne | features.unflattenFromOne #
#     ###################################################
#
#     # exception: either axis empty
#     def test_points_unflattenFromOne_empty(self):
#         self.back_unflatten_empty('point')
#
#     def test_features_unflattenFromOne_empty(self):
#         self.back_unflatten_empty('feature')
#
#     def back_unflatten_empty(self, axis):
#         checkMsg = True
#         target = (axis + 's', 'unflattenFromOne')
#         single = (0,2) if axis == 'point' else (2,0)
#
#         singleEmpty = self.constructor(numpy.empty(single))
#         exceptionHelper(singleEmpty, target, [2], ImproperObjectAction, checkMsg)
#
#         trueEmpty = self.constructor(numpy.empty((0,0)))
#         exceptionHelper(trueEmpty, target, [2], ImproperObjectAction, checkMsg)
#
#
#     # exceptions: opposite vector, 2d data
#     def test_points_unflattenFromOne_wrongShape(self):
#         self.back_unflatten_wrongShape('point')
#
#     def test_features_unflattenFromOne_wrongShape(self):
#         self.back_unflatten_wrongShape('feature')
#
#     def back_unflatten_wrongShape(self, axis):
#         checkMsg = True
#         target = (axis + 's', 'unflattenFromOne')
#         vecShape = (4,1) if axis == 'point' else (1,4)
#
#         wrongVector = self.constructor(numpyRandom.rand(*vecShape))
#         exceptionHelper(wrongVector, target, [2], ImproperObjectAction, checkMsg)
#
#         rectangle = self.constructor(numpyRandom.rand(4,4))
#         exceptionHelper(rectangle, target, [2], ImproperObjectAction, checkMsg)
#
#
#     # exception: numPoints / numFeatures does not divide length of mega P/F
#     def test_points_unflattenFromOne_doesNotDivide(self):
#         self.back_unflatten_doesNotDivide('point')
#
#     def test_features_unflattenFromOne_doesNotDivide(self):
#         self.back_unflatten_doesNotDivide('feature')
#
#     def back_unflatten_doesNotDivide(self, axis):
#         checkMsg = True
#         target = (axis + 's', 'unflattenFromOne')
#         primeLength = (1,7) if axis == 'point' else (7,1)
#         divisableLength = (1,8) if axis == 'point' else (8,1)
#
#         undivisable = self.constructor(numpyRandom.rand(*primeLength))
#         exceptionHelper(undivisable, target, [2], InvalidArgumentValue, checkMsg)
#
#         divisable = self.constructor(numpyRandom.rand(*divisableLength))
#         exceptionHelper(divisable, target, [5], InvalidArgumentValue, checkMsg)
#
#
#     # exception: unflattening would destroy an axis name
#     def test_points_unflattenFromOne_nameDestroyed(self):
#         self.back_unflatten_nameDestroyed('point')
#
#     def test_features_unflattenFromOne_nameDestroyed(self):
#         self.back_unflatten_nameDestroyed('feature')
#
#     def back_unflatten_nameDestroyed(self, axis):
#         checkMsg = True
#         target = (axis + 's', 'unflattenFromOne')
#         vecShape = (1,4) if axis == 'point' else (4,1)
#         data = numpyRandom.rand(*vecShape)
#
#         # non-default name, flattened axis
#         args = {"pointNames":["non-default"]} if axis == 'point' else {"featureNames":["non-default"]}
#         testObj = self.constructor(data, **args)
#         exceptionHelper(testObj, target, [2], ImproperObjectAction, checkMsg)
#
#         # all non-default names, unflattened axis
#         names = ["a", "b", "c", "d"]
#         args = {"featureNames":names} if axis == 'point' else {"pointNames":names}
#         testObj = self.constructor(data, **args)
#         exceptionHelper(testObj, target, [2], ImproperObjectAction, checkMsg)
#
#         # single non-default name, unflattened axis
#         testObj = self.constructor(data)
#         if axis == 'point':
#             testObj.features.setName(1, "non-default")
#         else:
#             testObj.points.setName(2, "non-default")
#         exceptionHelper(testObj, target, [2], ImproperObjectAction, checkMsg)
#
#     # exception: unflattening would destroy an axis name
#     def test_points_unflattenFromOne_nameFormatInconsistent(self):
#         self.back_unflatten_nameFormatInconsistent('point')
#
#     def test_features_unflattenFromOne_nameFormatInconsistent(self):
#         self.back_unflatten_nameFormatInconsistent('feature')
#
#     def back_unflatten_nameFormatInconsistent(self, axis):
#         checkMsg = True
#         target = (axis + 's', 'unflattenFromOne')
#         vecShape = (1,4) if axis == 'point' else (4,1)
#         data = numpyRandom.rand(*vecShape)
#
#         # unflattend axis, mix of default names and correctly formatted
#         names = ["a | 1", "b | 1", "a | 2", "b | 2"]
#         args = {"featureNames":names} if axis == 'point' else {"pointNames":names}
#         testObj = self.constructor(data, **args)
#         if axis == 'point':
#             testObj.features.setName(1, None)
#         else:
#             testObj.points.setName(1, None)
#         exceptionHelper(testObj, target, [2], ImproperObjectAction, checkMsg)
#
#         # unflattened axis, inconsistent along original unflattened axis
#         names = ["a | 1", "b | 1", "a | 2", "c | 2"]
#         args = {"featureNames":names} if axis == 'point' else {"pointNames":names}
#         testObj = self.constructor(data, **args)
#         exceptionHelper(testObj, target, [2], ImproperObjectAction, checkMsg)
#
#         # unflattened axis, inconsistent along original flattened axis
#         names = ["a | 1", "b | 2", "a | 2", "b | 3"]
#         args = {"featureNames":names} if axis == 'point' else {"pointNames":names}
#         testObj = self.constructor(data, **args)
#         exceptionHelper(testObj, target, [2], ImproperObjectAction, checkMsg)
#
#
#     # unflatten something that was flattened - include name transformation
#     def test_points_unflattenFromOne_handmadeWithNames(self):
#         raw = [["el0", "el1", "el2", "el3", "el4", "el5"]]
#         rawNames = ["1 | A", "2 | A", "3 | A", "1 | B", "2 | B", "3 | B"]
#         toTest = self.constructor(raw, pointNames=["Flattened"], featureNames=rawNames)
#         expData = numpy.array([["el0", "el1", "el2"], ["el3", "el4", "el5"]])
#         namesP = ["A", "B"]
#         namesF = ["1", "2", "3"]
#
#         exp = self.constructor(expData, pointNames=namesP, featureNames=namesF)
#
#         toTest.points.unflattenFromOne(2)
#         assert toTest == exp
#
#     # unflatten something that was flattend - include name transformation
#     def test_features_unflattenFromOne_handmadeWithNames(self):
#         raw = [["el0"], ["el1"], ["el2"], ["el3"], ["el4"], ["el5"]]
#         rawNames = ["1 | A", "2 | A", "3 | A", "1 | B", "2 | B", "3 | B"]
#         toTest = self.constructor(raw, pointNames=rawNames, featureNames=["Flattened"])
#         expData = [["el0", "el3"],["el1", "el4"], ["el2", "el5"]]
#         namesP = ["1", "2", "3"]
#         namesF = ["A", "B"]
#
#         exp = self.constructor(expData, pointNames=namesP, featureNames=namesF)
#
#         toTest.features.unflattenFromOne(2)
#         assert toTest == exp
#
#
#     # unflatten something that is just a vector - default names
#     def test_points_unflattenFromOne_handmadeDefaultNames(self):
#         self.back_unflatten_handmadeDefaultNames('point')
#
#     def test_features_unflattenFromOne_handmadeDefaultNames(self):
#         self.back_unflatten_handmadeDefaultNames('feature')
#
#     def back_unflatten_handmadeDefaultNames(self, axis):
#         raw = [[1, 10, 20, 2]]
#         toTest = self.constructor(raw)
#         expData = numpy.array([[1,10],[20,2]])
#
#         if axis == 'point':
#             exp = self.constructor(expData)
#         else:
#             toTest.transpose()
#             exp = self.constructor(expData.T)
#
#         axisObj = getattr(toTest, axis + 's')
#         getattr(axisObj, 'unflattenFromOne')(2)
#         assert toTest == exp
#
#         # check that the name conforms to the standards of how nimble objects assign
#         # default names
#         def checkName(n):
#             assert n.startswith(DEFAULT_PREFIX)
#             assert int(n[len(DEFAULT_PREFIX):]) >= 0
#
#         list(map(checkName, toTest.points.getNames()))
#         list(map(checkName, toTest.features.getNames()))
#
#
#     # random round trip
#     def test_flatten_to_unflatten_point_roundTrip(self):
#         self.back_flatten_to_unflatten_roundTrip('point')
#
#     def test_flatten_to_unflatten_feature_roundTrip(self):
#         self.back_flatten_to_unflatten_roundTrip('feature')
#
#     def back_flatten_to_unflatten_roundTrip(self, axis):
#         discardAxisLen = 30
#         keptAxisLen = 50
#         shape = (discardAxisLen, keptAxisLen) if axis == 'point' else (keptAxisLen, discardAxisLen)
#         origRaw = numpyRandom.randint(0, 2, shape)  # array of ones and zeroes
#         namesDiscard = list(map(str, numpyRandom.choice(100, discardAxisLen, replace=False).tolist()))
#         namesKept = list(map(str, numpyRandom.choice(100, keptAxisLen, replace=False).tolist()))
#         namesArgs = {"pointNames":namesDiscard, "featureNames":namesKept} if axis == 'point' else {"pointNames":namesKept, "featureNames":namesDiscard}
#
#         testObj = self.constructor(origRaw, **namesArgs)
#         expObj = testObj.copy()
#
#         axisObj = getattr(testObj, axis + 's')
#         getattr(axisObj, 'flattenToOne')()
#         axisObj = getattr(testObj, axis + 's')
#         getattr(axisObj, 'unflattenFromOne')(discardAxisLen)
#         assert testObj == expObj
#
#         # second round to see if status of hidden internal variable are still viable
#         axisObj = getattr(testObj, axis + 's')
#         getattr(axisObj, 'flattenToOne')()
#         axisObj = getattr(testObj, axis + 's')
#         getattr(axisObj, 'unflattenFromOne')(discardAxisLen)
#         assert testObj == expObj

    ###########
    # merge() #
    ###########

    @raises(InvalidArgumentValue)
    def test_merge_exception_invalidPointString(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        dataR = [['a', 3, 4], ['b', 7, 8], ['c', -3, -4]]
        pNames = ['a', 'b', 'c']
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, pointNames=pNames, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNames, featureNames=fNamesR)
        leftObj.merge(rightObj, point='abc', feature='union')

    @raises(InvalidArgumentValue)
    def test_merge_exception_invalidFeatureString(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        dataR = [['a', 3, 4], ['b', 7, 8], ['c', -3, -4]]
        pNames = ['a', 'b', 'c']
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, pointNames=pNames, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNames, featureNames=fNamesR)
        leftObj.merge(rightObj, point='union', feature='abc')

    @raises(InvalidArgumentValueCombination)
    def test_merge_exception_pointStrictAndFeature(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        pNames = ['a', 'b', 'c']
        fNames = ['id', 'f1', 'f2']
        leftObj = self.constructor(dataL, pointNames=pNames, featureNames=fNames)
        rightObj = self.constructor(dataL, pointNames=pNames, featureNames=fNames)
        leftObj.merge(rightObj, point='strict', feature='strict')

    @raises(InvalidArgumentValue)
    def test_merge_exception_strictDifferentPointCount(self):
        dataL = [['a', 1, 2], ['c', 5, 6], ['c', -1, -2], ['d', 99, 99]]
        dataR = [['a', 3, 4], ['b', 7, 8], ['c', -3, -4]]
        pNames = ['a', 'b', 'c']
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, pointNames=pNames + ['d'], featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNames, featureNames=fNamesR)
        leftObj.merge(rightObj, point='strict', feature='union')

    @raises(InvalidArgumentValue)
    def test_merge_exception_strictDifferentFeatureCount(self):
        dataL = [['a', 1, 2, 99], ['c', 5, 6, 99], ['c', -1, -2, 99]]
        dataR = [['a', 3, 4], ['b', 7, 8], ['c', -3, -4]]
        pNames = ['a', 'b', 'c']
        fNamesL = ['id', 'f1', 'f2', 'f3']
        fNamesR = ['id', 'f1', 'f2']
        leftObj = self.constructor(dataL, pointNames=pNames, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNames, featureNames=fNamesR)
        leftObj.merge(rightObj, point='union', feature='strict')

    @raises(InvalidArgumentValue)
    def test_merge_exception_pointStrictNameMismatch(self):
        dataL = [['a', 1, 2], ['c', 5, 6], ['c', -1, -2]]
        dataR = [['a', 3, 4], ['b', 7, 8], ['c', -3, -4]]
        pNamesL = ['a', 'b', 'c']
        pNamesR = ['a', 'b', 'z']
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)
        leftObj.merge(rightObj, point='strict', feature='union')

    @raises(InvalidArgumentValue)
    def test_merge_exception_featureStrictNameMismatch(self):
        dataL = [['a', 1, 2], ['c', 5, 6], ['c', -1, -2]]
        dataR = [['a', 3, 4], ['b', 7, 8], ['c', -3, -4]]
        pNamesL = ['a', 'b', 'c']
        pNamesR = ['a', 'b', 'c']
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f1', 'f99']
        leftObj = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)
        leftObj.merge(rightObj, point='union', feature='strict')

    @raises(InvalidArgumentValue)
    def test_merge_exception_pointStrictMissingOnFeature(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        dataR = [['a', 3, 4], ['b', 7, 8], ['c', -3, -4]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['di', 'f3', 'f4' ]
        pNames = ['a', 'b', 'c']
        leftObj = self.constructor(dataL, pointNames=pNames, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNames, featureNames=fNamesR)
        leftObj.merge(rightObj, point='strict', feature='union', onFeature='id')

    @raises(InvalidArgumentValueCombination)
    def test_merge_exception_pointStrictOnFeatureNotUnique(self):
        dataL = [['a', 1, 2], ['c', 5, 6], ['c', -1, -2]]
        dataR = [['a', 3, 4], ['b', 7, 8], ['c', -3, -4]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        pNamesL = ['a', 'b', 'c']
        pNamesR = ['a', 'b', 'c']
        leftObj = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)
        leftObj.merge(rightObj, point='strict', feature='union', onFeature='id')

    @raises(InvalidArgumentValueCombination)
    def test_merge_exception_pointStrictOnFeatureNotMatching(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        dataR = [['a', 3, 4], ['b', 7, 8], ['d', -3, -4]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        pNamesL = ['a', 'b', 'c']
        pNamesR = ['a', 'b', 'c']
        leftObj = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)
        leftObj.merge(rightObj, point='strict', feature='union', onFeature='id')

    @raises(InvalidArgumentValueCombination)
    def test_merge_exception_pointIntersectionNoPointNames(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        dataR = [['a', 3, 4], ['b', 7, 8], ['d', -3, -4]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        pNamesR = ['a', 'b', 'c']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)
        leftObj.merge(rightObj, point='intersection', feature='union')

    @raises(InvalidArgumentValueCombination)
    def test_merge_exception_featureIntersectionNoFeatureNames(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        dataR = [['a', 3, 4], ['b', 7, 8], ['d', -3, -4]]
        fNamesR = ['id', 'f1', 'f2']
        pNamesL = ['a', 'b', 'c']
        pNamesR = ['a', 'b', 'c']
        leftObj = self.constructor(dataL, pointNames=pNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)
        leftObj.merge(rightObj, point='union', feature='intersection')

    @raises(InvalidArgumentValue)
    def test_merge_exception_onFeaturebothNonUnique(self):
        dataL = [['a', 1, 2], ['c', 5, 6], ['c', -1, -2]]
        dataR = [['a', 3, 4], ['c', 7, 8], ['c', -3, -4]]
        pNames = ['a', 'b', 'c']
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, pointNames=pNames, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNames, featureNames=fNamesR)
        leftObj.merge(rightObj, point='union', feature='union', onFeature='id')

    def merge_backend(self, left, right, expected, on=None, includeStrict=False):
        combinations = [
            ('union', 'union'), ('union', 'intersection'), ('union', 'left'),
            ('intersection', 'union'), ('intersection', 'intersection'), ('intersection', 'left'),
            ('left', 'union'), ('left', 'intersection')
            ]
        if includeStrict:
            comboStrict = [
                ('strict', 'union'), ('strict', 'intersection'),
                ('union', 'strict'), ('intersection', 'strict')
                ]
            combinations += comboStrict

        @oneLogEntryExpected
        def performMerge():
            test.merge(tRight, point=pt, feature=ft, onFeature=on)

        for i, exp in enumerate(expected):
            pt = combinations[i][0]
            ft = combinations[i][1]
            try:
                test = left.copy()
                tRight = right.copy()
                performMerge()
                assert test == exp
            except InvalidArgumentValue:
                assert exp is InvalidArgumentValue
            except InvalidArgumentValueCombination:
                assert exp is InvalidArgumentValueCombination


        ##################
        # on point names #
        ##################

    def test_merge_onPtNames_samePointNames_sameFeatureNames(self):
        dataL = [["a", 1], ["b", 2], ["c", 3]]
        fNamesL = ["f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["a", 1], ["b", 2], ["c", 3]]
        fNamesR = ["f1", "f2"]
        pNamesR = ["p1", "p2", "p3"]
        right = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)

        pUnion_fUnion = right
        pUnion_fIntersection = right
        pUnion_fLeft = right
        pIntersection_fUnion = right
        pIntersection_fIntersection = right
        pIntersection_fLeft = right
        pLeft_fUnion = right
        pLeft_fIntersection = right
        pStrict_fUnion = right
        pStrict_fIntersection = right
        pUnion_fStrict = right
        pIntersection_fStrict = right

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
            pStrict_fUnion, pStrict_fIntersection,
            pUnion_fStrict, pIntersection_fStrict
        ]

        self.merge_backend(left, right, expected, includeStrict=True)

    def test_merge_onPtNames_samePointNames_newFeatureNames(self):
        dataL = [["a", 1], ["b", 2], ["c", 3]]
        fNamesL = ["f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["d", 4], ["e", 5], ["f", 6]]
        fNamesR = ["f3", "f4"]
        pNamesR = ["p1", "p2", "p3"]
        right = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)

        mData = [["a", 1, "d", 4], ["b", 2, "e", 5], ["c", 3, "f", 6]]
        mFtNames = ["f1", "f2", "f3", "f4"]
        mPtNames = ["p1", "p2", "p3"]
        mLargest = self.constructor(mData, pointNames=mPtNames, featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest[:,[]]
        pUnion_fLeft = mLargest[:, ["f1", "f2"]]
        pIntersection_fUnion = mLargest
        pIntersection_fIntersection = mLargest[:,[]]
        pIntersection_fLeft = mLargest[:, ["f1", "f2"]]
        pLeft_fUnion = mLargest
        pLeft_fIntersection = mLargest[:,[]]
        pStrict_fUnion = mLargest
        pStrict_fIntersection = mLargest[:,[]]
        pUnion_fStrict = InvalidArgumentValue
        pIntersection_fStrict = InvalidArgumentValue

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
            pStrict_fUnion, pStrict_fIntersection,
            pUnion_fStrict, pIntersection_fStrict
        ]

        self.merge_backend(left, right, expected, includeStrict=True)

    def test_merge_onPtNames_newPointNames_sameFeatureNames(self):
        dataL = [["a", 1], ["b", 2], ["c", 3]]
        fNamesL = ["f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["d", 4], ["e", 5], ["f", 6]]
        fNamesR = ["f1", "f2"]
        pNamesR = ["p4", "p5", "p6"]
        right = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)

        mData = [["a", 1], ["b", 2], ["c", 3], ["d", 4], ["e", 5], ["f", 6]]
        mFtNames = ["f1", "f2"]
        mPtNames = ["p1", "p2", "p3", "p4", "p5", "p6"]
        mLargest = self.constructor(mData, pointNames=mPtNames, featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest
        pUnion_fLeft = mLargest[:, ["f1", "f2"]]
        pIntersection_fUnion = mLargest[[], :]
        pIntersection_fIntersection = mLargest[[], :]
        pIntersection_fLeft = mLargest[[], ["f1", "f2"]]
        pLeft_fUnion = mLargest[["p1", "p2", "p3"], :]
        pLeft_fIntersection = mLargest[["p1", "p2", "p3"],["f1", "f2"]]
        pStrict_fUnion = InvalidArgumentValue
        pStrict_fIntersection = InvalidArgumentValue
        pUnion_fStrict = mLargest
        pIntersection_fStrict = mLargest[[], :]

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
            pStrict_fUnion, pStrict_fIntersection,
            pUnion_fStrict, pIntersection_fStrict
        ]

        self.merge_backend(left, right, expected, includeStrict=True)

    def test_merge_onPtNames_noPointNames_newFeatureNames(self):
        dataL = [["a", 1], ["b", 2], ["c", 3]]
        fNamesL = ["f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["d", 4], ["e", 5], ["f", 6]]
        fNamesR = ["f3", "f4"]
        right = self.constructor(dataR, featureNames=fNamesR)

        mFtNames = ["f1", "f2", "f3", "f4"]
        mData = [["a", 1, None, None], ["b", 2, None, None], ["c", 3, None, None],
                 [None, None, "d", 4], [None, None, "e", 5], [None, None, "f", 6]]
        mLargest = self.constructor(mData, featureNames=mFtNames)
        mLargest.points.setName(0, "p1")
        mLargest.points.setName(1, "p2")
        mLargest.points.setName(2, "p3")

        mDataStrict = [["a", 1, "d", 4], ["b", 2, "e", 5], ["c", 3, "f", 6]]
        mLargestStrict = self.constructor(mDataStrict, pointNames=["p1", "p2", "p3"], featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest[:, []]
        pUnion_fLeft = mLargest[:, ["f1", "f2"]]
        pIntersection_fUnion = InvalidArgumentValueCombination
        pIntersection_fIntersection = InvalidArgumentValueCombination
        pIntersection_fLeft = InvalidArgumentValueCombination
        pLeft_fUnion = mLargest[["p1", "p2", "p3"], :]
        pLeft_fIntersection = mLargest[["p1", "p2", "p3"],[]]
        pStrict_fUnion = mLargestStrict
        pStrict_fIntersection = mLargestStrict[:, []]
        pUnion_fStrict = InvalidArgumentValue
        pIntersection_fStrict = InvalidArgumentValue

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
            pStrict_fUnion, pStrict_fIntersection,
            pUnion_fStrict, pIntersection_fStrict
        ]

        self.merge_backend(left, right, expected, includeStrict=True)

    def test_merge_onPtNames_newPointNames_noFeatureNames(self):
        dataL = [["a", 1], ["b", 2], ["c", 3]]
        fNamesL = ["f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["d", 4], ["e", 5], ["f", 6]]
        pNamesR = ["p4", "p5", "p6"]
        right = self.constructor(dataR, pointNames=pNamesR)

        mPointNames = ["p1", "p2", "p3", "p4", "p5", "p6"]
        mData = [["a", 1, None, None], ["b", 2, None, None], ["c", 3, None, None],
                 [None, None, "d", 4], [None, None, "e", 5], [None, None, "f", 6]]
        mLargest = self.constructor(mData, pointNames=mPointNames)
        mLargest.features.setName(0, "f1")
        mLargest.features.setName(1, "f2")

        mDataStrict = [["a", 1], ["b", 2], ["c", 3], ["d", 4], ["e", 5], ["f", 6]]
        mLargestStrict = self.constructor(mDataStrict, pointNames=mPointNames, featureNames=["f1", "f2"])

        pUnion_fUnion = mLargest
        pUnion_fIntersection = InvalidArgumentValueCombination
        pUnion_fLeft = mLargest[:, ["f1", "f2"]]
        pIntersection_fUnion = mLargest[[], :]
        pIntersection_fIntersection = InvalidArgumentValueCombination
        pIntersection_fLeft = mLargest[[], ["f1", "f2"]]
        pLeft_fUnion = mLargest[["p1", "p2", "p3"], :]
        pLeft_fIntersection = InvalidArgumentValueCombination
        pStrict_fUnion = InvalidArgumentValue
        pStrict_fIntersection = InvalidArgumentValue
        pUnion_fStrict = mLargestStrict
        pIntersection_fStrict = mLargestStrict[[], :]

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
            pStrict_fUnion, pStrict_fIntersection,
            pUnion_fStrict, pIntersection_fStrict
        ]

        self.merge_backend(left, right, expected, includeStrict=True)

    def test_merge_onPtNames_noPointNames_noFeatureNames(self):
        dataL = [["a", 1], ["b", 2], ["c", 3]]
        fNamesL = ["f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["d", 4], ["e", 5], ["f", 6]]
        right = self.constructor(dataR)

        mPointNames = ["p1", "p2", "p3", "p4", "p5", "p6"]
        mData = [["a", 1, None, None], ["b", 2, None, None], ["c", 3, None, None],
                 [None, None, "d", 4], [None, None, "e", 5], [None, None, "f", 6]]
        mLargest = self.constructor(mData)
        mLargest.features.setName(0, "f1")
        mLargest.features.setName(1, "f2")

        mDataStrictFts = [["a", 1], ["b", 2], ["c", 3], ["d", 4], ["e", 5], ["f", 6]]
        mStrictFts = self.constructor(mDataStrictFts)

        mDataStrictPts = [["a", 1, "d", 4], ["b", 2, "e", 5], ["c", 3, "f", 6]]
        mStrictPts = self.constructor(mDataStrictPts)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = InvalidArgumentValueCombination
        pUnion_fLeft = mLargest[:, ["f1", "f2"]]
        pIntersection_fUnion = InvalidArgumentValueCombination
        pIntersection_fIntersection = InvalidArgumentValueCombination
        pIntersection_fLeft = InvalidArgumentValueCombination
        # TODO case below?
        pLeft_fUnion = mLargest[[0,1,2], :]
        pLeft_fIntersection = InvalidArgumentValueCombination
        pStrict_fUnion = mStrictPts
        pStrict_fIntersection = InvalidArgumentValue
        pUnion_fStrict = mStrictFts
        pIntersection_fStrict = InvalidArgumentValue

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
            pStrict_fUnion, pStrict_fIntersection,
            pUnion_fStrict, pIntersection_fStrict
        ]

    def test_merge_onPtNames_samePointNames_noFeatureNames(self):
        dataL = [["a", 1], ["b", 2], ["c", 3]]
        fNamesL = ["f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["d", 4], ["e", 5], ["f", 6]]
        pNamesR = ["p1", "p2", "p3"]
        right = self.constructor(dataR, pointNames=pNamesR)

        mData = [["a", 1, "d", 4], ["b", 2, "e", 5], ["c", 3, "f", 6]]
        mLargest = self.constructor(mData, pointNames=pNamesR)
        mLargest.features.setName(0, "f1")
        mLargest.features.setName(1, "f2")

        pUnion_fUnion = mLargest
        pUnion_fIntersection = InvalidArgumentValueCombination
        pUnion_fLeft = mLargest[:, ["f1", "f2"]]
        pIntersection_fUnion = mLargest
        pIntersection_fIntersection = InvalidArgumentValueCombination
        pIntersection_fLeft = mLargest[:, ["f1", "f2"]]
        pLeft_fUnion = mLargest[["p1", "p2", "p3"], :]
        pLeft_fIntersection = InvalidArgumentValueCombination
        pStrict_fUnion = mLargest
        pStrict_fIntersection = InvalidArgumentValueCombination
        pUnion_fStrict = InvalidArgumentValue
        pIntersection_fStrict = InvalidArgumentValue

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
            pStrict_fUnion, pStrict_fIntersection,
            pUnion_fStrict, pIntersection_fStrict
        ]

        self.merge_backend(left, right, expected, includeStrict=True)

    def test_merge_onPtNames_noPointNames_sameFeatureNames(self):
        dataL = [["a", 1], ["b", 2], ["c", 3]]
        fNamesL = ["f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["d", 4], ["e", 5], ["f", 6]]
        fNamesR = ["f1", "f2"]
        right = self.constructor(dataR, featureNames=fNamesR)

        mData = [["a", 1], ["b", 2], ["c", 3], ["d", 4], ["e", 5], ["f", 6]]
        mLargest = self.constructor(mData, featureNames=fNamesR)
        mLargest.points.setName(0, "p1")
        mLargest.points.setName(1, "p2")
        mLargest.points.setName(2, "p3")

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest
        pUnion_fLeft = mLargest[:, ["f1", "f2"]]
        pIntersection_fUnion = InvalidArgumentValueCombination
        pIntersection_fIntersection = InvalidArgumentValueCombination
        pIntersection_fLeft = InvalidArgumentValueCombination
        pLeft_fUnion = mLargest[["p1", "p2", "p3"], :]
        pLeft_fIntersection = mLargest[["p1", "p2", "p3"], ["f1", "f2"]]
        pStrict_fUnion = InvalidArgumentValue
        pStrict_fIntersection = InvalidArgumentValue
        pUnion_fStrict = mLargest
        pIntersection_fStrict = InvalidArgumentValueCombination

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
            pStrict_fUnion, pStrict_fIntersection,
            pUnion_fStrict, pIntersection_fStrict
        ]

        self.merge_backend(left, right, expected, includeStrict=True)

        # no strict

    def test_merge_onPtNames_newPointNames_newFeatureNames(self):
        dataL = [["a", 1], ["b", 2], ["c", 3]]
        fNamesL = ["f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["d", 4], ["e", 5], ["f", 6]]
        fNamesR = ["f3", "f4"]
        pNamesR = ["p4", "p5", "p6"]
        right = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)

        mData = [["a", 1, None, None], ["b", 2, None, None], ["c", 3, None, None],
                 [None, None, "d", 4], [None, None, "e", 5], [None, None, "f", 6]]
        mFtNames = ["f1", "f2", "f3", "f4"]
        mPtNames = ["p1", "p2", "p3", "p4", "p5", "p6"]
        mLargest = self.constructor(mData, pointNames=mPtNames, featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest[:, []]
        pUnion_fLeft = mLargest[:, ["f1", "f2"]]
        pIntersection_fUnion = mLargest[[], :]
        pIntersection_fIntersection = mLargest[[], []]
        pIntersection_fLeft = mLargest[[], ["f1", "f2"]]
        pLeft_fUnion = mLargest[["p1", "p2", "p3"], :]
        pLeft_fIntersection = mLargest[["p1", "p2", "p3"], []]

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection
        ]

        self.merge_backend(left, right, expected)

    def test_merge_onPtNames_sharedPointNames_sameFeatureNames_match(self):
        dataL = [["a", 1], ["b", 2], ["c", 3]]
        fNamesL = ["f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["c", 3], ["d", 4], ["e", 5]]
        fNamesR = ["f1", "f2"]
        pNamesR = ["p3", "p4", "p5"]
        right = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)

        mData = [["a", 1], ["b", 2], ["c", 3], ["d", 4], ["e", 5]]
        mFtNames = ["f1", "f2"]
        mPtNames = ["p1", "p2", "p3", "p4", "p5"]
        mLargest = self.constructor(mData, pointNames=mPtNames, featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest
        pUnion_fLeft = mLargest[:, ["f1", "f2"]]
        pIntersection_fUnion = mLargest["p3", :]
        pIntersection_fIntersection = mLargest["p3", :]
        pIntersection_fLeft = mLargest["p3", ["f1", "f2"]]
        pLeft_fUnion = mLargest[["p1", "p2", "p3"], :]
        pLeft_fIntersection = mLargest[["p1", "p2", "p3"], ["f1", "f2"]]

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection
        ]

        self.merge_backend(left, right, expected)

    def test_merge_onPtNames_sharedPointNames_sameFeatureNames_mismatch(self):
        dataL = [["a", 1], ["b", 2], ["c", 3]]
        fNamesL = ["f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["d", 4], ["e", 5], ["f", 6]]
        fNamesR = ["f1", "f2"]
        pNamesR = ["p3", "p4", "p5"]
        right = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)

        pUnion_fUnion = InvalidArgumentValue
        pUnion_fIntersection = InvalidArgumentValue
        pUnion_fLeft = InvalidArgumentValue
        pIntersection_fUnion = InvalidArgumentValue
        pIntersection_fIntersection = InvalidArgumentValue
        pIntersection_fLeft = InvalidArgumentValue
        pLeft_fUnion = InvalidArgumentValue
        pLeft_fIntersection = InvalidArgumentValue

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection
        ]

        self.merge_backend(left, right, expected)

    def test_merge_onPtNames_samePointNames_sharedFeatureNames_match(self):
        dataL = [["a", 1], ["b", 2], ["c", 3]]
        fNamesL = ["f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["a", 9], ["b", 8], ["c", 7]]
        fNamesR = ["f1", "f3"]
        pNamesR = ["p1", "p2", "p3"]
        right = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)

        mData = [["a", 1, 9], ["b", 2, 8], ["c", 3, 7]]
        mFtNames = ["f1", "f2", "f3"]
        mPtNames = ["p1", "p2", "p3"]
        mLargest = self.constructor(mData, pointNames=mPtNames, featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest[:, "f1"]
        pUnion_fLeft = mLargest[:, ["f1", "f2"]]
        pIntersection_fUnion = mLargest
        pIntersection_fIntersection = mLargest[:, "f1"]
        pIntersection_fLeft = mLargest[["p1", "p2", "p3"], ["f1", "f2"]]
        pLeft_fUnion = mLargest[["p1", "p2", "p3"], :]
        pLeft_fIntersection = mLargest[["p1", "p2", "p3"], ["f1"]]

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection
        ]

        self.merge_backend(left, right, expected)

    def test_merge_onPtNames_samePointNames_sharedFeatureNames_mismatch(self):
        dataL = [["a", 1], ["b", 2], ["c", 3]]
        fNamesL = ["f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["d", 9], ["e", 8], ["f", 7]]
        fNamesR = ["f1", "f3"]
        pNamesR = ["p1", "p2", "p3"]
        right = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)

        pUnion_fUnion = InvalidArgumentValue
        pUnion_fIntersection = InvalidArgumentValue
        pUnion_fLeft = InvalidArgumentValue
        pIntersection_fUnion = InvalidArgumentValue
        pIntersection_fIntersection = InvalidArgumentValue
        pIntersection_fLeft = InvalidArgumentValue
        pLeft_fUnion = InvalidArgumentValue
        pLeft_fIntersection = InvalidArgumentValue

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection
        ]

        self.merge_backend(left, right, expected)

    def test_merge_onPtNames_sharedPointNames_sharedFeatureNames_noConflict(self):
        dataL = [["a", 1], ["b", 2], ["c", 3]]
        fNamesL = ["f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["c", 9], ["d", 8], ["e", 7]]
        fNamesR = ["f1", "f3"]
        pNamesR = ["p3", "p4", "p5"]
        right = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)

        mData = [["a", 1, None], ["b", 2, None], ["c", 3, 9],
                 ["d", None, 8], ["e", None, 7]]
        mFtNames = ["f1", "f2", "f3"]
        mPtNames = ["p1", "p2", "p3", "p4", "p5"]
        mLargest = self.constructor(mData, pointNames=mPtNames, featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest[:, "f1"]
        pUnion_fLeft = mLargest[:, ["f1", "f2"]]
        pIntersection_fUnion = mLargest["p3", :]
        pIntersection_fIntersection = mLargest[["p3"], ["f1"]]
        pIntersection_fLeft = mLargest[["p3"], ["f1", "f2"]]
        pLeft_fUnion = mLargest[["p1", "p2", "p3"], :]
        pLeft_fIntersection = mLargest[["p1", "p2", "p3"], ["f1"]]

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection
        ]

        self.merge_backend(left, right, expected)

    def test_merge_onPtNames_newPointNames_subsetFeatureNames(self):
        dataL = [["a", 1], ["b", 2], ["c", 3]]
        fNamesL = ["f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["d"], ["e"], ["f"]]
        fNamesR = ["f1"]
        pNamesR = ["p4", "p5", "p6"]
        right = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)

        mData = [["a", 1], ["b", 2], ["c", 3], ["d", None], ["e", None], ["f", None]]
        mFtNames = ["f1", "f2"]
        mPtNames = ["p1", "p2", "p3", "p4", "p5", "p6"]
        mLargest = self.constructor(mData, pointNames=mPtNames, featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest[:, "f1"]
        pUnion_fLeft = mLargest[:, ["f1", "f2"]]
        pIntersection_fUnion = mLargest[[], :]
        pIntersection_fIntersection = mLargest[[], "f1"]
        pIntersection_fLeft = mLargest[[], ["f1", "f2"]]
        pLeft_fUnion = mLargest[["p1", "p2", "p3"], :]
        pLeft_fIntersection = mLargest[["p1", "p2", "p3"], ["f1"]]

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection
        ]

        self.merge_backend(left, right, expected)

    def test_merge_onPtNames_subsetPointNames_sharedFeatureNames(self):
        dataL = [["a", 1], ["b", 2], ["c", 3]]
        fNamesL = ["f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["b", 6], ["d", 8], ["e", 9]]
        fNamesR = ["f1", "f3"]
        pNamesR = ["p2", "p4", "p5"]
        right = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)

        mData = [["a", 1, None], ["b", 2, 6], ["c", 3, None], ["d", None, 8], ["e", None, 9]]
        mFtNames = ["f1", "f2", "f3"]
        mPtNames = ["p1", "p2", "p3", "p4", "p5"]
        mLargest = self.constructor(mData, pointNames=mPtNames, featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest[:, "f1"]
        pUnion_fLeft = mLargest[:, ["f1", "f2"]]
        pIntersection_fUnion = mLargest["p2", :]
        pIntersection_fIntersection = mLargest[["p2"], ["f1"]]
        pIntersection_fLeft = mLargest["p2", ["f1", "f2"]]
        pLeft_fUnion = mLargest[["p1", "p2", "p3"], :]
        pLeft_fIntersection = mLargest[["p1", "p2", "p3"], ["f1"]]

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection
        ]

        self.merge_backend(left, right, expected)

    def test_merge_onPtNames_sharedPointNames_noFeatureNames(self):
        dataL = [["a", 1], ["b", 2], ["c", 3]]
        fNamesL = ["f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["c", 3], ["d", 4], ["e", 5]]
        pNamesR = ["p3", "p4", "p5"]
        right = self.constructor(dataR, pointNames=pNamesR)

        mData = [["a", 1, None, None], ["b", 2, None, None], ["c", 3, "c", 3],
                 [None, None, "d", 4], [None, None, "e", 5]]
        mLargest = self.constructor(mData, pointNames=["p1", "p2", "p3", "p4", "p5"])
        mLargest.features.setName(0, "f1")
        mLargest.features.setName(1, "f2")

        pUnion_fUnion = mLargest
        pUnion_fIntersection = InvalidArgumentValueCombination
        pUnion_fLeft = mLargest[:, ["f1", "f2"]]
        pIntersection_fUnion = mLargest["p3", :]
        pIntersection_fIntersection = InvalidArgumentValueCombination
        pIntersection_fLeft = mLargest["p3", ["f1", "f2"]]
        pLeft_fUnion = mLargest[["p1", "p2", "p3"], :]
        pLeft_fIntersection = InvalidArgumentValueCombination

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection
        ]

        self.merge_backend(left, right, expected)

    def test_merge_onPtNames_noPointNames_sharedFeatureNames(self):
        dataL = [["a", 1], ["b", 2], ["c", 3]]
        fNamesL = ["f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["a", 4], ["b", 5], ["c", 6]]
        fNamesR = ["f1", "f3"]
        right = self.constructor(dataR, featureNames=fNamesR)

        mData = [["a", 1, None], ["b", 2, None], ["c", 3, None],
                 ["a", None, 4], ["b", None, 5], ["c", None, 6]]
        mLargest = self.constructor(mData, featureNames=["f1", "f2", "f3"])
        mLargest.points.setName(0, "p1")
        mLargest.points.setName(1, "p2")
        mLargest.points.setName(2, "p3")

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest[:, "f1"]
        pUnion_fLeft = mLargest[:, ["f1", "f2"]]
        pIntersection_fUnion = InvalidArgumentValueCombination
        pIntersection_fIntersection = InvalidArgumentValueCombination
        pIntersection_fLeft = InvalidArgumentValueCombination
        pLeft_fUnion = mLargest[["p1", "p2", "p3"], :]
        pLeft_fIntersection = mLargest[["p1", "p2", "p3"], "f1"]

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection
        ]

        self.merge_backend(left, right, expected)
    #TODO no point names/ shared feature names, Feature Match

        #############
        # onFeature #
        #############

    def test_merge_onFeature_sameIds_newFeatures_nonDuplicate(self):
        dataL = [["id1", "a", 1], ["id2", "b", 2], ["id3", "c", 3]]
        fNamesL = ["id", "f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["id1", "x", 9], ["id2", "y", 8], ["id3", "z", 9]]
        fNamesR = ["id", "f3", "f4"]
        right = self.constructor(dataR, featureNames=fNamesR)

        mData = [["id1", "a", 1, "x", 9], ["id2", "b", 2, "y", 8], ["id3", "c", 3, "z", 9]]
        mFtNames = ["id", "f1", "f2", "f3", "f4"]
        mLargest = self.constructor(mData, featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest[:, "id"]
        pUnion_fLeft = mLargest[:, ["id", "f1", "f2"]]
        pIntersection_fUnion = mLargest
        pIntersection_fIntersection = mLargest[:, "id"]
        pIntersection_fLeft = mLargest[:, ["id", "f1", "f2"]]
        pLeft_fUnion = mLargest
        pLeft_fIntersection = mLargest[:, ["id"]]
        # when point is strict, can return point names
        mLargestStrict = mLargest.copy()
        mLargestStrict.points.setNames(["p1", "p2", "p3"])
        pStrict_fUnion = mLargestStrict
        pStrict_fIntersection = mLargestStrict[:, "id"]
        pUnion_fStrict = InvalidArgumentValue
        pIntersection_fStrict = InvalidArgumentValue

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
            pStrict_fUnion, pStrict_fIntersection,
            pUnion_fStrict, pIntersection_fStrict
        ]

        self.merge_backend(left, right, expected, on="id", includeStrict=True)

    def test_merge_onFeature_newIds_sameFeatures_nonDuplicate(self):
        dataL = [["id1", "a", 1], ["id2", "b", 2], ["id3", "c", 3]]
        fNamesL = ["id", "f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["id4", "d", 4], ["id5", "e", 5], ["id6", "f", 6]]
        fNamesR = ["id", "f1", "f2"]
        right = self.constructor(dataR, featureNames=fNamesR)

        mData = [["id1", "a", 1], ["id2", "b", 2], ["id3", "c", 3],
                 ["id4", "d", 4], ["id5", "e", 5], ["id6", "f", 6]]
        mFtNames = ["id", "f1", "f2"]
        mLargest = self.constructor(mData, featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest
        pUnion_fLeft = mLargest
        pIntersection_fUnion = mLargest[[], :]
        pIntersection_fIntersection = mLargest[[], :]
        pIntersection_fLeft = mLargest[[], :]
        pLeft_fUnion = mLargest[[0,1,2], :]
        pLeft_fIntersection = mLargest[[0,1,2], :]
        pStrict_fUnion = InvalidArgumentValueCombination
        pStrict_fIntersection = InvalidArgumentValueCombination
        pUnion_fStrict = mLargest
        pIntersection_fStrict = mLargest[[], :]

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
            pStrict_fUnion, pStrict_fIntersection,
            pUnion_fStrict, pIntersection_fStrict
        ]

        self.merge_backend(left, right, expected, on="id", includeStrict=True)

        # no strict

    def test_merge_onFeature_newIds_newFeatures_nonDuplicate(self):
        dataL = [["id1", "a", 1], ["id2", "b", 2], ["id3", "c", 3]]
        fNamesL = ["id", "f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["id4", "x", 9], ["id5", "y", 8], ["id6", "z", 7]]
        fNamesR = ["id", "f3", "f4"]
        right = self.constructor(dataR, featureNames=fNamesR)

        mData = [["id1", "a", 1, None, None], ["id2", "b", 2, None, None], ["id3", "c", 3, None, None],
                 ["id4", None, None, "x", 9], ["id5", None, None, "y", 8], ["id6", None, None, "z", 7]]
        mFtNames = ["id", "f1", "f2", "f3", "f4"]
        mLargest = self.constructor(mData, featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest[:, "id"]
        pUnion_fLeft = mLargest[:, ["id", "f1", "f2"]]
        pIntersection_fUnion = mLargest[[], :]
        pIntersection_fIntersection = mLargest[[], "id"]
        pIntersection_fLeft = mLargest[[], ["id", "f1", "f2"]]
        pLeft_fUnion = mLargest[[0,1,2], :]
        pLeft_fIntersection = mLargest[[0,1,2], "id"]

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
        ]

        self.merge_backend(left, right, expected, on="id")

    def test_merge_onFeature_sharedIds_newFeatures_nonDuplicate(self):
        dataL = [["id1", "a", 1], ["id2", "b", 2], ["id3", "c", 3]]
        fNamesL = ["id", "f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["id2", "x", 9], ["id3", "y", 8], ["id4", "z", 7]]
        fNamesR = ["id", "f3", "f4"]
        right = self.constructor(dataR, featureNames=fNamesR)

        mData = [["id1", "a", 1, None, None], ["id2", "b", 2, "x", 9], ["id3", "c", 3, "y", 8],
                 ["id4", None, None, "z", 7]]
        mFtNames = ["id", "f1", "f2", "f3", "f4"]
        mLargest = self.constructor(mData, featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest[:, "id"]
        pUnion_fLeft = mLargest[:, ["id", "f1", "f2"]]
        pIntersection_fUnion = mLargest[1:2, :]
        pIntersection_fIntersection = mLargest[1:2, "id"]
        pIntersection_fLeft = mLargest[1:2, ["id", "f1", "f2"]]
        pLeft_fUnion = mLargest[[0,1,2], :]
        pLeft_fIntersection = mLargest[[0,1,2], "id"]

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
        ]

        self.merge_backend(left, right, expected, on="id")

    def test_merge_onFeature_newIds_sharedFeatures_nonDuplicate(self):
        dataL = [["id1", "a", 1], ["id2", "b", 2], ["id3", "c", 3]]
        fNamesL = ["id", "f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["id4", 4, "x"], ["id5", 5, "y"], ["id6", 6, "z"]]
        fNamesR = ["id", "f2", "f3"]
        right = self.constructor(dataR, featureNames=fNamesR)

        mData = [["id1", "a", 1, None], ["id2", "b", 2, None], ["id3", "c", 3, None],
                 ["id4", None, 4, "x"], ["id5", None, 5, "y"], ["id6", None, 6, "z"]]
        mFtNames = ["id", "f1", "f2", "f3"]
        mLargest = self.constructor(mData, featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest[:, ["id", "f2"]]
        pUnion_fLeft = mLargest[:, ["id", "f1", "f2"]]
        pIntersection_fUnion = mLargest[[], :]
        pIntersection_fIntersection = mLargest[[], ["id", "f2"]]
        pIntersection_fLeft = mLargest[[], ["id", "f1", "f2"]]
        pLeft_fUnion = mLargest[[0,1,2], :]
        pLeft_fIntersection = mLargest[[0,1,2], ["id", "f2"]]

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
        ]

        self.merge_backend(left, right, expected, on="id")

    def test_merge_onFeature_sharedIds_sharedFeatures_nonDuplicate_noConflict(self):
        dataL = [["id1", "a", 1], ["id2", "b", 2], ["id3", "c", 3]]
        fNamesL = ["id", "f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["id1", 1, "x"], ["id3", 3, "y"], ["id4", 4, "z"]]
        fNamesR = ["id", "f2", "f3"]
        right = self.constructor(dataR, featureNames=fNamesR)

        mData = [["id1", "a", 1, "x"], ["id2", "b", 2, None], ["id3", "c", 3, "y"],
                 ["id4", None, 4, "z"]]
        mFtNames = ["id", "f1", "f2", "f3"]
        mLargest = self.constructor(mData, featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest[:, ["id", "f2"]]
        pUnion_fLeft = mLargest[:, ["id", "f1", "f2"]]
        pIntersection_fUnion = mLargest[[0, 2], :]
        pIntersection_fIntersection = mLargest[[0, 2], ["id", "f2"]]
        pIntersection_fLeft = mLargest[[0, 2], ["id", "f1", "f2"]]
        pLeft_fUnion = mLargest[[0,1,2], :]
        pLeft_fIntersection = mLargest[[0,1,2], ["id", "f2"]]

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
        ]

        self.merge_backend(left, right, expected, on="id")

    def test_merge_onFeature_sharedIds_sharedFeatures_nonDuplicate_withConflict(self):
        dataL = [["id1", "a", 1], ["id2", "b", 2], ["id3", "c", 3]]
        fNamesL = ["id", "f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["id1", 1, "x"], ["id3", 99, "y"], ["id4", 4, "z"]]
        fNamesR = ["id", "f2", "f3"]
        right = self.constructor(dataR, featureNames=fNamesR)

        pUnion_fUnion = InvalidArgumentValue
        pUnion_fIntersection = InvalidArgumentValue
        pUnion_fLeft = InvalidArgumentValue
        pIntersection_fUnion = InvalidArgumentValue
        pIntersection_fIntersection = InvalidArgumentValue
        pIntersection_fLeft = InvalidArgumentValue
        pLeft_fUnion = InvalidArgumentValue
        pLeft_fIntersection = InvalidArgumentValue

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
        ]

        self.merge_backend(left, right, expected, on="id")

    def test_merge_onFeature_sameIds_newFeatures_duplicate(self):
        dataL = [["id1", "a", 1], ["id2", "b", 2], ["id3", "c", 3]]
        fNamesL = ["id", "f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["id1", "w", 9], ["id1", "v", 8], ["id2", "x", 7], ["id3", "y", 6], ["id3", "z", 5]]
        fNamesR = ["id", "f3", "f4"]
        right = self.constructor(dataR, featureNames=fNamesR)

        mData = [["id1", "a", 1, "w", 9], ["id1", "a", 1, "v", 8], ["id2", "b", 2, "x", 7],
                 ["id3", "c", 3, "y", 6], ["id3", "c", 3, "z", 5]]
        mFtNames = ["id", "f1", "f2", "f3", "f4"]
        mLargest = self.constructor(mData, featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest[:, "id"]
        pUnion_fLeft = mLargest[:, ["id", "f1", "f2"]]
        pIntersection_fUnion = mLargest
        pIntersection_fIntersection = mLargest[:, "id"]
        pIntersection_fLeft = mLargest[:, ["id", "f1", "f2"]]
        pLeft_fUnion = mLargest
        pLeft_fIntersection = mLargest[:, "id"]

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
        ]

        self.merge_backend(left, right, expected, on="id")

    def test_merge_onFeature_sharedIds_newFeatures_duplicate(self):
        dataL = [["id1", "a", 1], ["id2", "b", 2], ["id3", "c", 3]]
        fNamesL = ["id", "f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["id1", "w", 9], ["id1", "v", 8], ["id2", "x", 7], ["id2", "y", 6], ["id4", "z", 5]]
        fNamesR = ["id", "f3", "f4"]
        right = self.constructor(dataR, featureNames=fNamesR)

        mData = [["id1", "a", 1, "w", 9], ["id1", "a", 1, "v", 8], ["id2", "b", 2, "x", 7],
                 ["id2", "b", 2, "y", 6], ["id3", "c", 3, None, None], ["id4", None, None, "z", 5]]
        mFtNames = ["id", "f1", "f2", "f3", "f4"]
        mLargest = self.constructor(mData, featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest[:, "id"]
        pUnion_fLeft = mLargest[:, ["id", "f1", "f2"]]
        pIntersection_fUnion = mLargest[:3, :]
        pIntersection_fIntersection = mLargest[:3, "id"]
        pIntersection_fLeft = mLargest[:3, ["id", "f1", "f2"]]
        pLeft_fUnion = mLargest[:4, :]
        pLeft_fIntersection = mLargest[:4, "id"]

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
        ]

        self.merge_backend(left, right, expected, on="id")

    def test_merge_onFeature_newIds_newFeatures_duplicate(self):
        dataL = [["id1", "a", 1], ["id2", "b", 2], ["id3", "c", 3]]
        fNamesL = ["id", "f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["id4", "w", 9], ["id4", "v", 8], ["id5", "x", 7], ["id5", "y", 6], ["id5", "z", 5]]
        fNamesR = ["id", "f3", "f4"]
        right = self.constructor(dataR, featureNames=fNamesR)

        mData = [["id1", "a", 1, None, None], ["id2", "b", 2, None, None],
                 ["id3", "c", 3, None, None], ["id4", None, None, "w", 9],
                 ["id4", None, None, "v", 8], ["id5", None, None, "x", 7],
                 ["id5", None, None, "y", 6], ["id5", None, None, "z", 5]]
        mFtNames = ["id", "f1", "f2", "f3", "f4"]
        mLargest = self.constructor(mData, featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest[:, "id"]
        pUnion_fLeft = mLargest[:, ["id", "f1", "f2"]]
        pIntersection_fUnion = mLargest[[], :]
        pIntersection_fIntersection = mLargest[[], "id"]
        pIntersection_fLeft = mLargest[[], ["id", "f1", "f2"]]
        pLeft_fUnion = mLargest[:2, :]
        pLeft_fIntersection = mLargest[:2, "id"]

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
        ]

        self.merge_backend(left, right, expected, on="id")

    def test_merge_onFeature_sharedIds_newFeatures_duplicate_mixedOrder(self):
        dataL = [["id1", "a", 1], ["id3", "c", 3], ["id6", "f", 6]]
        dataR = [["id2", "w", 9], ["id3", "v", 8], ["id4", "x", 7], ["id4", "y", 6], ["id5", "z", 5]]
        fNamesR = ["id", "f3", "f4"]
        left = self.constructor(dataL, featureNames=["id", "f1", "f2"])
        right = self.constructor(dataR, featureNames=fNamesR)

        mData = [["id1", "a", 1, None, None], ["id3", "c", 3, "v", 8],
                 ["id6", "f", 6, None, None],["id2", None, None, "w", 9],
                 ["id4", None, None, "x", 7], ["id4", None, None, "y", 6], ["id5", None, None, "z", 5]]
        mFtNames = ["id", "f1", "f2", "f3", "f4"]
        mLargest = self.constructor(mData, featureNames=mFtNames)

        left.merge(right, point='union', feature='union', onFeature="id")
        assert left == mLargest

    def test_merge_onFeature_sharedIds_newFeatures_duplicate_differentMatchIdx(self):
        dataL = [["id1", "a", 1], ["id3", "c", 3], ["id6", "f", 6], ["id7", "g", 7]]
        dataR = [["id3", "v", 8], ["id4", "x", 7], ["id4", "y", 6]]
        fNamesR = ["id", "f3", "f4"]
        left = self.constructor(dataL, featureNames=["id", "f1", "f2"])
        right = self.constructor(dataR, featureNames=fNamesR)

        mData = [["id1", "a", 1, None, None], ["id3", "c", 3, "v", 8],
                 ["id6", "f", 6, None, None],["id7", "g", 7, None, None],
                 ["id4", None, None, "x", 7], ["id4", None, None, "y", 6]]
        mFtNames = ["id", "f1", "f2", "f3", "f4"]
        mLargest = self.constructor(mData, featureNames=mFtNames)

        left.merge(right, point='union', feature='union', onFeature="id")
        assert left == mLargest

    def test_merge_onFeature_sameIds_sameFeatures_duplicate(self):
        dataL = [["id1", "a", 1], ["id2", "b", 2], ["id3", "c", 3]]
        fNamesL = ["id", "f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["id1", "w", 9], ["id1", "v", 8], ["id2", "x", 7], ["id3", "y", 6], ["id3", "z", 5]]
        fNamesR = ["id", "f1", "f2"]
        right = self.constructor(dataR, featureNames=fNamesR)

        pUnion_fUnion = InvalidArgumentValue
        pUnion_fIntersection = InvalidArgumentValue
        pUnion_fLeft = InvalidArgumentValue
        pIntersection_fUnion = InvalidArgumentValue
        pIntersection_fIntersection = InvalidArgumentValue
        pIntersection_fLeft = InvalidArgumentValue
        pLeft_fUnion = InvalidArgumentValue
        pLeft_fIntersection = InvalidArgumentValue

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
        ]

        self.merge_backend(left, right, expected, on="id")

    def test_merge_onFeature_sameIds_sharedFeatures_duplicate(self):
        dataL = [["id1", "a", 1], ["id2", "b", 2], ["id3", "c", 3]]
        fNamesL = ["id", "f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["id1", 1, "v"], ["id1", 1, "w"], ["id2", 2, "x"], ["id3", 3, "y"], ["id3", 3, "z"]]
        fNamesR = ["id", "f2", "f3"]
        right = self.constructor(dataR, featureNames=fNamesR)

        mData = [["id1", "a", 1, "v"], ["id1", "a", 1, "w"], ["id2", "b", 2, "x"],
                 ["id3", "c", 3, "y"], ["id3", "c", 3, "z"]]
        mFtNames = ["id", "f1", "f2", "f3"]
        mLargest = self.constructor(mData, featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest[:, ["id", "f2"]]
        pUnion_fLeft = mLargest[:, ["id", "f1", "f2"]]
        pIntersection_fUnion = mLargest
        pIntersection_fIntersection = mLargest[:, ["id", "f2"]]
        pIntersection_fLeft = mLargest[:, ["id", "f1", "f2"]]
        pLeft_fUnion = mLargest
        pLeft_fIntersection = mLargest[:, ["id", "f2"]]

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
        ]

        self.merge_backend(left, right, expected, on="id")

    def test_merge_onFeature_newIds_sharedFeatures_duplicate_noConflict(self):
        dataL = [["id1", "a", 1], ["id2", "b", 2], ["id3", "c", 3]]
        fNamesL = ["id", "f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["id4", 99, "v"], ["id4", 99, "w"], ["id5", 99, "x"], ["id5", 99, "y"], ["id6", 99, "z"]]
        fNamesR = ["id", "f2", "f3"]
        right = self.constructor(dataR, featureNames=fNamesR)

        mData = [["id1", "a", 1, None], ["id2", "b", 2, None], ["id3", "c", 3, None],
                 ["id4", None, 99, "v"], ["id4", None, 99, "w"], ["id5", None, 99, "x"],
                 ["id5", None, 99, "y"], ["id6", None, 99, "z"]]
        mFtNames = ["id", "f1", "f2", "f3"]
        mLargest = self.constructor(mData, featureNames=mFtNames)

        pUnion_fUnion = mLargest
        pUnion_fIntersection = mLargest[:, ["id", "f2"]]
        pUnion_fLeft = mLargest[:, ["id", "f1", "f2"]]
        pIntersection_fUnion = mLargest[[], :]
        pIntersection_fIntersection = mLargest[[], ["id", "f2"]]
        pIntersection_fLeft = mLargest[[], ["id", "f1", "f2"]]
        pLeft_fUnion = mLargest[[0,1,2], :]
        pLeft_fIntersection = mLargest[[0,1,2], ["id", "f2"]]

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
        ]

        self.merge_backend(left, right, expected, on="id")

    def test_merge_onFeature_newIds_sharedFeatures_duplicate_withConflict(self):
        dataL = [["id1", "a", 1], ["id2", "b", 2], ["id3", "c", 3]]
        fNamesL = ["id", "f1", "f2"]
        pNamesL = ["p1", "p2", "p3"]
        left = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        dataR = [["id1", 1, "v"], ["id1", 1, "w"], ["id2", 99, "x"], ["id3", 3, "y"], ["id3", 3, "z"]]
        fNamesR = ["id", "f2", "f3"]
        right = self.constructor(dataR, featureNames=fNamesR)

        pUnion_fUnion = InvalidArgumentValue
        pUnion_fIntersection = InvalidArgumentValue
        pUnion_fLeft = InvalidArgumentValue
        pIntersection_fUnion = InvalidArgumentValue
        pIntersection_fIntersection = InvalidArgumentValue
        pIntersection_fLeft = InvalidArgumentValue
        pLeft_fUnion = InvalidArgumentValue
        pLeft_fIntersection = InvalidArgumentValue

        expected = [
            pUnion_fUnion, pUnion_fIntersection, pUnion_fLeft,
            pIntersection_fUnion, pIntersection_fIntersection, pIntersection_fLeft,
            pLeft_fUnion, pLeft_fIntersection,
        ]

        self.merge_backend(left, right, expected, on="id")

        ###################
        # ptUnion/ftUnion #
        ###################

    @raises(InvalidArgumentValue)
    def test_merge_ptUnion_ftUnion_pointNames_ftMismatch(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        dataR = [['a',3, 4], ['b', 7, 8], ['c', -3, -4]]
        pNamesL = ['p1', 'p2', 'p3']
        pNamesR = ['p2', 'p1', 'p3']
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)
        leftObj.merge(rightObj, point='union', feature='union')

    def test_merge_pointNamesWithDefaults(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        dataR = [[3, 4], [7, 8], [-3, -4]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['f3', 'f4']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        leftObj.points.setName(0, 'a')
        rightObj.points.setName(0, 'a')
        assert leftObj.points.getName(1).startswith(DEFAULT_PREFIX)
        assert rightObj.points.getName(1).startswith(DEFAULT_PREFIX)


        leftObj.merge(rightObj, point='union', feature='union')

    def test_merge_ptUnion_ftUnion_pointNames_exactMatch(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        dataR = [[3, 4], [7, 8], [-3, -4]]
        pNames = ['a', 'b', 'c']
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['f3', 'f4']
        leftObj = self.constructor(dataL, pointNames=pNames, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNames, featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['b', 5, 6, 7, 8], ['c', -1, -2, -3, -4]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, pointNames=pNames, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='union', feature='union')
        assert leftObj == exp

    def test_merge_ptUnion_ftUnion_pointNames_exactMatch_sharedFt(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        dataR = [[3, 4, 'a'], [7, 8, 'b'], [-3, -4, 'c']]
        pNames = ['a', 'b', 'c']
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['f3', 'f4', 'id']
        leftObj = self.constructor(dataL, pointNames=pNames, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNames, featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['b', 5, 6, 7, 8], ['c', -1, -2, -3, -4]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, pointNames=pNames, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='union', feature='union')
        assert leftObj == exp

    def test_merge_ptUnion_ftUnion_pointNames_allRightInLeft(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2], ['d', -5, -6]]
        dataR = [[3, 4], [-3, -4]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['f3', 'f4']
        leftObj = self.constructor(dataL, pointNames=['a', 'b', 'c', 'd'], featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=['a', 'c'], featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['b', 5, 6, None, None], ['c', -1, -2, -3, -4], ['d', -5, -6, None, None]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, pointNames=['a', 'b', 'c', 'd'], featureNames=fNamesExp)
        leftObj.merge(rightObj, point='union', feature='union')
        assert leftObj == exp

    def test_merge_ptUnion_ftUnion_pointNames_rightNotAlwaysInLeft(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2], ['d', -5, -6]]
        dataR = [[3, 4], [0, 0], [-3, -4], [9, 9]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['f3', 'f4']
        leftObj = self.constructor(dataL, pointNames=['a', 'b', 'c', 'd'], featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=['a', 'x', 'c', 'y'], featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['b', 5, 6, None, None], ['c', -1, -2, -3, -4],
                   ['d', -5, -6, None, None], [None, None, None, 0, 0], [None, None, None, 9, 9]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, pointNames=['a', 'b', 'c', 'd', 'x', 'y'], featureNames=fNamesExp)
        leftObj.merge(rightObj, point='union', feature='union')
        assert leftObj == exp

    def test_merge_ptUnion_ftUnion_onFeature_exactMatch(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        dataR = [['a', 3, 4], ['b', 7, 8], ['c', -3, -4]]
        pNames = ['a', 'b', 'c']
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, pointNames=pNames, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNames, featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['b', 5, 6, 7, 8], ['c', -1, -2, -3, -4]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='union', feature='union', onFeature='id')
        assert leftObj == exp

    def test_merge_ptUnion_ftUnion_onFeature_exactMatch_sharedFt(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        dataR = [[3, 4, 'a'], [7, 8, 'b'], [-3, -4, 'c']]
        pNames = ['a', 'b', 'c']
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['f3', 'f4', 'id']
        leftObj = self.constructor(dataL, pointNames=pNames, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNames, featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['b', 5, 6, 7, 8], ['c', -1, -2, -3, -4]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='union', feature='union', onFeature='id')
        assert leftObj == exp

    def test_merge_ptUnion_ftUnion_onFeature_allRightInLeft(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2], ['d', -5, -6]]
        dataR = [['a', 3, 4], ['c', -3, -4]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['b', 5, 6, None, None], ['c', -1, -2, -3, -4], ['d', -5, -6, None, None]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='union', feature='union', onFeature='id')
        assert leftObj == exp

    def test_merge_ptUnion_ftUnion_onFeature_rightNotAlwaysInLeft(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2], ['d', -5, -6]]
        dataR = [['a', 3, 4], ['x', 0, 0], ['c', -3, -4], ['y', 9, 9]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['b', 5, 6, None, None], ['c', -1, -2, -3, -4],
                   ['d', -5, -6, None, None], ['x', None, None, 0, 0], ['y', None, None, 9, 9]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='union', feature='union', onFeature='id')
        assert leftObj == exp

    def test_merge_ptUnion_ftUnion_onFeature_rightOnlyUnique_matchForEachLeft(self):
        dataL = [['a', 1, 2], ['a', 4, 3], ['b', -1, -2], ['c', -6, -5], ['c', -3, -4]]
        dataR = [['a', 3, 4], ['b', -3, -4], ['c', -4, -3]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['a', 4, 3, 3, 4], ['b', -1, -2, -3, -4],
                   ['c', -6, -5, -4, -3], ['c', -3, -4, -4, -3]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='union', feature='union', onFeature='id')
        assert leftObj == exp

    def test_merge_ptUnion_ftUnion_onFeature_rightOnlyUnique_missingLeftMatches(self):
        dataL = [['a', 1, 2], ['a', 4, 3], ['b', -1, -2], ['c', -6, -5], ['c', -3, -4]]
        dataR = [['a', 3, 4], ['c', -4, -3], ['d', -3, -4]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['a', 4, 3, 3, 4], ['b', -1, -2, None, None],
                   ['c', -6, -5, -4, -3], ['c', -3, -4, -4, -3], ['d', None, None, -3, -4]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='union', feature='union', onFeature='id')
        assert leftObj == exp

    def test_merge_ptUnion_ftUnion_onFeature_leftOnlyUnique_matchForEachRight(self):
        dataL = [['a', 3, 4], ['b', -3, -4], ['c', -4, -3]]
        dataR = [['a', 1, 2], ['a', 4, 3], ['b', -1, -2], ['c', -6, -5], ['c', -3, -4]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [['a', 3, 4, 1, 2], ['a', 3, 4, 4, 3], ['b', -3, -4, -1, -2],
                   ['c', -4, -3, -6, -5], ['c', -4, -3, -3, -4]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='union', feature='union', onFeature='id')
        assert leftObj == exp

    def test_merge_ptUnion_ftUnion_onFeature_notFirstFeature(self):
        dataL = [[1, 'a', 2], [5, 'b', 6], [-1, 'c', -2], [-5, 'd', -6]]
        dataR = [[3, 4, 'a'], [-3, -4, 'c']]
        fNamesL = ['f1', 'id', 'f2']
        fNamesR = ['f3', 'f4', 'id']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [[1, 'a', 2, 3, 4], [5, 'b', 6, None, None], [-1, 'c', -2, -3, -4], [-5, 'd', -6, None, None]]
        fNamesExp = ['f1', 'id', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='union', feature='union', onFeature='id')
        assert leftObj == exp

    def test_merge_ptUnion_ftUnion_onFeature_leftOnlyUnique_notFirstFeature(self):
        dataL = [[3, 'a', 4], [-3, 'b', -4], [-4, 'c', -3]]
        dataR = [[1, 2, 'a'], [4, 3, 'a'], [-1, -2, 'b'], [-6, -5, 'c'], [-3, -4, 'c']]
        fNamesL = ['f1', 'id', 'f2']
        fNamesR = ['f3', 'f4', 'id']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [[3, 'a', 4, 1, 2], [3, 'a', 4, 4, 3], [-3, 'b', -4, -1, -2],
                   [-4, 'c', -3, -6, -5], [-4, 'c', -3, -3, -4]]
        fNamesExp = ['f1', 'id', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='union', feature='union', onFeature='id')
        assert leftObj == exp

    def test_merge_ptUnion_ftUnion_noPointNamesOrOnFeature(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        dataR = [[3, 4], [7, 8], [-3, -4]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['f3', 'f4']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [['a',1,2,None,None], ['b',5,6,None,None], ['c',-1,-2,None,None],
                   [None,None,None,3,4], [None,None,None,7,8], [None,None,None,-3,-4]]
        expFNames = fNamesL + fNamesR
        exp = self.constructor(expData, featureNames=expFNames)
        leftObj.merge(rightObj, point='union', feature='union')
        assert leftObj == exp

        ##########################
        # ptUnion/ftIntersection #
        ##########################

    @raises(InvalidArgumentValue)
    def test_merge_ptUnion_ftIntersection_ptNames_ftMismatch(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        dataR = [['a',3, 4], ['b', 7, 8], ['c', -3, -4]]
        pNamesL = ['p1', 'p2', 'p3']
        pNamesR = ['p2', 'p1', 'p3']
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)
        leftObj.merge(rightObj, point='union', feature='intersection')

    def test_merge_ptUnion_ftIntersection_pointNames_sharedFt(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2], ['d', -5, -6]]
        dataR = [['a',3, 4], ['b', 7, 8], ['c', -3, -4], ['d', -7, -8]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, pointNames=['a', 'b', 'c', 'd'], featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=['a', 'b', 'c', 'd'], featureNames=fNamesR)
        expData = [['a'], ['b'], ['c'], ['d']]
        fNamesExp = ['id']
        exp = self.constructor(expData, pointNames=['a', 'b', 'c', 'd'], featureNames=fNamesExp)
        leftObj.merge(rightObj, point='union', feature='intersection')
        assert leftObj == exp

    def test_merge_ptUnion_ftIntersection_pointNames_sharedFt_missing(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2], ['d', -5, -6]]
        dataR = [['a',3, 4], [None, 7, 8], [None, -3, -4], ['d', -7, -8]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, pointNames=['a', 'b', 'c', 'd'], featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=['a', 'b', 'c', 'd'], featureNames=fNamesR)
        expData = [['a'], ['b'], ['c'], ['d']]
        fNamesExp = ['id']
        exp = self.constructor(expData, pointNames=['a', 'b', 'c', 'd'], featureNames=fNamesExp)
        leftObj.merge(rightObj, point='union', feature='intersection')
        assert leftObj == exp


    def test_merge_ptUnion_ftIntersection_onFeature_sharedFt(self):
        dataL = [['x', 3, 'a', 4], ['y', -3, 'b', -4], ['y', -4, 'c', -3]]
        dataR = [['x', 1, 2, 'a'], ['x', 4, 3, 'a'], ['y', -1, -2, 'b'], ['y', -6, -5, 'c'], ['y', -3, -4, 'c']]
        fNamesL = ['f0', 'f1', 'id', 'f2']
        fNamesR = ['f0', 'f3', 'f4', 'id']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [['x', 'a'], ['x', 'a'], ['y', 'b'],['y', 'c'], ['y', 'c']]
        fNamesExp = ['f0', 'id']
        exp = self.constructor(expData, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='union', feature='intersection', onFeature='id')
        assert leftObj == exp

        ##########################
        # ptIntersection/ftUnion #
        ##########################

    @raises(InvalidArgumentValueCombination)
    def test_merge_ptIntersection_ftUnion_exception_noPointNamesOrOnFeature(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        dataR = [[3, 4], [7, 8], [-3, -4]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['f3', 'f4']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        leftObj.merge(rightObj, point='intersection', feature='union')

    def test_merge_ptIntersection_ftUnion_exception_pointNamesWithDefaults(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        dataR = [[3, 4], [7, 8], [-3, -4]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['f3', 'f4']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        leftObj.points.setName(0, 'a')
        rightObj.points.setName(0, 'a')
        assert leftObj.points.getName(1).startswith(DEFAULT_PREFIX)
        assert rightObj.points.getName(1).startswith(DEFAULT_PREFIX)
        expData = [['a', 1, 2, 3, 4]]
        exp = self.constructor(expData, pointNames=['a'], featureNames=fNamesL+fNamesR)
        leftObj.merge(rightObj, point='intersection', feature='union')
        assert leftObj == exp

    @raises(InvalidArgumentValue)
    def test_merge_ptIntersection_ftUnion_exception_bothNonUnique(self):
        dataL = [['a', 1, 2], ['c', 5, 6], ['c', -1, -2]]
        dataR = [['a', 3, 4], ['c', 7, 8], ['c', -3, -4]]
        pNames = ['a', 'b', 'c']
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, pointNames=pNames, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNames, featureNames=fNamesR)
        leftObj.merge(rightObj, point='intersection', feature='union', onFeature='id')

    def test_merge_ptIntersection_ftUnion_pointNames_exactMatch(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        dataR = [[3, 4], [7, 8], [-3, -4]]
        pNames = ['a', 'b', 'c']
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['f3', 'f4']
        leftObj = self.constructor(dataL, pointNames=pNames, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNames, featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['b', 5, 6, 7, 8], ['c', -1, -2, -3, -4]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, pointNames=pNames, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='intersection', feature='union')
        assert leftObj == exp

    def test_merge_ptIntersection_ftUnion_pointNames_allRightInLeft(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2], ['d', -5, -6]]
        dataR = [[3, 4], [-3, -4]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['f3', 'f4']
        leftObj = self.constructor(dataL, pointNames=['a', 'b', 'c', 'd'], featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=['a', 'c'], featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['c', -1, -2, -3, -4]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, pointNames=['a', 'c'],featureNames=fNamesExp)
        leftObj.merge(rightObj, point='intersection', feature='union')
        assert leftObj == exp

    def test_merge_ptIntersection_ftUnion_pointNames_rightNotAlwaysInLeft(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2], ['d', -5, -6]]
        dataR = [[3, 4], [0, 0], [-3, -4], [9, 9]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['f3', 'f4']
        leftObj = self.constructor(dataL, pointNames=['a', 'b', 'c', 'd'], featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=['a', 'x', 'c', 'y'], featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['c', -1, -2, -3, -4]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, pointNames=['a', 'c'], featureNames=fNamesExp)
        leftObj.merge(rightObj, point='intersection', feature='union')
        assert leftObj == exp

    def test_merge_ptIntersection_ftUnion_onFeature_exactMatch(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        dataR = [['a', 3, 4], ['b', 7, 8], ['c', -3, -4]]
        pNames = ['a', 'b', 'c']
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, pointNames=pNames, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNames, featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['b', 5, 6, 7, 8], ['c', -1, -2, -3, -4]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='intersection', feature='union', onFeature='id')
        assert leftObj == exp

    def test_merge_ptIntersection_ftUnion_onFeature_allRightInLeft(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2], ['d', -5, -6]]
        dataR = [['a', 3, 4], ['c', -3, -4]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['c', -1, -2, -3, -4]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='intersection', feature='union', onFeature='id')
        assert leftObj == exp

    def test_merge_ptIntersection_ftUnion_onFeature_rightNotAlwaysInLeft(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2], ['d', -5, -6]]
        dataR = [['a', 3, 4], ['x', 0, 0], ['c', -3, -4], ['y', 9, 9]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['c', -1, -2, -3, -4]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='intersection', feature='union', onFeature='id')
        assert leftObj == exp

    def test_merge_ptIntersection_ftUnion_onFeature_rightOnlyUnique_matchForEachLeft(self):
        dataL = [['a', 1, 2], ['a', 4, 3], ['b', -1, -2], ['c', -6, -5], ['c', -3, -4]]
        dataR = [['a', 3, 4], ['b', -3, -4], ['c', -4, -3]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['a', 4, 3, 3, 4], ['b', -1, -2, -3, -4],
                   ['c', -6, -5, -4, -3], ['c', -3, -4, -4, -3]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='intersection', feature='union', onFeature='id')
        assert leftObj == exp

    def test_merge_ptIntersection_ftUnion_onFeature_rightOnlyUnique_missingLeftMatches(self):
        dataL = [['a', 1, 2], ['a', 4, 3], ['b', -1, -2], ['c', -6, -5], ['c', -3, -4]]
        dataR = [['a', 3, 4], ['c', -4, -3], ['d', -3, -4]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['a', 4, 3, 3, 4], ['c', -6, -5, -4, -3], ['c', -3, -4, -4, -3]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='intersection', feature='union', onFeature='id')
        assert leftObj == exp

    def test_merge_ptIntersection_ftUnion_onFeature_leftOnlyUnique_matchForEachRight(self):
        dataL = [['a', 3, 4], ['b', -3, -4], ['c', -4, -3]]
        dataR = [['a', 1, 2], ['a', 4, 3], ['b', -1, -2], ['c', -6, -5], ['c', -3, -4]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [['a', 3, 4, 1, 2], ['a', 3, 4, 4, 3], ['b', -3, -4, -1, -2],
                   ['c', -4, -3, -6, -5], ['c', -4, -3, -3, -4]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='intersection', feature='union', onFeature='id')
        assert leftObj == exp

    def test_merge_ptIntersection_ftUnion_onFeature_notFirstFeature(self):
        dataL = [[1, 'a', 2], [5, 'b', 6], [-1, 'c', -2], [-5, 'd', -6]]
        dataR = [[3, 4, 'a'], [-3, -4, 'c']]
        fNamesL = ['f1', 'id', 'f2']
        fNamesR = ['f3', 'f4', 'id']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [[1, 'a', 2, 3, 4], [-1, 'c', -2, -3, -4]]
        fNamesExp = ['f1', 'id', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='intersection', feature='union', onFeature='id')
        assert leftObj == exp

    def test_merge_ptIntersection_ftUnion_onFeature_leftOnlyUnique_notFirstFeature(self):
        dataL = [[3, 'a', 4], [-3, 'b', -4], [-4, 'c', -3]]
        dataR = [[1, 2, 'a'], [4, 3, 'a'], [-1, -2, 'b'], [-6, -5, 'c'], [-3, -4, 'c']]
        fNamesL = ['f1', 'id', 'f2']
        fNamesR = ['f3', 'f4', 'id']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [[3, 'a', 4, 1, 2], [3, 'a', 4, 4, 3], [-3, 'b', -4, -1, -2],
                   [-4, 'c', -3, -6, -5], [-4, 'c', -3, -3, -4]]
        fNamesExp = ['f1', 'id', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)
        leftObj.merge(rightObj, point='intersection', feature='union', onFeature='id')
        assert leftObj == exp

        ##################
        # ptLeft/ftUnion #
        ##################

    def test_merge_ptLeft_ftUnion_onFeature_exactMatch(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2]]
        dataR = [['a', 3, 4], ['b', 7, 8], ['c', -3, -4]]
        pNames = ['a', 'b', 'c']
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, pointNames=pNames, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNames, featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['b', 5, 6, 7, 8], ['c', -1, -2, -3, -4]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)

        leftObj.merge(rightObj, point='left', feature='union', onFeature='id')
        assert leftObj == exp

    def test_merge_ptLeft_ftUnion_onFeature_allRightInLeft(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2], ['d', -5, -6]]
        dataR = [['a', 3, 4], ['c', -3, -4]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['b', 5, 6, None, None], ['c', -1, -2, -3, -4], ['d', -5, -6, None, None]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)

        leftObj.merge(rightObj, point='left', feature='union', onFeature='id')
        assert leftObj == exp

    def test_merge_ptLeft_ftUnion_onFeature_rightNotAlwaysInLeft(self):
        dataL = [['a', 1, 2], ['b', 5, 6], ['c', -1, -2], ['d', -5, -6]]
        dataR = [['a', 3, 4], ['x', 0, 0], ['c', -3, -4], ['y', 9, 9]]
        fNamesL = ['id', 'f1', 'f2']
        fNamesR = ['id', 'f3', 'f4']
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [['a', 1, 2, 3, 4], ['b', 5, 6, None, None], ['c', -1, -2, -3, -4],
                   ['d', -5, -6, None, None]]
        fNamesExp = ['id', 'f1', 'f2', 'f3', 'f4']
        exp = self.constructor(expData, featureNames=fNamesExp)

        leftObj.merge(rightObj, point='left', feature='union', onFeature='id')
        assert leftObj == exp

        ###############
        # pointStrict #
        ###############

    def test_merge_pointStrict_featureUnion_ptNames_allNames(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["c",2,2], ["b",2,2], ["a",2,2], ["d",2,2]]
        fNamesL = ["f1", "f2", "id"]
        fNamesR = ["id", "f3", "f4"]
        pNamesL = ["p1", "p2", "p3", "p4"]
        pNamesR = ["p3", "p2", "p1", "p4"]
        leftObj = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)
        expData = [[1,1,"a",2,2], [1,1,"b",2,2], [1,1,"c",2,2], [1,1,"d",2,2]]
        expFNames = ["f1", "f2", "id", "f3", "f4"]
        exp = self.constructor(expData, pointNames=pNamesL, featureNames=expFNames)

        leftObj.merge(rightObj, point='strict', feature='union', onFeature=None)
        assert leftObj == exp

    def test_merge_pointStrict_featureIntersection_ptNames_allNames(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["c",2,2], ["b",2,2], ["a",2,2], ["d",2,2]]
        fNamesL = ["f1", "f2", "id"]
        fNamesR = ["id", "f3", "f4"]
        pNamesL = ["p1", "p2", "p3", "p4"]
        pNamesR = ["p3", "p2", "p1", "p4"]
        leftObj = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)
        expData = [["a"], ["b"], ["c"], ["d"]]
        expFNames = ["id"]
        exp = self.constructor(expData, pointNames=pNamesL, featureNames=expFNames)

        leftObj.merge(rightObj, point='strict', feature='intersection', onFeature=None)
        assert leftObj == exp

    def test_merge_pointStrict_featureUnion_onFeature_allNames(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["c",2,2], ["b",2,2], ["a",2,2], ["d",2,2]]
        fNamesL = ["f1", "f2", "id"]
        fNamesR = ["id", "f3", "f4"]
        pNamesL = ["p1", "p2", "p3", "p4"]
        pNamesR = ["p3", "p2", "p1", "p4"]
        leftObj = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)
        expData = [[1,1,"a",2,2], [1,1,"b",2,2], [1,1,"c",2,2], [1,1,"d",2,2]]
        expFNames = ["f1", "f2", "id", "f3", "f4"]
        exp = self.constructor(expData, featureNames=expFNames, pointNames=pNamesL)

        leftObj.merge(rightObj, point='strict', feature='union', onFeature="id")
        assert leftObj == exp

    def test_merge_pointStrict_featureIntersection_onFeature_allNames(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["c",2,2], ["b",2,2], ["a",2,2], ["d",2,2]]
        fNamesL = ["f1", "f2", "id"]
        fNamesR = ["id", "f3", "f4"]
        pNamesL = ["p1", "p2", "p3", "p4"]
        pNamesR = ["p3", "p2", "p1", "p4"]
        leftObj = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)
        expData = [["a"], ["b"], ["c"], ["d"]]
        expFNames = ["id"]
        exp = self.constructor(expData, featureNames=expFNames, pointNames=pNamesL)

        leftObj.merge(rightObj, point='strict', feature='intersection', onFeature="id")
        assert leftObj == exp

    def test_merge_pointStrict_featureUnion_ptNames_ptNamesOnly(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["c",2,2], ["b",2,2], ["a",2,2], ["d",2,2]]
        pNamesL = ["p1", "p2", "p3", "p4"]
        pNamesR = ["p3", "p2", "p1", "p4"]
        leftObj = self.constructor(dataL, pointNames=pNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR)
        expData = [[1,1,"a","a",2,2], [1,1,"b","b",2,2], [1,1,"c","c",2,2], [1,1,"d","d",2,2]]
        exp = self.constructor(expData, pointNames=pNamesL)

        leftObj.merge(rightObj, point='strict', feature='union', onFeature=None)
        assert leftObj == exp

    @raises(InvalidArgumentValueCombination)
    def test_merge_pointStrict_featureIntersection_ptNames_ptNamesOnly(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["c",2,2], ["b",2,2], ["a",2,2], ["d",2,2]]
        pNamesL = ["p1", "p2", "p3", "p4"]
        pNamesR = ["p3", "p2", "p1", "p4"]
        leftObj = self.constructor(dataL, pointNames=pNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR)

        leftObj.merge(rightObj, point='strict', feature='intersection', onFeature=None)

    @raises(InvalidArgumentValue)
    def test_merge_pointStrict_featureUnion_onFeature_ptNamesOnly(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["c",2,2], ["b",2,2], ["a",2,2], ["d",2,2]]
        pNamesL = ["p1", "p2", "p3", "p4"]
        pNamesR = ["p3", "p2", "p1", "p4"]
        leftObj = self.constructor(dataL, pointNames=pNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR)

        leftObj.merge(rightObj, point='strict', feature='union', onFeature="id")

    @raises(InvalidArgumentValue)
    def test_merge_pointStrict_featureIntersection_onFeature_ptNamesOnly(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["c",2,2], ["b",2,2], ["a",2,2], ["d",2,2]]
        pNamesL = ["p1", "p2", "p3", "p4"]
        pNamesR = ["p3", "p2", "p1", "p4"]
        leftObj = self.constructor(dataL, pointNames=pNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR)

        leftObj.merge(rightObj, point='strict', feature='intersection', onFeature="id")

    @raises(InvalidArgumentValue)
    def test_merge_pointStrict_featureUnion_ptNames_ftNamesOnly(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["c",2,2], ["b",2,2], ["a",2,2], ["d",2,2]]
        fNamesL = ["f1", "f2", "id"]
        fNamesR = ["id", "f3", "f4"]
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)

        leftObj.merge(rightObj, point='strict', feature='union', onFeature=None)

    @raises(InvalidArgumentValue)
    def test_merge_pointStrict_featureIntersection_ptNames_ftNamesOnly(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["c",2,2], ["b",2,2], ["a",2,2], ["d",2,2]]
        fNamesL = ["f1", "f2", "id"]
        fNamesR = ["id", "f3", "f4"]
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)

        leftObj.merge(rightObj, point='strict', feature='intersection', onFeature=None)

    def test_merge_pointStrict_featureUnion_onFeature_ftNamesOnly(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["c",2,2], ["b",2,2], ["a",2,2], ["d",2,2]]
        fNamesL = ["f1", "f2", "id"]
        fNamesR = ["id", "f3", "f4"]
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [[1,1,"a",2,2], [1,1,"b",2,2], [1,1,"c",2,2], [1,1,"d",2,2]]
        expFNames = ["f1", "f2", "id", "f3", "f4"]
        exp = self.constructor(expData, featureNames=expFNames)

        leftObj.merge(rightObj, point='strict', feature='union', onFeature="id")
        assert leftObj == exp

    def test_merge_pointStrict_featureIntersection_onFeature_ftNamesOnly(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["c",2,2], ["b",2,2], ["a",2,2], ["d",2,2]]
        fNamesL = ["f1", "f2", "id"]
        fNamesR = ["id", "f3", "f4"]
        leftObj = self.constructor(dataL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, featureNames=fNamesR)
        expData = [["a"], ["b"], ["c"], ["d"]]
        expFNames = ["id"]
        exp = self.constructor(expData, featureNames=expFNames)

        leftObj.merge(rightObj, point='strict', feature='intersection', onFeature="id")
        assert leftObj == exp

    def test_merge_pointStrict_featureUnion_ptNames_noNames(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["c",2,2], ["b",2,2], ["a",2,2], ["d",2,2]]
        leftObj = self.constructor(dataL)
        rightObj = self.constructor(dataR)
        expData = [[1,1,"a","c",2,2], [1,1,"b","b",2,2], [1,1,"c","a",2,2], [1,1,"d","d",2,2]]
        exp = self.constructor(expData)

        leftObj.merge(rightObj, point='strict', feature='union', onFeature=None)
        assert leftObj == exp

    @raises(InvalidArgumentValueCombination)
    def test_merge_pointStrict_featureIntersection_ptNames_noNames(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["c",2,2], ["b",2,2], ["a",2,2], ["d",2,2]]
        leftObj = self.constructor(dataL)
        rightObj = self.constructor(dataR)

        leftObj.merge(rightObj, point='strict', feature='intersection')

    @raises(InvalidArgumentValue)
    def test_merge_pointStrict_featureIntersection_onFeature_noNames(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["c",2,2], ["b",2,2], ["a",2,2], ["d",2,2]]
        leftObj = self.constructor(dataL)
        rightObj = self.constructor(dataR)

        leftObj.merge(rightObj, point='strict', feature='intersection', onFeature="id")

        #################
        # featureStrict #
        #################

    def test_merge_featureStrict_pointUnion_ptNames_allNames(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["d",1,1], ["x",2,2], ["y",2,2], ["z",2,2]]
        fNamesL = ["f1", "f2", "id"]
        fNamesR = ["id", "f1", "f2"]
        pNamesL = ["p1", "p2", "p3", "p4"]
        pNamesR = ["p4", "p5", "p6", "p7"]
        leftObj = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)
        expData = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"], [2,2,"x"], [2,2,"y"], [2,2,"z"]]
        expPNames = ["p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        exp = self.constructor(expData, pointNames=expPNames, featureNames=fNamesL)

        leftObj.merge(rightObj, point='union', feature='strict', onFeature=None)
        assert leftObj == exp

    @raises(InvalidArgumentValue)
    def test_merge_featureStrict_pointUnion_ptNames_allNames_ftMismatch(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["d",1,1], ["x",2,2], ["y",2,2], ["z",2,2]]
        fNamesL = ["f1", "f2", "id"]
        fNamesR = ["id", "f1", "f3"]
        pNamesL = ["p1", "p2", "p3", "p4"]
        pNamesR = ["p4", "p5", "p6", "p7"]
        leftObj = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)
        expData = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"], [2,2,"x"], [2,2,"y"], [2,2,"z"]]
        expPNames = ["p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        exp = self.constructor(expData, pointNames=expPNames, featureNames=fNamesL)

        leftObj.merge(rightObj, point='union', feature='strict', onFeature=None)
        assert leftObj == exp

    def test_merge_featureStrict_pointIntersection_ptNames_allNames(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["d",1,1], ["x",2,2], ["y",2,2], ["z",2,2]]
        fNamesL = ["f1", "f2", "id"]
        fNamesR = ["id", "f1", "f2"]
        pNamesL = ["p1", "p2", "p3", "p4"]
        pNamesR = ["p4", "p5", "p6", "p7"]
        leftObj = self.constructor(dataL, pointNames=pNamesL, featureNames=fNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR, featureNames=fNamesR)
        expData = [[1,1,"d"]]
        expPNames = ["p4"]
        exp = self.constructor(expData, pointNames=expPNames, featureNames=fNamesL)

        leftObj.merge(rightObj, point='intersection', feature='strict', onFeature=None)
        assert leftObj == exp

    def test_merge_featureStrict_pointUnion_ptNames_ptNamesOnly(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["d",1,1], ["x",2,2], ["y",2,2], ["z",2,2]]
        pNamesL = ["p1", "p2", "p3", "p4"]
        pNamesR = ["p5", "p6", "p7", "p8"]
        leftObj = self.constructor(dataL, pointNames=pNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR)
        expData = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"], ["d",1,1], ["x",2,2], ["y",2,2], ["z",2,2]]
        expPNames = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
        exp = self.constructor(expData, pointNames=expPNames)

        leftObj.merge(rightObj, point='union', feature='strict', onFeature=None)
        assert leftObj == exp

    @raises(InvalidArgumentValue)
    def test_merge_featureStrict_pointUnion_ptNames_ptNamesOnly_ptMismatch(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["d",1,1], ["x",2,2], ["y",2,2], ["z",2,2]]
        pNamesL = ["p1", "p2", "p3", "p4"]
        pNamesR = ["p4", "p5", "p6", "p7"]
        leftObj = self.constructor(dataL, pointNames=pNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR)

        leftObj.merge(rightObj, point='union', feature='strict', onFeature=None)

    def test_merge_featureStrict_pointIntersection_ptNames_ptNamesOnly(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["d",1,1], ["x",2,2], ["y",2,2], ["z",2,2]]
        pNamesL = ["p1", "p2", "p3", "p4"]
        pNamesR = ["p5", "p6", "p7", "p8"]
        leftObj = self.constructor(dataL, pointNames=pNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR)
        expData = numpy.array([[],[],[]]).T
        exp = self.constructor(expData)
        leftObj.merge(rightObj, point='intersection', feature='strict', onFeature=None)
        assert leftObj == exp

    @raises(InvalidArgumentValue)
    def test_merge_featureStrict_pointIntersection_ptNames_ptNamesOnly_ptMismatch(self):
        dataL = [[1,1,"a"], [1,1,"b"], [1,1,"c"], [1,1,"d"]]
        dataR = [["d",1,1], ["x",2,2], ["y",2,2], ["z",2,2]]
        pNamesL = ["p1", "p2", "p3", "p4"]
        pNamesR = ["p4", "p5", "p6", "p7"]
        leftObj = self.constructor(dataL, pointNames=pNamesL)
        rightObj = self.constructor(dataR, pointNames=pNamesR)

        leftObj.merge(rightObj, point='intersection', feature='strict', onFeature=None)

def exceptionHelper(testObj, target, args, wanted, checkMsg):
    try:
        getattr(testObj, target)(*args)
        assert False  # expected an exception
    except wanted as check:
        if checkMsg:
            print(check)

class StructureAll(StructureDataSafe, StructureModifying):
    pass
