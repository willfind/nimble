"""

Methods tested in this file:

In object StructureDataSafe:
copyAs, copyPoints, copyFeatures


In object StructureModifying:
__init__,  transpose, addPoints, addFeatures, sortPoints, sortFeatures,
extractPoints, extractFeatures, deletePoints, deleteFeatures, retainPoints,
retainFeatures, referenceDataFrom, transformEachPoint, transformEachFeature,
transformEachElement, fillWith, flattenToOnePoint, flattenToOneFeature,
unflattenFromOnePoint, unflattenFromOneFeature

"""

from __future__ import absolute_import
from __future__ import print_function
import tempfile
import numpy
import os
import os.path
from nose.tools import *
try:
    from unittest import mock #python >=3.3
except:
    import mock

from copy import deepcopy

import UML
from UML import createData
from UML.data import List
from UML.data import Matrix
from UML.data import DataFrame
from UML.data import Sparse
from UML.data import BaseView
from UML.data.dataHelpers import DEFAULT_PREFIX
from UML.exceptions import ArgumentException
from UML.exceptions import ImproperActionException
from UML.randomness import numpyRandom

from UML.data.tests.baseObject import DataTestObject
from six.moves import map
scipy = UML.importModule('scipy.sparse')
pd = UML.importModule('pandas')

preserveName = "PreserveTestName"
preserveAPath = os.path.join(os.getcwd(), "correct", "looking", "path")
preserveRPath = os.path.relpath(preserveAPath)
preservePair = (preserveAPath, preserveRPath)


### Helpers used by tests in the test class ###

def passThrough(value):
    return value


def plusOne(value):
    return (value + 1)


def plusOneOnlyEven(value):
    if value % 2 == 0:
        return (value + 1)
    else:
        return None

class CalledFunctionException(Exception):
    def __init__(self):
        pass

def calledException(*args, **kwargs):
    raise CalledFunctionException()

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
            toCall = structure + "Points"
        else:
            toCall = structure + "Features"
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = getattr(toTest, toCall)([0,1,2], randomize=True)

    def back_structural_list_numberGreaterThanTargeted(self, structure, axis):
        if axis == 'point':
            toCall = structure + "Points"
        else:
            toCall = structure + "Features"
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = getattr(toTest, toCall)([0,1], number=3)

    def back_structural_function_numberGreaterThanTargeted(self, structure, axis):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']

        if axis == 'point':
            toCall = structure + "Points"
            def selTwo(p):
                return p.getPointName(0) in pointNames[:2]
        else:
            toCall = structure + "Features"
            def selTwo(f):
                return f.getFeatureName(0) in featureNames[:2]

        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = getattr(toTest, toCall)(selTwo, number=3)

    def back_structural_range_numberGreaterThanTargeted(self, structure, axis):
        if axis == 'point':
            toCall = structure + "Points"
        else:
            toCall = structure + "Features"
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = getattr(toTest, toCall)(start=0, end=1, number=3)


class StructureDataSafe(StructureShared):

    #############
    # copyAs #
    #############

    def test_copy_withZeros(self):
        """ Test copyAs() produces an equal object and doesn't just copy the references """
        data1 = [[1, 2, 3, 0], [1, 0, 3, 0], [2, 4, 6, 0], [0, 0, 0, 0]]
        featureNames = ['one', 'two', 'three', 'four']
        pointNames = ['1', 'one', '2', '0']
        orig = self.constructor(data1, pointNames=pointNames, featureNames=featureNames)

        dup1 = orig.copy()
        dup2 = orig.copyAs(orig.getTypeString())

        assert orig.isIdentical(dup1)
        assert dup1.isIdentical(orig)

        assert orig.data is not dup1.data

        assert orig.isIdentical(dup2)
        assert dup2.isIdentical(orig)

        assert orig.data is not dup2.data


    def test_copy_Pempty(self):
        """ test copyAs() produces the correct outputs when given an point empty object """
        data = [[], []]
        data = numpy.array(data).T

        orig = self.constructor(data)
        sparseObj = createData(returnType="Sparse", data=data)
        listObj = createData(returnType="List", data=data)
        matixObj = createData(returnType="Matrix", data=data)
        dataframeObj = createData(returnType="DataFrame", data=data)

        copySparse = orig.copyAs(format='Sparse')
        assert copySparse.isIdentical(sparseObj)
        assert sparseObj.isIdentical(copySparse)

        copyList = orig.copyAs(format='List')
        assert copyList.isIdentical(listObj)
        assert listObj.isIdentical(copyList)

        copyMatrix = orig.copyAs(format='Matrix')
        assert copyMatrix.isIdentical(matixObj)
        assert matixObj.isIdentical(copyMatrix)

        copyDataFrame = orig.copyAs(format='DataFrame')
        assert copyDataFrame.isIdentical(copyDataFrame)
        assert dataframeObj.isIdentical(copyDataFrame)

        pyList = orig.copyAs(format='python list')
        assert pyList == []

        numpyArray = orig.copyAs(format='numpy array')
        assert numpy.array_equal(numpyArray, data)

        numpyMatrix = orig.copyAs(format='numpy matrix')
        assert numpy.array_equal(numpyMatrix, numpy.matrix(data))

        listOfDict = orig.copyAs(format='list of dict')
        assert listOfDict == []

        dictOfList = orig.copyAs(format='dict of list')
        assert dictOfList == {'_DEFAULT_#0': [], '_DEFAULT_#1': []}


    def test_copy_Fempty(self):
        """ test copyAs() produces the correct outputs when given an feature empty object """
        data = [[], []]
        data = numpy.array(data)

        orig = self.constructor(data)
        sparseObj = createData(returnType="Sparse", data=data)
        listObj = createData(returnType="List", data=data)
        matixObj = createData(returnType="Matrix", data=data)
        dataframeObj = createData(returnType="DataFrame", data=data)

        copySparse = orig.copyAs(format='Sparse')
        assert copySparse.isIdentical(sparseObj)
        assert sparseObj.isIdentical(copySparse)

        copyList = orig.copyAs(format='List')
        assert copyList.isIdentical(listObj)
        assert listObj.isIdentical(copyList)

        copyMatrix = orig.copyAs(format='Matrix')
        assert copyMatrix.isIdentical(matixObj)
        assert matixObj.isIdentical(copyMatrix)

        copyDataFrame = orig.copyAs(format='DataFrame')
        assert copyDataFrame.isIdentical(copyDataFrame)
        assert dataframeObj.isIdentical(copyDataFrame)

        pyList = orig.copyAs(format='python list')
        assert pyList == [[], []]

        numpyArray = orig.copyAs(format='numpy array')
        assert numpy.array_equal(numpyArray, data)

        numpyMatrix = orig.copyAs(format='numpy matrix')
        assert numpy.array_equal(numpyMatrix, numpy.matrix(data))

        listOfDict = orig.copyAs(format='list of dict')
        assert listOfDict == [{}, {}]

        dictOfList = orig.copyAs(format='dict of list')
        assert dictOfList == {}

    def test_copy_Trueempty(self):
        """ test copyAs() produces the correct outputs when given a point and feature empty object """
        data = numpy.empty(shape=(0, 0))

        orig = self.constructor(data)
        sparseObj = createData(returnType="Sparse", data=data)
        listObj = createData(returnType="List", data=data)
        matixObj = createData(returnType="Matrix", data=data)
        dataframeObj = createData(returnType="DataFrame", data=data)

        copySparse = orig.copyAs(format='Sparse')
        assert copySparse.isIdentical(sparseObj)
        assert sparseObj.isIdentical(copySparse)

        copyList = orig.copyAs(format='List')
        assert copyList.isIdentical(listObj)
        assert listObj.isIdentical(copyList)

        copyMatrix = orig.copyAs(format='Matrix')
        assert copyMatrix.isIdentical(matixObj)
        assert matixObj.isIdentical(copyMatrix)

        copyDataFrame = orig.copyAs(format='DataFrame')
        assert copyDataFrame.isIdentical(copyDataFrame)
        assert dataframeObj.isIdentical(copyDataFrame)

        pyList = orig.copyAs(format='python list')
        assert pyList == []

        numpyArray = orig.copyAs(format='numpy array')
        assert numpy.array_equal(numpyArray, data)

        numpyMatrix = orig.copyAs(format='numpy matrix')
        assert numpy.array_equal(numpyMatrix, numpy.matrix(data))

        listOfDict = orig.copyAs(format='list of dict')
        assert listOfDict == []

        dictOfList = orig.copyAs(format='dict of list')
        assert dictOfList == {}

    def test_copy_rightTypeTrueCopy(self):
        """ Test copyAs() will return all of the right type and do not show each other's modifications"""

        data = [[1, 2, 3], [1, 0, 3], [2, 4, 6], [0, 0, 0]]
        featureNames = ['one', 'two', 'three']
        pointNames = ['1', 'one', '2', '0']
        orig = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        sparseObj = createData(returnType="Sparse", data=data, pointNames=pointNames, featureNames=featureNames)
        listObj = createData(returnType="List", data=data, pointNames=pointNames, featureNames=featureNames)
        matixObj = createData(returnType="Matrix", data=data, pointNames=pointNames, featureNames=featureNames)
        dataframeObj = createData(returnType="DataFrame", data=data, pointNames=pointNames, featureNames=featureNames)

        pointsShuffleIndices = [3, 1, 2, 0]
        featuresShuffleIndices = [1, 2, 0]

        copySparse = orig.copyAs(format='Sparse')
        assert copySparse.isIdentical(sparseObj)
        assert sparseObj.isIdentical(copySparse)
        assert type(copySparse) == Sparse
        copySparse.setFeatureName('two', '2')
        copySparse.setPointName('one', 'WHAT')
        assert 'two' in orig.getFeatureNames()
        assert 'one' in orig.getPointNames()
        copySparse.sortPoints(sortHelper=pointsShuffleIndices)
        copySparse.sortFeatures(sortHelper=featuresShuffleIndices)
        assert orig[0, 0] == 1

        copyList = orig.copyAs(format='List')
        assert copyList.isIdentical(listObj)
        assert listObj.isIdentical(copyList)
        assert type(copyList) == List
        copyList.setFeatureName('two', '2')
        copyList.setPointName('one', 'WHAT')
        assert 'two' in orig.getFeatureNames()
        assert 'one' in orig.getPointNames()
        copyList.sortPoints(sortHelper=pointsShuffleIndices)
        copyList.sortFeatures(sortHelper=featuresShuffleIndices)
        assert orig[0, 0] == 1

        copyMatrix = orig.copyAs(format='Matrix')
        assert copyMatrix.isIdentical(matixObj)
        assert matixObj.isIdentical(copyMatrix)
        assert type(copyMatrix) == Matrix
        copyMatrix.setFeatureName('two', '2')
        copyMatrix.setPointName('one', 'WHAT')
        assert 'two' in orig.getFeatureNames()
        assert 'one' in orig.getPointNames()
        copyMatrix.sortPoints(sortHelper=pointsShuffleIndices)
        copyMatrix.sortFeatures(sortHelper=featuresShuffleIndices)
        assert orig[0, 0] == 1

        copyDataFrame = orig.copyAs(format='DataFrame')
        assert copyDataFrame.isIdentical(dataframeObj)
        assert dataframeObj.isIdentical(copyDataFrame)
        assert type(copyDataFrame) == DataFrame
        copyDataFrame.setFeatureName('two', '2')
        copyDataFrame.setPointName('one', 'WHAT')
        assert 'two' in orig.getFeatureNames()
        assert 'one' in orig.getPointNames()
        copyDataFrame.sortPoints(sortHelper=pointsShuffleIndices)
        copyDataFrame.sortFeatures(sortHelper=featuresShuffleIndices)
        assert orig[0, 0] == 1


        pyList = orig.copyAs(format='python list')
        assert type(pyList) == list
        pyList[0][0] = 5
        assert orig[0, 0] == 1

        numpyArray = orig.copyAs(format='numpy array')
        assert type(numpyArray) == type(numpy.array([]))
        numpyArray[0, 0] = 5
        assert orig[0, 0] == 1

        numpyMatrix = orig.copyAs(format='numpy matrix')
        assert type(numpyMatrix) == type(numpy.matrix([]))
        numpyMatrix[0, 0] = 5
        assert orig[0, 0] == 1

        if scipy:
            spcsc = orig.copyAs(format='scipy csc')
            assert type(spcsc) == type(scipy.sparse.csc_matrix(numpy.matrix([])))
            spcsc[0, 0] = 5
            assert orig[0, 0] == 1

            spcsr = orig.copyAs(format='scipy csr')
            assert type(spcsr) == type(scipy.sparse.csr_matrix(numpy.matrix([])))
            spcsr[0, 0] = 5
            assert orig[0, 0] == 1

        listOfDict = orig.copyAs(format='list of dict')
        assert type(listOfDict) == list
        assert type(listOfDict[0]) == dict
        listOfDict[0]['one'] = 5
        assert orig[0, 0] == 1

        dictOfList = orig.copyAs(format='dict of list')
        assert type(dictOfList) == dict
        assert type(dictOfList['one']) == list
        dictOfList['one'][0] = 5
        assert orig[0, 0] == 1

    def test_copy_rowsArePointsFalse(self):
        """ Test copyAs() will return data in the right places when rowsArePoints is False"""
        data = [[1, 2, 3], [1, 0, 3], [2, 4, 6], [0, 0, 0]]
        dataT = numpy.array(data).T

        featureNames = ['one', 'two', 'three']
        pointNames = ['1', 'one', '2', '0']
        orig = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        for retType in UML.data.available:

            out = orig.copyAs(retType, rowsArePoints=False)

            desired = UML.createData(retType, dataT, pointNames=featureNames, featureNames=pointNames)

            assert out == desired

        out = orig.copyAs(format='pythonlist', rowsArePoints=False)

        assert out == dataT.tolist()

        out = orig.copyAs(format='numpyarray', rowsArePoints=False)

        assert numpy.array_equal(out, dataT)

        out = orig.copyAs(format='numpymatrix', rowsArePoints=False)

        assert numpy.array_equal(out, dataT)

        if scipy:

            out = orig.copyAs(format='scipycsr', rowsArePoints=False)

            assert numpy.array_equal(out.toarray(), dataT)

            out = orig.copyAs(format='scipycsc', rowsArePoints=False)

            assert numpy.array_equal(out.toarray(), dataT)

        out = orig.copyAs(format='list of dict', rowsArePoints=False)

        desired = self.constructor(dataT, pointNames=featureNames, featureNames=pointNames)
        desired = desired.copyAs(format='list of dict')

        assert out == desired

        out = orig.copyAs(format='dict of list', rowsArePoints=False)

        desired = self.constructor(dataT, pointNames=featureNames, featureNames=pointNames)
        desired = desired.copyAs(format='dict of list')

        assert out == desired

    def test_copy_outputAs1DWrongFormat(self):
        """ Test copyAs will raise exception when given an unallowed format """
        data = [[1, 2, 3], [1, 0, 3], [2, 4, 6], [0, 0, 0]]
        featureNames = ['one', 'two', 'three']
        pointNames = ['1', 'one', '2', '0']
        orig = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        try:
            orig.copyAs("List", outputAs1D=True)
            assert False
        except ArgumentException as ae:
            print(ae)
        try:
            orig.copyAs("Matrix", outputAs1D=True)
            assert False
        except ArgumentException as ae:
            print(ae)
        try:
            orig.copyAs("Sparse", outputAs1D=True)
            assert False
        except ArgumentException as ae:
            print(ae)
        try:
            orig.copyAs("numpy matrix", outputAs1D=True)
            assert False
        except ArgumentException as ae:
            print(ae)
        if scipy:
            try:
                orig.copyAs("scipy csr", outputAs1D=True)
                assert False
            except ArgumentException as ae:
                print(ae)
            try:
                orig.copyAs("scipy csc", outputAs1D=True)
                assert False
            except ArgumentException as ae:
                print(ae)
        try:
            orig.copyAs("list of dict", outputAs1D=True)
            assert False
        except ArgumentException as ae:
            print(ae)
        try:
            orig.copyAs("dict of list", outputAs1D=True)
            assert False
        except ArgumentException as ae:
            print(ae)

    @raises(ArgumentException)
    def test_copy_outputAs1DWrongShape(self):
        """ Test copyAs will raise exception when given an unallowed shape """
        data = [[1, 2, 3], [1, 0, 3], [2, 4, 6], [0, 0, 0]]
        featureNames = ['one', 'two', 'three']
        pointNames = ['1', 'one', '2', '0']
        orig = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        orig.copyAs("numpy array", outputAs1D=True)


    def test_copyAs_outpuAs1DTrue(self):
        """ Test copyAs() will return successfully output 1d for all allowable possibilities"""
        dataPv = [[1, 2, 0, 3]]
        dataFV = [[1], [2], [3], [0]]
        origPV = self.constructor(dataPv)
        origFV = self.constructor(dataFV)

        outPV = origPV.copyAs('python list', outputAs1D=True)
        assert outPV == [1, 2, 0, 3]

        outFV = origFV.copyAs('numpy array', outputAs1D=True)
        assert numpy.array_equal(outFV, numpy.array([1, 2, 3, 0]))

    def test_copyAs_NameAndPath(self):
        """ Test copyAs() will preserve name and path attributes"""

        data = [[1, 2, 3], [1, 0, 3], [2, 4, 6], [0, 0, 0]]
        name = 'copyAsTestName'
        orig = self.constructor(data)
        with tempfile.NamedTemporaryFile(suffix=".csv") as source:
            orig.writeFile(source.name, 'csv', includeNames=False)
            orig = self.constructor(source.name, name=name)
            path = source.name

        assert orig.name == name
        assert orig.path == path
        assert orig.absolutePath == path
        assert orig.relativePath == os.path.relpath(path)

        copySparse = orig.copyAs(format='Sparse')
        assert copySparse.name == orig.name
        assert copySparse.path == orig.path
        assert copySparse.absolutePath == path
        assert copySparse.relativePath == os.path.relpath(path)

        copyList = orig.copyAs(format='List')
        assert copyList.name == orig.name
        assert copyList.path == orig.path
        assert copyList.absolutePath == path
        assert copyList.relativePath == os.path.relpath(path)

        copyMatrix = orig.copyAs(format='Matrix')
        assert copyMatrix.name == orig.name
        assert copyMatrix.path == orig.path
        assert copyMatrix.absolutePath == path
        assert copyMatrix.relativePath == os.path.relpath(path)

        copyDataFrame = orig.copyAs(format='DataFrame')
        assert copyDataFrame.name == orig.name
        assert copyDataFrame.path == orig.path
        assert copyDataFrame.absolutePath == path
        assert copyDataFrame.relativePath == os.path.relpath(path)


    ###################
    # copyPoints #
    ###################

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_copyPoints_calls_constructIndicesList(self, mockFunc):
        toTest = self.constructor([[1,2,],[3,4]], pointNames=['a', 'b'])

        ret = toTest.copyPoints(['a', 'b'])

    def test_copyPoints_handmadeSingle(self):
        """ Test copyPoints() against handmade output when copying one point """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ext1 = toTest.copyPoints(0)
        exp1 = self.constructor([[1, 2, 3]])
        assert ext1.isIdentical(exp1)
        expEnd = self.constructor(data)
        assert toTest.isIdentical(expEnd)

    def test_copyPoints_index_NamePath_Preserve(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        # need to set source paths for view objects
        if isinstance(toTest, UML.data.BaseView):
            toTest._source._absPath = 'testAbsPath'
            toTest._source._relPath = 'testRelPath'
        else:
            toTest._absPath = 'testAbsPath'
            toTest._relPath = 'testRelPath'
        toTest._name = 'testName'

        ext1 = toTest.copyPoints(0)

        assert ext1.nameIsDefault()
        assert ext1.path == 'testAbsPath'
        assert ext1.absolutePath == 'testAbsPath'
        assert ext1.relativePath == 'testRelPath'

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

    def test_copyPoints_ListIntoPEmpty(self):
        """ Test copyPoints() by copying a list of all points """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        expRet = self.constructor(data)
        expTest = toTest.copy()
        ret = toTest.copyPoints([0, 1, 2, 3])

        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(expTest)


    def test_copyPoints_handmadeListSequence(self):
        """ Test copyPoints() against handmade output for multiple copies """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        names = ['1', '4', '7', '10']
        toTest = self.constructor(data, pointNames=names)
        ext1 = toTest.copyPoints('1')
        exp1 = self.constructor([[1, 2, 3]], pointNames=['1'])
        assert ext1.isIdentical(exp1)
        ext2 = toTest.copyPoints([1, 2])
        exp2 = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=['4', '7'])
        assert ext2.isIdentical(exp2)
        expEnd = self.constructor(data, pointNames=names)
        assert toTest.isIdentical(expEnd)

    def test_copyPoints_handmadeListOrdering(self):
        """ Test copyPoints() against handmade output for out of order copying """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
        names = ['1', '4', '7', '10', '13']
        toTest = self.constructor(data, pointNames=names)
        ext1 = toTest.copyPoints([3, 4, 1])
        exp1 = self.constructor([[10, 11, 12], [13, 14, 15], [4, 5, 6]], pointNames=['10', '13', '4'])
        assert ext1.isIdentical(exp1)
        expEnd = self.constructor(data, pointNames=names)
        assert toTest.isIdentical(expEnd)

    def test_copyPoints_List_trickyOrdering(self):
        data = [[0], [2], [2], [2], [0], [0], [0], [0], [2], [0]]
        toCopy = [6, 5, 3, 9]

        toTest = self.constructor(data)

        ret = toTest.copyPoints(toCopy)

        expRaw = [[0], [0], [2], [0]]
        expRet = self.constructor(expRaw)

        expTest = self.constructor(data)

        assert ret == expRet
        assert toTest == expTest

    def test_copyPoints_function_selectionGap(self):
        data = [[0], [2], [2], [2], [0], [0], [0], [0], [2], [0]]
        copyIndices = [3, 5, 6, 9]
        pnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        def sel(point):
            if int(point.getPointName(0)) in copyIndices:
                return True
            else:
                return False

        toTest = self.constructor(data, pointNames=pnames)
        ret = toTest.copyPoints(sel)

        expRaw = [[2], [0], [0], [0]]
        expNames = ['3', '5', '6', '9']
        expRet = self.constructor(expRaw, pointNames=expNames)
        expTest = self.constructor(data, pointNames=pnames)
        assert ret == expRet
        assert toTest == expTest


    def test_copyPoints_functionIntoPEmpty(self):
        """ Test copyPoints() by copying all points using a function """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        expRet = self.constructor(data)

        ret = toTest.copyPoints(allTrue)
        expTest = self.constructor(data)

        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(expTest)

    def test_copyPoints_function_returnPointEmpty(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        expTest = self.constructor(data)

        ret = toTest.copyPoints(allFalse)

        data = [[], [], []]
        data = numpy.array(data).T
        expRet = self.constructor(data)

        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(expTest)

    def test_copyPoints_handmadeFunction(self):
        """ Test copyPoints() against handmade output for function copying """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        ext = toTest.copyPoints(oneOrFour)
        exp = self.constructor([[1, 2, 3], [4, 5, 6]])
        assert ext.isIdentical(exp)
        expEnd = self.constructor(data)
        assert toTest.isIdentical(expEnd)

    def test_copyPoints_func_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        # need to set source paths for view objects
        if isinstance(toTest, UML.data.BaseView):
            toTest._source._absPath = 'testAbsPath'
            toTest._source._relPath = 'testRelPath'
        else:
            toTest._absPath = 'testAbsPath'
            toTest._relPath = 'testRelPath'
        toTest._name = 'testName'

        ext = toTest.copyPoints(oneOrFour)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

        assert ext.nameIsDefault()
        assert ext.absolutePath == 'testAbsPath'
        assert ext.relativePath == 'testRelPath'

    def test_copyPoints_handmadeFuncionWithFeatureNames(self):
        """ Test copyPoints() against handmade output for function copying with featureNames"""
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)

        ext = toTest.copyPoints(oneOrFour)
        exp = self.constructor([[1, 2, 3], [4, 5, 6]], featureNames=featureNames)
        assert ext.isIdentical(exp)
        expEnd = self.constructor(data, featureNames=featureNames)
        assert toTest.isIdentical(expEnd)

    @raises(ArgumentException)
    def test_copyPoints_exceptionStartInvalid(self):
        """ Test copyPoints() for ArgumentException when start is not a valid point index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.copyPoints(start=1.1, end=2)

    @raises(ArgumentException)
    def test_copyPoints_exceptionEndInvalid(self):
        """ Test copyPoints() for ArgumentException when start is not a valid Point index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.copyPoints(start=1, end=5)

    @raises(ArgumentException)
    def test_copyPoints_exceptionInversion(self):
        """ Test copyPoints() for ArgumentException when start comes after end """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.copyPoints(start=2, end=0)

    def test_copyPoints_handmadeRange(self):
        """ Test copyPoints() against handmade output for range copying """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.copyPoints(start=1, end=2)

        expectedRet = self.constructor([[4, 5, 6], [7, 8, 9]])
        expectedTest = self.constructor(data)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_copyPoints_range_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        # need to set source paths for view objects
        if isinstance(toTest, UML.data.BaseView):
            toTest._source._absPath = 'testAbsPath'
            toTest._source._relPath = 'testRelPath'
        else:
            toTest._absPath = 'testAbsPath'
            toTest._relPath = 'testRelPath'
        toTest._name = 'testName'

        ret = toTest.copyPoints(start=1, end=2)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

        assert ret.nameIsDefault()
        assert ret.absolutePath == 'testAbsPath'
        assert ret.relativePath == 'testRelPath'


    def test_copyPoints_rangeIntoPEmpty(self):
        """ Test copyPoints() copies all points using ranges """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints(start=0, end=2)

        assert ret.isIdentical(expRet)

        expTest = self.constructor(data)

        toTest.isIdentical(expTest)


    def test_copyPoints_handmadeRangeWithFeatureNames(self):
        """ Test copyPoints() against handmade output for range copying with featureNames """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints(start=1, end=2)

        expectedRet = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=['4', '7'], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)


    def test_copyPoints_handmadeRangeRand_FM(self):
        """ Test copyPoints() for correct sizes when using randomized range extraction and featureNames """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.copyPoints(start=0, end=2, number=2, randomize=True)

        assert ret.points == 2
        assert toTest.points == 3


    def test_copyPoints_handmadeRangeDefaults(self):
        """ Test copyPoints uses the correct defaults in the case of range based copy """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints(end=1)

        expectedRet = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=['1', '4'], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints(start=1)

        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expectedRet = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=['4', '7'], featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_copyPoints_handmade_calling_pointNames(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints(start='4', end='7')

        expectedRet = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_copyPoints_handmadeString(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('one=1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('one==1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('one<2')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('one<=1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('one>4')
        expectedRet = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('one>=7')
        expectedRet = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('one!=4')
        expectedRet = self.constructor([[1, 2, 3], [7, 8, 9]], pointNames=[pointNames[0], pointNames[-1]],
                                       featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('one<1')
        expectedRet = self.constructor([], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('one>0')
        expectedRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_copyPoints_handmadeStringWithOperatorWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('one = 1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('one == 1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('one < 2')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('one <= 1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('one > 4')
        expectedRet = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('one >= 7')
        expectedRet = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('one != 4')
        expectedRet = self.constructor([[1, 2, 3], [7, 8, 9]], pointNames=[pointNames[0], pointNames[-1]],
                                       featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('one < 1')
        expectedRet = self.constructor([], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('one > 0')
        expectedRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_copyPoints_handmadeStringWithFeatureWhitespace(self):
        featureNames = ["feature one", "feature two", "feature three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('feature one=1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName=value with operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('feature one = 1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_copyPoints_list_mixed(self):
        """ Test copyPoints() list input with mixed names and indices """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        names = ['1', '4', '7', '10']
        toTest = self.constructor(data, pointNames=names)
        ret = toTest.copyPoints(['1',1,-1])
        expRet = self.constructor([[1, 2, 3], [4, 5, 6], [10, 11, 12]], pointNames=['1','4','10'])
        expTest = self.constructor(data, pointNames=names)
        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(expTest)

    @raises(ArgumentException)
    def test_copyPoints_handmadeString_featureNotExist(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyPoints('four=1')

    def test_copyPoints_numberOnly(self):
        self.back_copy_numberOnly('point')

    def test_copyPoints_functionAndNumber(self):
        self.back_copy_functionAndNumber('point')

    def test_copyPoints_numberAndRandomizeAllData(self):
        self.back_copy_numberAndRandomizeAllData('point')

    def test_copyPoints_numberAndRandomizeSelectedData(self):
        self.back_copy_numberAndRandomizeSelectedData('point')

    @raises(ArgumentException)
    def test_copyPoints_randomizeNoNumber(self):
        self.back_structural_randomizeNoNumber('copy', 'point')

    @raises(ArgumentException)
    def test_copyPoints_list_numberGreaterThanTargeted(self):
        self.back_structural_list_numberGreaterThanTargeted('copy', 'point')

    @raises(ArgumentException)
    def test_copyPoints_function_numberGreaterThanTargeted(self):
        self.back_structural_function_numberGreaterThanTargeted('copy', 'point')

    @raises(ArgumentException)
    def test_copyPoints_range_numberGreaterThanTargeted(self):
        self.back_structural_range_numberGreaterThanTargeted('copy', 'point')

    #######################
    # copy common backend #
    #######################

    def back_copy_numberOnly(self, axis):
        if axis == 'point':
            toCall = "copyPoints"
        else:
            toCall = "copyFeatures"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        ret = getattr(toTest, toCall)(number=3)
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
            toCall = "copyPoints"
        else:
            toCall = "copyFeatures"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        ret = getattr(toTest, toCall)(allTrue, number=2)
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
            toCall = "copyPoints"
        else:
            toCall = "copyFeatures"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest1 = self.constructor(data, pointNames=pnames, featureNames=fnames)
        toTest2 = toTest1.copy()
        toTest3 = toTest1.copy()
        toTest4 = toTest1.copy()
        expTest = toTest1.copy()

        seed = UML.randomness.generateSubsidiarySeed()
        UML.randomness.startAlternateControl(seed)
        ret = getattr(toTest1, toCall)(number=3, randomize=True)
        UML.randomness.endAlternateControl()

        UML.randomness.startAlternateControl(seed)
        retList = getattr(toTest2, toCall)([0, 1, 2, 3], number=3, randomize=True)
        UML.randomness.endAlternateControl()

        UML.randomness.startAlternateControl(seed)
        retRange = getattr(toTest3, toCall)(start=0, end=3, number=3, randomize=True)
        UML.randomness.endAlternateControl()

        UML.randomness.startAlternateControl(seed)
        retFunc = getattr(toTest4, toCall)(allTrue, number=3, randomize=True)
        UML.randomness.endAlternateControl()

        if axis == 'point':
            assert ret.points == 3
        else:
            assert ret.features == 3

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
            toCall = "copyPoints"
        else:
            toCall = "copyFeatures"

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

        seed = UML.randomness.generateSubsidiarySeed()
        UML.randomness.startAlternateControl(seed)
        retList = getattr(toTest1, toCall)([1, 2], number=1, randomize=True)
        UML.randomness.endAlternateControl()

        UML.randomness.startAlternateControl(seed)
        retRange = getattr(toTest2, toCall)(start=1, end=2, number=1, randomize=True)
        UML.randomness.endAlternateControl()

        def middleRowsOrCols(value):
            return value[0] in [2, 4, 5, 7]

        UML.randomness.startAlternateControl(seed)
        retFunc = getattr(toTest3, toCall)(middleRowsOrCols, number=1, randomize=True)
        UML.randomness.endAlternateControl()

        assert retList.isIdentical(exp1) or retList.isIdentical(exp2)
        assert retRange.isIdentical(exp1) or retList.isIdentical(exp2)
        assert retFunc.isIdentical(exp1) or retList.isIdentical(exp2)

    #####################
    # copyFeatures #
    #####################

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_copyFeatures_calls_constructIndicesList(self, mockFunc):
        toTest = self.constructor([[1,2,],[3,4]], featureNames=['a', 'b'])

        ret = toTest.copyFeatures(['a', 'b'])

    def test_copyFeatures_handmadeSingle(self):
        """ Test copyFeatures() against handmade output when copying one feature """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ext1 = toTest.copyFeatures(0)
        exp1 = self.constructor([[1], [4], [7]])

        assert ext1.isIdentical(exp1)
        expEnd = self.constructor(data)
        assert toTest.isIdentical(expEnd)

    def test_copyFeatures_List_NamePath_Preserve(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        # need to set source paths for view objects
        if isinstance(toTest, UML.data.BaseView):
            toTest._source._absPath = 'testAbsPath'
            toTest._source._relPath = 'testRelPath'
        else:
            toTest._absPath = 'testAbsPath'
            toTest._relPath = 'testRelPath'
        toTest._name = 'testName'

        ext1 = toTest.copyFeatures(0)

        assert toTest.path == 'testAbsPath'
        assert toTest.absolutePath == 'testAbsPath'
        assert toTest.relativePath == 'testRelPath'

        assert ext1.nameIsDefault()
        assert ext1.absolutePath == 'testAbsPath'
        assert ext1.relativePath == 'testRelPath'

    def test_copyFeatures_ListIntoFEmpty(self):
        """ Test copyFeatures() by copying a list of all features """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        expRet = self.constructor(data)
        ret = toTest.copyFeatures([0, 1, 2])

        assert ret.isIdentical(expRet)
        expEnd = self.constructor(data)
        assert toTest.isIdentical(expEnd)

    def test_copyFeatures_ListIntoFEmptyOutOfOrder(self):
        """ Test copyFeatures() by copying a list of all features """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        expData = [[3, 1, 2], [6, 4, 5], [9, 7, 8], [12, 10, 11]]
        expRet = self.constructor(expData)
        ret = toTest.copyFeatures([2, 0, 1])

        assert ret.isIdentical(expRet)
        expEnd = self.constructor(data)
        assert toTest.isIdentical(expEnd)


    def test_copyFeatures_handmadeListSequence(self):
        """ Test copyFeatures() against handmade output for several copies by list """
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data, pointNames=pointNames)
        ext1 = toTest.copyFeatures([0])
        exp1 = self.constructor([[1], [4], [7]], pointNames=pointNames)
        assert ext1.isIdentical(exp1)
        ext2 = toTest.copyFeatures([3, 2])
        exp2 = self.constructor([[-1, 3], [-2, 6], [-3, 9]], pointNames=pointNames)
        assert ext2.isIdentical(exp2)
        expEnd = self.constructor(data, pointNames=pointNames)
        assert toTest.isIdentical(expEnd)

    def test_copyFeatures_handmadeListWithFeatureName(self):
        """ Test copyFeatures() against handmade output for list copies when specifying featureNames """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        toTest = self.constructor(data, featureNames=featureNames)
        ext1 = toTest.copyFeatures(["one"])
        exp1 = self.constructor([[1], [4], [7]], featureNames=["one"])
        assert ext1.isIdentical(exp1)
        ext2 = toTest.copyFeatures(["three", "neg"])
        exp2 = self.constructor([[3, -1], [6, -2], [9, -3]], featureNames=["three", "neg"])
        assert ext2.isIdentical(exp2)
        expEnd = self.constructor(data, featureNames=featureNames)
        assert toTest.isIdentical(expEnd)


    def test_copyFeatures_List_trickyOrdering(self):
        data = [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
        toExtract = [6, 5, 3, 9]
        #		toExtract = [3,5,6,9]

        toTest = self.constructor(data)

        ret = toTest.copyFeatures(toExtract)

        expRaw = [0, 0, 1, 0]
        expRet = self.constructor(expRaw)

        expRem = self.constructor(data)

        assert ret == expRet
        assert toTest == expRem

    def test_copyFeatures_List_reorderingWithFeatureNames(self):
        data = [[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12]]
        fnames = ['a', 'b', 'c', 'd']
        test = self.constructor(data, featureNames=fnames)

        expRetRaw = [[1, 3, 2], [4, 6, 5], [7, 9, 8]]
        expRetNames = ['a', 'c', 'b']
        expRet = self.constructor(expRetRaw, featureNames=expRetNames)

        expTestRaw = [[10], [11], [12]]
        expTestNames = ['d']
        expTest = self.constructor(data, featureNames=fnames)

        ret = test.copyFeatures(expRetNames)
        assert ret == expRet
        assert test == expTest


    def test_copyFeatures_function_selectionGap(self):
        data = [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
        fnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        copyIndices = [3, 5, 6, 9]

        def sel(feature):
            if int(feature.getFeatureName(0)) in copyIndices:
                return True
            else:
                return False

        toTest = self.constructor(data, featureNames=fnames)

        ret = toTest.copyFeatures(sel)

        expRaw = [1, 0, 0, 0]
        expNames = ['3', '5', '6', '9']
        expRet = self.constructor(expRaw, featureNames=expNames)

        expRaw = [0, 1, 1, 0, 0, 1]
        expNames = ['0', '1', '2', '4', '7', '8']
        expRem = self.constructor(data, featureNames=fnames)

        assert ret == expRet
        assert toTest == expRem


    def test_copyFeatures_functionIntoFEmpty(self):
        """ Test copyFeatures() by copying all featuress using a function """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        expRet = self.constructor(data)

        ret = toTest.copyFeatures(allTrue)
        assert ret.isIdentical(expRet)
        expEnd = self.constructor(data)
        assert toTest.isIdentical(expEnd)

    def test_copyFeatures_function_returnPointEmpty(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        exp = self.constructor(data)

        ret = toTest.copyFeatures(allFalse)
        expRet = self.constructor([[],[],[]])
        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(exp)


    def test_copyFeatures_handmadeFunction(self):
        """ Test copyFeatures() against handmade output for function copies """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data)

        ext = toTest.copyFeatures(absoluteOne)
        exp = self.constructor([[1, -1], [4, -2], [7, -3]])
        assert ext.isIdentical(exp)
        expEnd = self.constructor(data)
        assert toTest.isIdentical(expEnd)


    def test_copyFeatures_func_NamePath_preservation(self):
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data)

        # need to set source paths for view objects
        if isinstance(toTest, UML.data.BaseView):
            toTest._source._absPath = 'testAbsPath'
            toTest._source._relPath = 'testRelPath'
        else:
            toTest._absPath = 'testAbsPath'
            toTest._relPath = 'testRelPath'
        toTest._name = 'testName'

        ext = toTest.copyFeatures(absoluteOne)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

        assert ext.nameIsDefault()
        assert ext.absolutePath == 'testAbsPath'
        assert ext.relativePath == 'testRelPath'

    def test_copyFeatures_handmadeFunctionWithFeatureName(self):
        """ Test copyFeatures() against handmade output for function copies with featureNames """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        pointNames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        ext = toTest.copyFeatures(absoluteOne)
        exp = self.constructor([[1, -1], [4, -2], [7, -3]], pointNames=pointNames, featureNames=['one', 'neg'])
        assert ext.isIdentical(exp)
        expEnd = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert toTest.isIdentical(expEnd)

    @raises(ArgumentException)
    def test_copyFeatures_exceptionStartInvalid(self):
        """ Test copyFeatures() for ArgumentException when start is not a valid feature index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.copyFeatures(start=1.1, end=2)

    @raises(ArgumentException)
    def test_copyFeatures_exceptionStartInvalidFeatureName(self):
        """ Test copyFeatures() for ArgumentException when start is not a valid feature FeatureName """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.copyFeatures(start="wrong", end=2)

    @raises(ArgumentException)
    def test_copyFeatures_exceptionEndInvalid(self):
        """ Test copyFeatures() for ArgumentException when start is not a valid feature index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.copyFeatures(start=0, end=5)

    @raises(ArgumentException)
    def test_copyFeatures_exceptionEndInvalidFeatureName(self):
        """ Test copyFeatures() for ArgumentException when start is not a valid featureName """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.copyFeatures(start="two", end="five")

    @raises(ArgumentException)
    def test_copyFeatures_exceptionInversion(self):
        """ Test copyFeatures() for ArgumentException when start comes after end """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.copyFeatures(start=2, end=0)

    @raises(ArgumentException)
    def test_copyFeatures_exceptionInversionFeatureName(self):
        """ Test copyFeatures() for ArgumentException when start comes after end as FeatureNames"""
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.copyFeatures(start="two", end="one")


    def test_copyFeatures_rangeIntoFEmpty(self):
        """ Test copyFeatures() copies all Featuress using ranges """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        expRet = self.constructor(data, featureNames=featureNames)
        ret = toTest.copyFeatures(start=0, end=2)

        assert ret.isIdentical(expRet)
        exp = self.constructor(data, featureNames=featureNames)
        toTest.isIdentical(exp)

    def test_copyFeatures_handmadeRange(self):
        """ Test copyFeatures() against handmade output for range copies """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.copyFeatures(start=1, end=2)

        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]])
        expectedTest = self.constructor(data)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_copyFeatures_range_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        # need to set source paths for view objects
        if isinstance(toTest, UML.data.BaseView):
            toTest._source._absPath = 'testAbsPath'
            toTest._source._relPath = 'testRelPath'
        else:
            toTest._absPath = 'testAbsPath'
            toTest._relPath = 'testRelPath'
        toTest._name = 'testName'

        ret = toTest.copyFeatures(start=1, end=2)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

        assert ret.nameIsDefault()
        assert ret.absolutePath == 'testAbsPath'
        assert ret.relativePath == 'testRelPath'


    def test_copyFeatures_handmadeWithFeatureNames(self):
        """ Test copyFeatures() against handmade output for range copies with FeatureNames """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures(start=1, end=2)

        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=["two", "three"])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_copyFeatures_handmade_calling_featureNames(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures(start="two", end="three")

        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=["two", "three"])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_copyFeatures_handmadeString(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['p1', 'p2', 'p3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('p2=5')
        expectedRet = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('p1==1')
        expectedRet = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('p3<9')
        expectedRet = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('p3<=8')
        expectedRet = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('p3>8')
        expectedRet = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('p3>8.5')
        expectedRet = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('p1!=1.0')
        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('p1<1')
        expectedRet = self.constructor([], pointNames=pointNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('p1>0')
        expectedRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_copyFeatures_handmadeStringWithOperatorWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['p1', 'p2', 'p3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('p2 = 5')
        expectedRet = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('p1 == 1')
        expectedRet = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('p3 < 9')
        expectedRet = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('p3 <= 8')
        expectedRet = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('p3 > 8')
        expectedRet = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('p3 > 8.5')
        expectedRet = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('p1 != 1.0')
        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('p1 < 1')
        expectedRet = self.constructor([], pointNames=pointNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('p1 > 0')
        expectedRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_copyFeatures_handmadeStringWithPointWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['pt 1', 'pt 2', 'pt 3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value with no operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('pt 2=5')
        expectedRet = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test pointName=value with operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('pt 2 = 5')
        expectedRet = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_copyFeatures_list_mixed(self):
        """ Test copyFeatures() list input with mixed names and indices """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.copyFeatures([1, "three", -1])
        expRet = self.constructor([[2, 3, -1], [5, 6, -2], [8, 9, -3]], featureNames=["two", "three", "neg"])
        expTest = self.constructor(data, featureNames=featureNames)
        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(expTest)

    @raises(ArgumentException)
    def test_copyFeatures_handmadeString_pointNotExist(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.copyFeatures('5=1')

    def test_copyFeatures_numberOnly(self):
        self.back_copy_numberOnly('feature')

    def test_copyFeatures_functionAndNumber(self):
        self.back_copy_functionAndNumber('feature')

    def test_copyFeatures_numberAndRandomizeAllData(self):
        self.back_copy_numberAndRandomizeAllData('feature')

    def test_copyFeatures_numberAndRandomizeSelectedData(self):
        self.back_copy_numberAndRandomizeSelectedData('feature')

    @raises(ArgumentException)
    def test_copyFeatures_randomizeNoNumber(self):
        self.back_structural_randomizeNoNumber('copy', 'feature')

    @raises(ArgumentException)
    def test_copyFeatures_list_numberGreaterThanTargeted(self):
        self.back_structural_list_numberGreaterThanTargeted('copy', 'feature')

    @raises(ArgumentException)
    def test_copyFeatures_function_numberGreaterThanTargeted(self):
        self.back_structural_function_numberGreaterThanTargeted('copy', 'feature')

    @raises(ArgumentException)
    def test_copyFeatures_range_numberGreaterThanTargeted(self):
        self.back_structural_range_numberGreaterThanTargeted('copy', 'feature')


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
        orig3.sortFeatures(sortBy=orig3.getPointName(0))
        orig10 = self.constructor([{'a':1, 'b':2, 'c':3}])
        orig10.sortFeatures(sortBy=orig10.getPointName(0))
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
        orig3.sortFeatures(sortBy=orig3.getPointName(0))
        orig7 = self.constructor([{'a':1, 'b':2, 'c':'a'}, {'a':3, 'b':4, 'c':'b'}], elementType=object)
        orig7.sortFeatures(sortBy=orig7.getPointName(0))
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
        tmpCSV = tempfile.NamedTemporaryFile(mode='w', suffix=".csv")
        tmpCSV.write("1,2,3\n")
        tmpCSV.flush()
        fromCSV = self.constructor(data=tmpCSV.name)

        # instantiate from mtx array file
        tmpMTXArr = tempfile.NamedTemporaryFile(mode='w', suffix=".mtx")
        tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
        tmpMTXArr.write("1 3\n")
        tmpMTXArr.write("1\n")
        tmpMTXArr.write("2\n")
        tmpMTXArr.write("3\n")
        tmpMTXArr.flush()
        fromMTXArr = self.constructor(data=tmpMTXArr.name)

        # instantiate from mtx coordinate file
        tmpMTXCoo = tempfile.NamedTemporaryFile(mode='w', suffix=".mtx")
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
        tmpCSV = tempfile.NamedTemporaryFile(mode='w', suffix=".csv")
        tmpCSV.write("\n")
        tmpCSV.write("\n")
        tmpCSV.write("point_names,one,two,three\n")
        tmpCSV.write("1P,1,2,3\n")
        tmpCSV.flush()
        fromCSV = self.constructor(data=tmpCSV.name)

        # instantiate from mtx file
        tmpMTXArr = tempfile.NamedTemporaryFile(mode='w', suffix=".mtx")
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
        tmpMTXCoo = tempfile.NamedTemporaryFile(mode='w', suffix=".mtx")
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


    @raises(ArgumentException, TypeError)
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
    @mock.patch('UML.data.base.valuesToPythonList', side_effect=calledException)
    def test_init_pointNames_calls_valuesToPythonList(self, mockFunc):
        self.constructor([1,2,3], pointNames=['one'])

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.valuesToPythonList', side_effect=calledException)
    def test_init_featureNames_calls_valuesToPythonList(self, mockFunc):
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


    def test_transpose_handmade(self):
        """ Test transpose() function against handmade output """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        dataTrans = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

        dataObj1 = self.constructor(deepcopy(data))
        dataObj2 = self.constructor(deepcopy(data))
        dataObjT = self.constructor(deepcopy(dataTrans))

        ret1 = dataObj1.transpose() # RET CHECK
        assert dataObj1.isIdentical(dataObjT)
        assert ret1 is None
        dataObj1.transpose()
        dataObjT.transpose()
        assert dataObj1.isIdentical(dataObj2)
        assert dataObj2.isIdentical(dataObjT)

    def test_transpose_handmadeWithZeros(self):
        """ Test transpose() function against handmade output """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0], [11, 12, 13]]
        dataTrans = [[1, 4, 7, 0, 11], [2, 5, 8, 0, 12], [3, 6, 9, 0, 13]]

        dataObj1 = self.constructor(deepcopy(data))
        dataObj2 = self.constructor(deepcopy(data))
        dataObjT = self.constructor(deepcopy(dataTrans))

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

        dataObj1 = self.constructor(deepcopy(data), pointNames=origPointNames,
                                                    featureNames=origFeatureNames)
        dataObj2 = self.constructor(deepcopy(data), pointNames=origPointNames,
                                                    featureNames=origFeatureNames)
        dataObjT = self.constructor(deepcopy(dataTrans), pointNames=transPointNames,
                                                         featureNames=transFeatureNames)
        dataObj1.transpose()
        assert dataObj1.getPointNames() == transPointNames
        assert dataObj1.getFeatureNames() == transFeatureNames
        assert dataObj1.isIdentical(dataObjT)

        dataObj1.transpose()
        dataObjT.transpose()
        assert dataObj1.getPointNames() == dataObj2.getPointNames()
        assert dataObj1.getFeatureNames() == dataObj2.getFeatureNames()
        assert dataObj1.isIdentical(dataObj2)

        assert dataObj2.getPointNames() == dataObjT.getPointNames()
        assert dataObj2.getFeatureNames() == dataObjT.getFeatureNames()
        assert dataObj2.isIdentical(dataObjT)

    def test_transpose_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0], [11, 12, 13]]

        dataObj1 = self.constructor(deepcopy(data))

        dataObj1._name = "TestName"
        dataObj1._absPath = "TestAbsPath"
        dataObj1._relPath = "testRelPath"

        dataObj1.transpose()

        assert dataObj1.name == "TestName"
        assert dataObj1.absolutePath == "TestAbsPath"
        assert dataObj1.relativePath == 'testRelPath'


    #####################################
    # addPoints() / addFeatures() #
    #####################################

    def backend_insert_exceptionNone(self, axis):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        if axis == 'point':
            toTest.addPoints(None)
        else:
            toTest.addFeatures(None)

    @raises(ArgumentException)
    def test_addPoints_exceptionNone(self):
        """ Test addPoints() for ArgumentException when toInsert is None"""
        self.backend_insert_exceptionNone('point')

    @raises(ArgumentException)
    def test_addFeatures_exceptionNone(self):
        """ Test addFeatures() for ArgumentException when toInsert is None """
        self.backend_insert_exceptionNone('feature')


    def backend_insert_exceptionWrongSize(self, axis):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toInsert = self.constructor([[2, 3, 4, 5, 6]])

        if axis == 'point':
            toTest.addPoints(toInsert)
        else:
            toInsert.transpose()
            toTest.addFeatures(toInsert)

    @raises(ArgumentException)
    def test_addPoints_exceptionWrongSize(self):
        """ Test addPoints() for ArgumentException when toInsert has too many features """
        self.backend_insert_exceptionWrongSize('point')

    @raises(ArgumentException)
    def test_addFeatures_exceptionWrongSize(self):
        """ Test addFeatures() for ArgumentException when toInsert has too many points """
        self.backend_insert_exceptionWrongSize('feature')


    def backend_insert_exception_extendAxis_SameName(self, axis):
        toTest1 = self.constructor([[1, 2]], pointNames=["hello"])
        toTest2 = self.constructor([[1, 2], [5, 6]], pointNames=["hello", "goodbye"])

        if axis == 'point':
            toTest2.addPoints(toTest1)
        else:
            toTest1.transpose()
            toTest2.transpose()
            toTest2.addFeatures(toTest1)

    @raises(ArgumentException)
    def test_addPoints_exceptionSamePointName(self):
        """ Test addPoints() for ArgumentException when toInsert and self have a pointName in common """
        self.backend_insert_exception_extendAxis_SameName('point')

    @raises(ArgumentException)
    def test_addFeatures_exceptionSameFeatureName(self):
        """ Test addFeatures() for ArgumentException when toInsert and self have a featureName in common """
        self.backend_insert_exception_extendAxis_SameName('feature')


    def backend_insert_exception_sharedAxis_unsharedName(self, axis):
        toTest1 = self.constructor([[1, 2]], featureNames=['1', '2'])
        toTest2 = self.constructor([[2, 1], [6, 5]], featureNames=['6', '1'])

        if axis == 'point':
            toTest2.addPoints(toTest1)
        else:
            toTest1.transpose()
            toTest2.transpose()
            toTest2.addFeatures(toTest1)

    @raises(ArgumentException)
    def test_addPoints_exception_unsharedFeatureName(self):
        """ Test addPoints() for ArgumentException when toInsert and self have a featureName not in common """
        self.backend_insert_exception_sharedAxis_unsharedName('point')

    @raises(ArgumentException)
    def test_addFeatures_exception_unsharedPointName(self):
        """ Test addFeatures() for ArgumentException when toInsert and self have a pointName not in common """
        self.backend_insert_exception_sharedAxis_unsharedName('feature')


    def backend_insert_exceptionNonUMLDataType(self, axis):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        if axis == 'point':
            toTest.addPoints([[1, 1, 1]])
        else:
            toTest.addFeatures([[1], [1], [1]])

    @raises(ArgumentException)
    def test_addPoints_exceptionNonUMLDataType(self):
        self.backend_insert_exceptionNonUMLDataType('point')

    @raises(ArgumentException)
    def test_addFeatures_exceptionNonUMLDataType(self):
        self.backend_insert_exceptionNonUMLDataType('feature')


    def backend_insert_exception_outOfOrder_with_defaults(self, axis):
        toTest1 = self.constructor([[1, 2, 3]])
        toTest2 = self.constructor([[1, 3, 2]])

        toTest1.setFeatureName(1, '2')
        toTest1.setFeatureName(2, '3')
        toTest2.setFeatureName(1, '3')
        toTest2.setFeatureName(2, '2')

        if axis == 'point':
            toTest1.addPoints(toTest2)
        else:
            toTest1.transpose()
            toTest2.transpose()
            toTest1.addFeatures(toTest2)


    @raises(ArgumentException)
    def test_addPoints_exception_outOfOrder_with_defaults(self):
        """ Test addPoints() for ArgumentException when toInsert and self contain a mix of set names and default names not in the same order"""
        self.backend_insert_exception_outOfOrder_with_defaults('point')

    @raises(ArgumentException)
    def test_addFeatures_exception_outOfOrder_with_defaults(self):
        """ Test addFeatures() for ArgumentException when toInsert and self contain a mix of set names and default names not in the same order"""
        self.backend_insert_exception_outOfOrder_with_defaults('feature')


    def backend_insert_emptyObject(self, axis, insertBefore=None):
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
            toTest.addPoints(toInsert, insertBefore)
        else:
            toTest.addFeatures(toInsert, insertBefore)

        assert toTest.isIdentical(toExp)

    def test_addPoints_fromEmpty_bottom(self):
        """ Test addPoints() to bottom when the calling object is point empty """
        self.backend_insert_emptyObject('point')

    def test_addFeatures_fromEmpty_right(self):
        """ Test addFeatures() to right when the calling object is feature empty """
        self.backend_insert_emptyObject('feature')

    @raises(ArgumentException)
    def test_addPoints_fromEmpty_top(self):
        """ Test addPoints() with an appendBefore ID when the calling object is point empty raises exception """
        self.backend_insert_emptyObject('point', 0)

    @raises(ArgumentException)
    def test_addFeatures_fromEmpty_left(self):
        """ Test addFeatures() with an appendBefore ID when the calling object is feature empty raises exception """
        self.backend_insert_emptyObject('feature', 0)


    def backend_insert_handmadeSingle(self, axis, insertBefore=None):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        offNames = ['o1', 'o2', 'o3']
        names = ['one', 'two', 'three']
        addName = ['new']

        if axis == 'point':
            if insertBefore is None:
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
            ret = toTest.addPoints(toInsert, insertBefore)  # RET CHECK
        else:
            if insertBefore is None:
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
            ret = toTest.addFeatures(toInsert, insertBefore)  # RET CHECK

        assert toTest.isIdentical(expected)
        assert ret is None

    def test_addPoints_handmadeSingle_bottom(self):
        """ Test addPoints() against handmade output for a single added point to the bottom"""
        self.backend_insert_handmadeSingle('point')

    def test_addFeatures_handmadeSingle_right(self):
        """ Test addFeatures() against handmade output for a single added feature to the right"""
        self.backend_insert_handmadeSingle('feature')

    def test_addPoints_handmadeSingle_top(self):
        """ Test addPoints() against handmade output for a single added point the the top"""
        self.backend_insert_handmadeSingle('point', 0)

    def test_addFeatures_handmadeSingle_left(self):
        """ Test addFeatures() against handmade output for a single added feature to the left"""
        self.backend_insert_handmadeSingle('feature', 0)

    def test_addPoints_handmadeSingle_mid(self):
        """ Test addPoints() against handmade output for a single added point in the middle"""
        self.backend_insert_handmadeSingle('point', 1)

    def test_addFeatures_handmadeSingle_mid(self):
        """ Test addFeatures() against handmade output for a single added feature in the middle"""
        self.backend_insert_handmadeSingle('feature', 1)


    def backend_insert_handmadeSequence(self, axis, insertBefore=None):
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
            toTest = self.constructor(data, pointNames=names, featureNames=offNames)
            for nextAdd in toInsert.pointIterator():
                toTest.addPoints(nextAdd, insertBefore)
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
            toTest = self.constructor(data, pointNames=offNames, featureNames=names)
            toInsert.transpose()
            for nextAdd in toInsert.featureIterator():
                toTest.addFeatures(nextAdd, insertBefore)
            expected = self.constructor(dataExpected, pointNames=offNames, featureNames=namesExp)

        assert toTest.isIdentical(expected)

    def test_addPoints_handmadeSequence_bottom(self):
        """ Test addPoints() against handmade output for a sequence of additions to the bottom"""
        self.backend_insert_handmadeSequence('point')

    def test_addFeatures_handmadeSequence_right(self):
        """ Test addFeatures() against handmade output for a sequence of additions to the right"""
        self.backend_insert_handmadeSequence('feature')

    def test_addPoints_handmadeSequence_top(self):
        """ Test addPoints() against handmade output for a sequence of additions to the top"""
        self.backend_insert_handmadeSequence('point', 0)

    def test_addFeatures_handmadeSequence_left(self):
        """ Test addFeatures() against handmade output for a sequence of additions to the left"""
        self.backend_insert_handmadeSequence('feature', 0)

    def test_addPoints_handmadeSequence_mid(self):
        """ Test addPoints() against handmade output for a sequence of additions in the middle"""
        self.backend_insert_handmadeSequence('point', 1)

    def test_addFeatures_handmadeSequence_mid(self):
        """ Test addFeatures() against handmade output for a sequence of additions in the middle"""
        self.backend_insert_handmadeSequence('feature', 1)


    def backend_insert_selfInsert(self, axis, insertBefore=None):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        names = ['one', 'two', 'three']

        if axis == 'point':
            orig = self.constructor(data, featureNames=names)
        else:
            orig = self.constructor(data, pointNames=names)

        dup = orig.copy()

        if axis == 'point':
            dupNames = dup.getPointNames()
            assert orig.getPointNames() == dupNames

            orig.addPoints(orig, insertBefore)

            if insertBefore is None or insertBefore == 0:
                dataExp = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]]
            elif insertBefore == 1:
                dataExp = [[1, 2, 3], [1, 2, 3], [4, 5, 6], [7, 8, 9], [4, 5, 6], [7, 8, 9]]
            expected = self.constructor(dataExp, featureNames=names)
        else:
            dupNames = dup.getFeatureNames()
            assert orig.getFeatureNames() == dupNames

            orig.addFeatures(orig, insertBefore)

            if insertBefore is None or insertBefore == 0:
                dataExp = [[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6], [7, 8, 9, 7, 8, 9]]
            elif insertBefore == 1:
                dataExp = [[1, 1, 2, 3, 2, 3], [4, 4, 5, 6, 5, 6], [7, 7, 8, 9, 8, 9]]
            expected = self.constructor(dataExp, pointNames=names)

        assert orig == expected

        checkNames = orig.getPointNames() if axis == 'point' else orig.getFeatureNames()
        lastDefIndex = int(dupNames[2][-1])
        if insertBefore is None:
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

    def test_addPoints_selfInsert_bottom(self):
        self.backend_insert_selfInsert('point')

    def test_addFeatures_selfInsert_right(self):
        self.backend_insert_selfInsert('feature')

    def test_addPoints_selfInsert_top(self):
        self.backend_insert_selfInsert('point', 0)

    def test_addFeatures_selfInsert_left(self):
        self.backend_insert_selfInsert('feature', 0)

    def test_addPoints_selfInsert_mid(self):
        self.backend_insert_selfInsert('point', 1)

    def test_addFeatures_selfInsert_mid(self):
        self.backend_insert_selfInsert('feature', 1)


    def backend_insert_automaticReorder(self, axis, defPrimaryNames, insertBefore=None):
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
            if insertBefore is None:
                namesExp = ['one', 'two', 'three', 'new']
            elif insertBefore == 0:
                namesExp = ['new', 'one', 'two', 'three']
            elif insertBefore == 1:
                namesExp = ['one', 'new', 'two', 'three']

        if axis == 'point':
            toAddData = [[-3, -2, -1]]
            if insertBefore is None:
                dataExpected = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]]
            elif insertBefore == 0:
                dataExpected = [[-1, -2, -3], [1, 2, 3], [4, 5, 6], [7, 8, 9]]
            elif insertBefore == 1:
                dataExpected = [[1, 2, 3], [-1, -2, -3], [4, 5, 6], [7, 8, 9]]
            toTest = self.constructor(data, pointNames=names, featureNames=offNames)
            toInsert = self.constructor(toAddData, pointNames=addName, featureNames=addOffName)
            expInsert = toInsert.copy()
            expected = self.constructor(dataExpected, pointNames=namesExp, featureNames=offNames)
            toTest.addPoints(toInsert, insertBefore)
        else:
            toAddData = [[-3], [-2], [-1]]
            if insertBefore is None:
                dataExpected = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
            elif insertBefore == 0:
                dataExpected = [[-1, 1, 2, 3], [-2, 4, 5, 6], [-3, 7, 8, 9]]
            elif insertBefore == 1:
                dataExpected = [[1, -1, 2, 3], [4, -2, 5, 6], [7, -3, 8, 9]]
            toTest = self.constructor(data, pointNames=offNames, featureNames=names)
            toInsert = self.constructor(toAddData, pointNames=addOffName, featureNames=addName)
            expInsert = toInsert.copy()
            expected = self.constructor(dataExpected, pointNames=offNames, featureNames=namesExp)
            toTest.addFeatures(toInsert, insertBefore)

        # check that toInsert object was not modified when reordering occurred
        assert toInsert.isIdentical(expInsert)
        assert toTest.isIdentical(expected)


    def test_addPoints_automaticReorder_fullySpecifiedNames_bottom(self):
        self.backend_insert_automaticReorder('point', False)

    def test_addFeatures_automaticReorder_fullySpecifiedNames_right(self):
        self.backend_insert_automaticReorder('feature', False)

    def test_addPoints_automaticReorder_defaultPointNames_bottom(self):
        self.backend_insert_automaticReorder('point', True)

    def test_addFeatures_automaticReorder_defaultFeatureNames_right(self):
        self.backend_insert_automaticReorder('feature', True)

    def test_addPoints_automaticReorder_fullySpecifiedNames_top(self):
        self.backend_insert_automaticReorder('point', False, 0)

    def test_addFeatures_automaticReorder_fullySpecifiedNames_left(self):
        self.backend_insert_automaticReorder('feature', False, 0)

    def test_addPoints_automaticReorder_defaultPointNames_top(self):
        self.backend_insert_automaticReorder('point', True, 0)

    def test_addFeatures_automaticReorder_defaultFeatureNames_left(self):
        self.backend_insert_automaticReorder('feature', True, 0)

    def test_addPoints_automaticReorder_fullySpecifiedNames_mid(self):
        self.backend_insert_automaticReorder('point', False, 1)

    def test_addFeatures_automaticReorder_fullySpecifiedNames_mid(self):
        self.backend_insert_automaticReorder('feature', False, 1)

    def test_addPoints_automaticReorder_defaultPointNames_mid(self):
        self.backend_insert_automaticReorder('point', True, 1)

    def test_addFeatures_automaticReorder_defaultFeatureNames_mid(self):
        self.backend_insert_automaticReorder('feature', True, 1)

    def backend_insert_allPossibleUMLDataType(self, axis):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        currType = self.constructor([]).getTypeString()
        availableTypes = UML.data.available
        otherTypes = [retType for retType in availableTypes if retType != currType]
        inserted = []
        for other in otherTypes:
            toTest = self.constructor(data)
            if axis == 'point':
                insertData = [[-1, -2, -3]]
                otherTest = UML.createData(other, insertData)
                exp = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
                toTest.addPoints(otherTest)
                inserted.append(toTest)
            else:
                insertData = [[-1], [-2], [-3]]
                otherTest = UML.createData(other, insertData)
                exp = self.constructor([[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]])
                toTest.addFeatures(otherTest)
                inserted.append(toTest)

        assert all(exp == obj for obj in inserted)

    def test_addPoints_allPossibleUMLDataType(self):
        self.backend_insert_allPossibleUMLDataType('point')

    def test_addFeatures_allPossibleUMLDataType(self):
        self.backend_insert_allPossibleUMLDataType('feature')


    def backend_insert_noReorderWithAllDefaultNames(self, axis):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        if axis == 'point':
            insertData = [[-1, -2, -3]]
            # assign names to be the reverse of toTest's default names
            fNames = list(reversed(toTest.getFeatureNames()))
            toInsert = self.constructor(insertData, featureNames=fNames)
            assert toTest.getFeatureNames() != toInsert.getFeatureNames()

            exp = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
            toTest.addPoints(toInsert)

        else:
            insertData = [[-1], [-2], [-3]]
            # assign names to be the reverse of toTest's default names
            pNames = list(reversed(toTest.getPointNames()))
            toInsert = self.constructor(insertData, pointNames=pNames)
            assert toTest.getPointNames() != toInsert.getPointNames()

            exp = self.constructor([[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]])
            toTest.addFeatures(toInsert)

        assert toTest == exp

    def test_addPoints_noReorderWithAllDefaultNames(self):
        self.backend_insert_noReorderWithAllDefaultNames('point')

    def test_addFeatures_noReorderWithAllDefaultNames(self):
        self.backend_insert_noReorderWithAllDefaultNames('feature')

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
            toTest.addPoints(toInsert)
        else:
            toTest.addFeatures(toInsert)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'

    def test_addPoints_NamePath_preservation(self):
        self.backend_insert_NamePath_preservation('point')

    def test_addFeatures_NamePath_preservation(self):
        self.backend_insert_NamePath_preservation('feature')

    def test_addPoints_noNamesCreated(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toInsert = self.constructor([[-1, -2, -3]])
        toTest.addPoints(toInsert)

        assert not toTest._pointNamesCreated()
        assert not toTest._featureNamesCreated()

    def test_addFeatures_noNamesCreated(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toInsert = self.constructor([[-1], [-2], [-3]])
        toTest.addFeatures(toInsert)

        assert not toTest._featureNamesCreated()
        assert not toTest._pointNamesCreated()


    ##############
    # sortPoints() #
    ##############

    @raises(ArgumentException)
    def test_sortPoints_exceptionAtLeastOne(self):
        """ Test sortPoints() has at least one paramater """
        data = [[7, 8, 9], [1, 2, 3], [4, 5, 6]]
        toTest = self.constructor(data)

        toTest.sortPoints()

    @raises(ArgumentException)
    def test_sortPoints_exceptionBothNotNone(self):
        """ Test sortPoints() has only one parameter """
        data = [[7, 8, 9], [1, 2, 3], [4, 5, 6]]
        toTest = self.constructor(data)

        toTest.sortPoints(sortBy=1, sortHelper=[1,2,0])

    def test_sortPoints_naturalByFeature(self):
        """ Test sortPoints() when we specify a feature to sort by """
        data = [[1, 2, 3], [7, 1, 9], [4, 5, 6]]
        names = ['1', '7', '4']
        toTest = self.constructor(data, pointNames=names)

        ret = toTest.sortPoints(sortBy=1) # RET CHECK

        dataExpected = [[7, 1, 9], [1, 2, 3], [4, 5, 6]]
        namesExp = ['7', '1', '4']
        objExp = self.constructor(dataExpected, pointNames=namesExp)

        assert toTest.isIdentical(objExp)
        assert ret is None

    def test_sortPoints_naturalByFeatureName(self):
        """ Test sortPoints() when we specify a feature name to sort by """
        data = [[1, 2, 3], [7, 1, 9], [4, 5, 6]]
        pnames = ['1', '7', '4']
        fnames = ['1', '2', '3']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)

        ret = toTest.sortPoints(sortBy='2') # RET CHECK

        dataExpected = [[7, 1, 9], [1, 2, 3], [4, 5, 6]]
        namesExp = ['7', '1', '4']
        objExp = self.constructor(dataExpected, pointNames=namesExp, featureNames=fnames)

        assert toTest.isIdentical(objExp)
        assert ret is None


    def test_sortPoints_scorer(self):
        """ Test sortPoints() when we specify a scoring function """
        data = [[1, 2, 3], [4, 5, 6], [7, 1, 9], [0, 0, 0]]
        toTest = self.constructor(data)

        def numOdds(point):
            assert isinstance(point, BaseView)
            ret = 0
            for val in point:
                if val % 2 != 0:
                    ret += 1
            return ret

        toTest.sortPoints(sortHelper=numOdds)

        dataExpected = [[0, 0, 0], [4, 5, 6], [1, 2, 3], [7, 1, 9]]
        objExp = self.constructor(dataExpected)

        assert toTest.isIdentical(objExp)

    def test_sortPoints_comparator(self):
        """ Test sortPoints() when we specify a comparator function """
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

        toTest.sortPoints(sortHelper=compOdds)

        dataExpected = [[0, 0, 0], [4, 5, 6], [1, 2, 3], [7, 1, 9]]
        objExp = self.constructor(dataExpected)

        assert toTest.isIdentical(objExp)

    def test_sortPoints_dataTypeRetainedFromList(self):
        """ Test sortPoints() data not converted when sorting by list"""
        data = [['a', 2, 3.0], ['b', 5, 6.0], ['c', 8, 9.0]]
        toTest = self.constructor(data)

        toTest.sortPoints(sortHelper=[2, 1, 0])

        expData = [['c', 8, 9.0], ['b', 5, 6.0], ['a', 2, 3.0]]
        exp = self.constructor(expData)

        assert toTest == exp

    def test_sortPoints_indicesList(self):
        """ Test sortPoints() when we specify a list of indices """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        toTest = self.constructor(data)

        toTest.sortPoints(sortHelper=[2, 1, 0])

        expData = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        exp = self.constructor(expData)

        assert toTest == exp

    def test_sortPoints_namesList(self):
        """ Test sortPoints() when we specify a list of point names """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        pnames = ['3', '6', '9']
        toTest = self.constructor(data, pointNames=pnames)

        toTest.sortPoints(sortHelper=['9', '6', '3'])

        expData = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        expNames = ['9', '6', '3']
        exp = self.constructor(expData, pointNames=expNames)

        assert toTest == exp

    def test_sortPoints_mixedList(self):
        """ Test sortPoints() when we specify a mixed list (names/indices) """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        pnames = ['3', '6', '9']
        toTest = self.constructor(data, pointNames=pnames)

        toTest.sortPoints(sortHelper=['9', '6', 0])

        expData = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        expNames = ['9', '6', '3']
        exp = self.constructor(expData, pointNames=expNames)

        assert toTest == exp

    @raises(ArgumentException)
    def test_sortPoints_exceptionIndicesPEmpty(self):
        """ tests sortPoints() throws an ArgumentException when given invalid indices """
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)
        toTest.sortPoints(sortHelper=[1, 3])

    @raises(ArgumentException)
    def test_sortPoints_exceptionIndicesSmall(self):
        """ tests sortPoints() throws an ArgumentException when given an incorrectly sized indices list """
        data = [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
        toTest = self.constructor(data)

        toTest.sortPoints(sortHelper=[1, 0])

    @raises(ArgumentException)
    def test_sortPoints_exceptionNotUniqueIds(self):
        """ tests sortPoints() throws an ArgumentException when given duplicate indices """
        data = [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
        toTest = self.constructor(data)
        toTest.sortPoints(sortHelper=[1, 1, 0])

    #################
    # sortFeatures() #
    #################

    @raises(ArgumentException)
    def test_sortFeatures_exceptionAtLeastOne(self):
        """ Test sortFeatures() has at least one paramater """
        data = [[7, 8, 9], [1, 2, 3], [4, 5, 6]]
        toTest = self.constructor(data)

        toTest.sortFeatures()

    @raises(ArgumentException)
    def test_sortFeatures_exceptionBothNotNone(self):
        """ Test sortPoints() has only one parameter """
        data = [[7, 8, 9], [1, 2, 3], [4, 5, 6]]
        toTest = self.constructor(data)

        toTest.sortFeatures(sortBy=1, sortHelper=[1,2,0])

    def test_sortFeatures_naturalByPointWithNames(self):
        """ Test sortFeatures() when we specify a point to sort by; includes featureNames """
        data = [[1, 2, 3], [7, 1, 9], [4, 5, 6]]
        names = ["1", "2", "3"]
        toTest = self.constructor(data, featureNames=names)

        ret = toTest.sortFeatures(sortBy=1) # RET CHECK

        dataExpected = [[2, 1, 3], [1, 7, 9], [5, 4, 6]]
        namesExp = ["2", "1", "3"]
        objExp = self.constructor(dataExpected, featureNames=namesExp)

        assert toTest.isIdentical(objExp)
        assert ret is None

    def test_sortFeatures_naturalByPointNameWithFNames(self):
        """ Test sortFeatures() when we specify a point name to sort by; includes featureNames """
        data = [[1, 2, 3], [7, 1, 9], [4, 5, 6]]
        pnames = ['1', '7', '4']
        fnames = ["1", "2", "3"]
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)

        ret = toTest.sortFeatures(sortBy='7') # RET CHECK

        dataExpected = [[2, 1, 3], [1, 7, 9], [5, 4, 6]]
        namesExp = ["2", "1", "3"]
        objExp = self.constructor(dataExpected, pointNames=pnames, featureNames=namesExp)

        assert toTest.isIdentical(objExp)
        assert ret is None


    def test_sortFeatures_scorer(self):
        """ Test sortFeatures() when we specify a scoring function """
        data = [[7, 1, 9, 0], [1, 2, 3, 0], [4, 2, 9, 0]]
        names = ["2", "1", "3", "0"]
        toTest = self.constructor(data, featureNames=names)

        def numOdds(feature):
            ret = 0
            for val in feature:
                if val % 2 != 0:
                    ret += 1
            return ret

        toTest.sortFeatures(sortHelper=numOdds)

        dataExpected = [[0, 1, 7, 9], [0, 2, 1, 3], [0, 2, 4, 9]]
        namesExp = ['0', '1', '2', '3']
        objExp = self.constructor(dataExpected, featureNames=namesExp)

        assert toTest.isIdentical(objExp)

    def test_sortFeatures_comparator(self):
        """ Test sortFeatures() when we specify a comparator function """
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

        toTest.sortFeatures(sortHelper=compOdds)

        dataExpected = [[0, 1, 7, 9], [0, 2, 1, 3], [0, 2, 4, 9]]
        objExp = self.constructor(dataExpected)

        assert toTest.isIdentical(objExp)


    def test_sortFeatures_dataTypeRetainedFromList(self):
        """ Test sortFeatures() data not converted when sorting by list"""
        data = [['a', 2, 3.0], ['b', 5, 6.0], ['c', 8, 9.0]]
        toTest = self.constructor(data)

        toTest.sortFeatures(sortHelper=[2, 1, 0])

        expData = [[3.0, 2, 'a'], [6.0, 5, 'b'], [9.0, 8, 'c']]
        exp = self.constructor(expData)

        assert toTest == exp

    def test_sortFeatures_indicesList(self):
        """ Test sortFeatures() when we specify a list of indices """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        toTest = self.constructor(data)

        toTest.sortFeatures(sortHelper=[2, 1, 0])

        expData = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        exp = self.constructor(expData)

        assert toTest == exp

    def test_sortFeatures_namesList(self):
        """ Test sortFeatures() when we specify a list of feature names """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        fnames = ['third', 'second', 'first']
        toTest = self.constructor(data, featureNames=fnames)

        toTest.sortFeatures(sortHelper=['first', 'second', 'third'])

        expData = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expNames = ['first', 'second', 'third']
        exp = self.constructor(expData, featureNames=expNames)

        assert toTest == exp

    def test_sortFeatures_mixedList(self):
        """ Test sortFeatures() when we specify a mixed list (names/indices) """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        fnames = ['third', 'second', 'first']
        toTest = self.constructor(data, featureNames=fnames)

        toTest.sortFeatures(sortHelper=['first', 'second', 0])

        expData = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expNames = ['first', 'second', 'third']
        exp = self.constructor(expData, featureNames=expNames)

        assert toTest == exp

    @raises(ArgumentException)
    def test_sortFeatures_exceptionIndicesFEmpty(self):
        """ tests sortFeatures() throws an ArgumentException when given invalid indices """
        data = [[], []]
        data = numpy.array(data)
        toTest = self.constructor(data)
        toTest.sortFeatures(sortHelper=[1, 3])


    @raises(ArgumentException)
    def test_sortFeatures_exceptionIndicesSmall(self):
        """ tests sortFeatures() throws an ArgumentException when given an incorrectly sized indices list """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        toTest = self.constructor(data)

        toTest.sortFeatures(sortHelper=[1, 0])


    @raises(ArgumentException)
    def test_sortFeatures_exceptionNotUniqueIds(self):
        """ tests sortFeatures() throws an ArgumentException when given duplicate indices """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        data = numpy.array(data)
        toTest = self.constructor(data)
        toTest.sortFeatures(sortHelper=[1, 1, 0])

    #################
    # extractPoints() #
    #################

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_extractPoints_calls_constructIndicesList(self, mockFunc):
        toTest = self.constructor([[1,2,],[3,4]], pointNames=['a', 'b'])

        ret = toTest.extractPoints(['a', 'b'])

    def test_extractPoints_handmadeSingle(self):
        """ Test extractPoints() against handmade output when extracting one point """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ext1 = toTest.extractPoints(0)
        exp1 = self.constructor([[1, 2, 3]])
        assert ext1.isIdentical(exp1)
        expEnd = self.constructor([[4, 5, 6], [7, 8, 9]])
        assert toTest.isIdentical(expEnd)

    def test_extractPoints_index_NamePath_Preserve(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = 'testName'
        toTest._absPath = 'testAbsPath'
        toTest._relPath = 'testRelPath'

        ext1 = toTest.extractPoints(0)

        assert ext1.nameIsDefault()
        assert ext1.path == 'testAbsPath'
        assert ext1.absolutePath == 'testAbsPath'
        assert ext1.relativePath == 'testRelPath'

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

    def test_extractPoints_ListIntoPEmpty(self):
        """ Test extractPoints() by removing a list of all points """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        expRet = self.constructor(data)
        ret = toTest.extractPoints([0, 1, 2, 3])

        assert ret.isIdentical(expRet)

        data = [[], [], []]
        data = numpy.array(data).T
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)


    def test_extractPoints_handmadeListSequence(self):
        """ Test extractPoints() against handmade output for several list extractions """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        names = ['1', '4', '7', '10']
        toTest = self.constructor(data, pointNames=names)
        ext1 = toTest.extractPoints('1')
        exp1 = self.constructor([[1, 2, 3]], pointNames=['1'])
        assert ext1.isIdentical(exp1)
        ext2 = toTest.extractPoints([1, 2])
        exp2 = self.constructor([[7, 8, 9], [10, 11, 12]], pointNames=['7', '10'])
        assert ext2.isIdentical(exp2)
        expEnd = self.constructor([[4, 5, 6]], pointNames=['4'])
        assert toTest.isIdentical(expEnd)

    def test_extractPoints_handmadeListOrdering(self):
        """ Test extractPoints() against handmade output for out of order extraction """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
        names = ['1', '4', '7', '10', '13']
        toTest = self.constructor(data, pointNames=names)
        ext1 = toTest.extractPoints([3, 4, 1])
        exp1 = self.constructor([[10, 11, 12], [13, 14, 15], [4, 5, 6]], pointNames=['10', '13', '4'])
        assert ext1.isIdentical(exp1)
        expEnd = self.constructor([[1, 2, 3], [7, 8, 9]], pointNames=['1', '7'])
        assert toTest.isIdentical(expEnd)

    def test_extractPoints_List_trickyOrdering(self):
        data = [[0], [2], [2], [2], [0], [0], [0], [0], [2], [0]]
        toExtract = [6, 5, 3, 9]

        toTest = self.constructor(data)

        ret = toTest.extractPoints(toExtract)

        expRaw = [[0], [0], [2], [0]]
        expRet = self.constructor(expRaw)

        expRaw = [[0], [2], [2], [0], [0], [2]]
        expRem = self.constructor(expRaw)

        assert ret == expRet
        assert toTest == expRem

    def test_extractPoints_function_selectionGap(self):
        data = [[0], [2], [2], [2], [0], [0], [0], [0], [2], [0]]
        extractIndices = [3, 5, 6, 9]
        pnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        def sel(point):
            if int(point.getPointName(0)) in extractIndices:
                return True
            else:
                return False

        toTest = self.constructor(data, pointNames=pnames)

        ret = toTest.extractPoints(sel)

        expRaw = [[2], [0], [0], [0]]
        expNames = ['3', '5', '6', '9']
        expRet = self.constructor(expRaw, pointNames=expNames)

        expRaw = [[0], [2], [2], [0], [0], [2]]
        expNames = ['0', '1', '2', '4', '7', '8']
        expRem = self.constructor(expRaw, pointNames=expNames)

        assert ret == expRet
        assert toTest == expRem


    def test_extractPoints_functionIntoPEmpty(self):
        """ Test extractPoints() by removing all points using a function """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        expRet = self.constructor(data)

        ret = toTest.extractPoints(allTrue)
        assert ret.isIdentical(expRet)

        data = [[], [], []]
        data = numpy.array(data).T
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    def test_extractPoints_function_returnPointEmpty(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        exp = self.constructor(data)

        ret = toTest.extractPoints(allFalse)

        data = [[], [], []]
        data = numpy.array(data).T
        expRet = self.constructor(data)

        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(exp)

    def test_extractPoints_handmadeFunction(self):
        """ Test extractPoints() against handmade output for function extraction """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        ext = toTest.extractPoints(oneOrFour)
        exp = self.constructor([[1, 2, 3], [4, 5, 6]])
        assert ext.isIdentical(exp)
        expEnd = self.constructor([[7, 8, 9]])
        assert toTest.isIdentical(expEnd)

    def test_extractPoints_func_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        ext = toTest.extractPoints(oneOrFour)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

        assert ext.nameIsDefault()
        assert ext.absolutePath == 'testAbsPath'
        assert ext.relativePath == 'testRelPath'

    def test_extractPoints_handmadeFuncionWithFeatureNames(self):
        """ Test extractPoints() against handmade output for function extraction with featureNames"""
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)

        ext = toTest.extractPoints(oneOrFour)
        exp = self.constructor([[1, 2, 3], [4, 5, 6]], featureNames=featureNames)
        assert ext.isIdentical(exp)
        expEnd = self.constructor([[7, 8, 9]], featureNames=featureNames)
        assert toTest.isIdentical(expEnd)


    @raises(ArgumentException)
    def test_extractPoints_exceptionStartInvalid(self):
        """ Test extractPoints() for ArgumentException when start is not a valid point index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.extractPoints(start=1.1, end=2)

    @raises(ArgumentException)
    def test_extractPoints_exceptionEndInvalid(self):
        """ Test extractPoints() for ArgumentException when start is not a valid Point index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.extractPoints(start=1, end=5)

    @raises(ArgumentException)
    def test_extractPoints_exceptionInversion(self):
        """ Test extractPoints() for ArgumentException when start comes after end """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.extractPoints(start=2, end=0)

    def test_extractPoints_handmadeRange(self):
        """ Test extractPoints() against handmade output for range extraction """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.extractPoints(start=1, end=2)

        expectedRet = self.constructor([[4, 5, 6], [7, 8, 9]])
        expectedTest = self.constructor([[1, 2, 3]])

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_extractPoints_range_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        ret = toTest.extractPoints(start=1, end=2)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

        assert ret.nameIsDefault()
        assert ret.absolutePath == 'testAbsPath'
        assert ret.relativePath == 'testRelPath'


    def test_extractPoints_rangeIntoPEmpty(self):
        """ Test extractPoints() removes all points using ranges """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints(start=0, end=2)

        assert ret.isIdentical(expRet)

        data = [[], [], []]
        data = numpy.array(data).T
        exp = self.constructor(data, featureNames=featureNames)

        assert toTest.isIdentical(exp)


    def test_extractPoints_handmadeRangeWithFeatureNames(self):
        """ Test extractPoints() against handmade output for range extraction with featureNames """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints(start=1, end=2)

        expectedRet = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=['4', '7'], featureNames=featureNames)
        expectedTest = self.constructor([[1, 2, 3]], pointNames=['1'], featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_extractPoints_handmadeRangeRand_FM(self):
        """ Test extractPoints() for correct sizes when using randomized range extraction and featureNames """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.extractPoints(start=0, end=2, number=2, randomize=True)

        assert ret.points == 2
        assert toTest.points == 1

    def test_extractPoints_handmadeRangeDefaults(self):
        """ Test extractPoints uses the correct defaults in the case of range based extraction """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints(end=1)

        expectedRet = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=['1', '4'], featureNames=featureNames)
        expectedTest = self.constructor([[7, 8, 9]], pointNames=['7'], featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints(start=1)

        expectedTest = self.constructor([[1, 2, 3]], pointNames=['1'], featureNames=featureNames)
        expectedRet = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=['4', '7'], featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_extractPoints_handmade_calling_pointNames(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints(start='4', end='7')

        expectedRet = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_extractPoints_handmadeString(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('one=1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('one==1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('one<2')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('one<=1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('one>4')
        expectedRet = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)
        expectedTest = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=pointNames[:-1], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('one>=7')
        expectedRet = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)
        expectedTest = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=pointNames[:-1], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('one!=4')
        expectedRet = self.constructor([[1, 2, 3], [7, 8, 9]], pointNames=[pointNames[0], pointNames[-1]],
                                       featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6]], pointNames=[pointNames[1]], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('one<1')
        expectedRet = self.constructor([], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('one>0')
        expectedRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expectedTest = self.constructor([], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_extractPoints_handmadeStringWithOperatorWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('one = 1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('one == 1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('one < 2')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('one <= 1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('one > 4')
        expectedRet = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)
        expectedTest = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=pointNames[:-1], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('one >= 7')
        expectedRet = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)
        expectedTest = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=pointNames[:-1], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('one != 4')
        expectedRet = self.constructor([[1, 2, 3], [7, 8, 9]], pointNames=[pointNames[0], pointNames[-1]],
                                       featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6]], pointNames=[pointNames[1]], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('one < 1')
        expectedRet = self.constructor([], featureNames=featureNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('one > 0')
        expectedRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expectedTest = self.constructor([], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_extractPoints_handmadeStringWithFeatureWhitespace(self):
        featureNames = ["feature one", "feature two", "feature three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('feature one=1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName=value with operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('feature one = 1')
        expectedRet = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_extractPoints_list_mixed(self):
        """ Test extractPoints() list input with mixed names and indices """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        names = ['1', '4', '7', '10']
        toTest = self.constructor(data, pointNames=names)
        ret = toTest.extractPoints(['1',1,-1])
        expRet = self.constructor([[1, 2, 3], [4, 5, 6], [10, 11, 12]], pointNames=['1','4','10'])
        expTest = self.constructor([[7, 8, 9]], pointNames=['7'])
        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(expTest)

    @raises(ArgumentException)
    def test_extractPoints_handmadeString_featureNotExist(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractPoints('four=1')

    def test_extractPoints_numberOnly(self):
        self.back_extract_numberOnly('point')

    def test_extractPoints_functionAndNumber(self):
        self.back_extract_functionAndNumber('point')

    def test_extractPoints_numberAndRandomizeAllData(self):
        self.back_extract_numberAndRandomizeAllData('point')

    def test_extractPoints_numberAndRandomizeSelectedData(self):
        self.back_extract_numberAndRandomizeSelectedData('point')

    @raises(ArgumentException)
    def test_extractPoints_randomizeNoNumber(self):
        self.back_structural_randomizeNoNumber('extract', 'point')

    @raises(ArgumentException)
    def test_extractPoints_list_numberGreaterThanTargeted(self):
        self.back_structural_list_numberGreaterThanTargeted('extract', 'point')

    @raises(ArgumentException)
    def test_extractPoints_function_numberGreaterThanTargeted(self):
        self.back_structural_function_numberGreaterThanTargeted('extract', 'point')

    @raises(ArgumentException)
    def test_extractPoints_range_numberGreaterThanTargeted(self):
        self.back_structural_range_numberGreaterThanTargeted('extract', 'point')

    ##########################
    # extract common backend #
    ##########################

    def back_extract_numberOnly(self, axis):
        if axis == 'point':
            toCall = "extractPoints"
        else:
            toCall = "extractFeatures"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        ret = getattr(toTest, toCall)(number=3)
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
            toCall = "extractPoints"
        else:
            toCall = "extractFeatures"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        ret = getattr(toTest, toCall)(allTrue, number=2)
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
            toCall = "extractPoints"
        else:
            toCall = "extractFeatures"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest1 = self.constructor(data, pointNames=pnames, featureNames=fnames)
        toTest2 = toTest1.copy()
        toTest3 = toTest1.copy()
        toTest4 = toTest1.copy()

        seed = UML.randomness.generateSubsidiarySeed()
        UML.randomness.startAlternateControl(seed)
        ret = getattr(toTest1, toCall)(number=3, randomize=True)
        UML.randomness.endAlternateControl()

        UML.randomness.startAlternateControl(seed)
        retList = getattr(toTest2, toCall)([0, 1, 2, 3], number=3, randomize=True)
        UML.randomness.endAlternateControl()

        UML.randomness.startAlternateControl(seed)
        retRange = getattr(toTest3, toCall)(start=0, end=3, number=3, randomize=True)
        UML.randomness.endAlternateControl()

        UML.randomness.startAlternateControl(seed)
        retFunc = getattr(toTest4, toCall)(allTrue, number=3, randomize=True)
        UML.randomness.endAlternateControl()

        if axis == 'point':
            assert ret.points == 3
            assert toTest1.points == 1
        else:
            assert ret.features == 3
            assert toTest1.features == 1

        assert ret.isIdentical(retList)
        assert ret.isIdentical(retRange)
        assert ret.isIdentical(retFunc)
        assert toTest1.isIdentical(toTest2)
        assert toTest1.isIdentical(toTest3)
        assert toTest1.isIdentical(toTest4)

    def back_extract_numberAndRandomizeSelectedData(self, axis):
        """test that randomization occurs after the data has been selected from the user inputs """
        if axis == 'point':
            toCall = "extractPoints"
        else:
            toCall = "extractFeatures"

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

        seed = UML.randomness.generateSubsidiarySeed()
        UML.randomness.startAlternateControl(seed)
        retList = getattr(toTest1, toCall)([1, 2], number=1, randomize=True)
        UML.randomness.endAlternateControl()

        UML.randomness.startAlternateControl(seed)
        retRange = getattr(toTest2, toCall)(start=1, end=2, number=1, randomize=True)
        UML.randomness.endAlternateControl()

        def middleRowsOrCols(value):
            return value[0] in [2, 4, 5, 7]

        UML.randomness.startAlternateControl(seed)
        retFunc = getattr(toTest3, toCall)(middleRowsOrCols, number=1, randomize=True)
        UML.randomness.endAlternateControl()

        assert retList.isIdentical(expRet1) or retList.isIdentical(expRet2)
        assert retRange.isIdentical(expRet1) or retList.isIdentical(expRet2)
        assert retFunc.isIdentical(expRet1) or retList.isIdentical(expRet2)

        assert toTest1.isIdentical(expTest1) or toTest1.isIdentical(expTest2)
        assert toTest2.isIdentical(expTest1) or toTest2.isIdentical(expTest2)
        assert toTest3.isIdentical(expTest1) or toTest3.isIdentical(expTest2)

    ####################
    # extractFeatures() #
    ####################

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_extractFeatures_calls_constructIndicesList(self, mockFunc):
        toTest = self.constructor([[1,2,],[3,4]], featureNames=['a', 'b'])

        ret = toTest.extractFeatures(['a', 'b'])

    def test_extractFeatures_handmadeSingle(self):
        """ Test extractFeatures() against handmade output when extracting one feature """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ext1 = toTest.extractFeatures(0)
        exp1 = self.constructor([[1], [4], [7]])

        assert ext1.isIdentical(exp1)
        expEnd = self.constructor([[2, 3], [5, 6], [8, 9]])
        assert toTest.isIdentical(expEnd)

    def test_extractFeatures_List_NamePath_Preserve(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        ext1 = toTest.extractFeatures(0)

        assert toTest.path == 'testAbsPath'
        assert toTest.absolutePath == 'testAbsPath'
        assert toTest.relativePath == 'testRelPath'

        assert ext1.nameIsDefault()
        assert ext1.absolutePath == 'testAbsPath'
        assert ext1.relativePath == 'testRelPath'

    def test_extractFeatures_ListIntoFEmpty(self):
        """ Test extractFeatures() by removing a list of all features """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        expRet = self.constructor(data)
        ret = toTest.extractFeatures([0, 1, 2])

        assert ret.isIdentical(expRet)

        data = [[], [], [], []]
        data = numpy.array(data)
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    def test_extractFeatures_ListIntoFEmptyOutOfOrder(self):
        """ Test extractFeatures() by removing a list of all features """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        expData = [[3, 1, 2], [6, 4, 5], [9, 7, 8], [12, 10, 11]]
        expRet = self.constructor(expData)
        ret = toTest.extractFeatures([2, 0, 1])

        assert ret.isIdentical(expRet)

        data = [[], [], [], []]
        data = numpy.array(data)
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)


    def test_extractFeatures_handmadeListSequence(self):
        """ Test extractFeatures() against handmade output for several extractions by list """
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data, pointNames=pointNames)
        ext1 = toTest.extractFeatures([0])
        exp1 = self.constructor([[1], [4], [7]], pointNames=pointNames)
        assert ext1.isIdentical(exp1)
        ext2 = toTest.extractFeatures([2, 1])
        exp2 = self.constructor([[-1, 3], [-2, 6], [-3, 9]], pointNames=pointNames)
        assert ext2.isIdentical(exp2)
        expEndData = [[2], [5], [8]]
        expEnd = self.constructor(expEndData, pointNames=pointNames)
        assert toTest.isIdentical(expEnd)

    def test_extractFeatures_handmadeListWithFeatureName(self):
        """ Test extractFeatures() against handmade output for list extraction when specifying featureNames """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        toTest = self.constructor(data, featureNames=featureNames)
        ext1 = toTest.extractFeatures(["one"])
        exp1 = self.constructor([[1], [4], [7]], featureNames=["one"])
        assert ext1.isIdentical(exp1)
        ext2 = toTest.extractFeatures(["three", "neg"])
        exp2 = self.constructor([[3, -1], [6, -2], [9, -3]], featureNames=["three", "neg"])
        assert ext2.isIdentical(exp2)
        expEnd = self.constructor([[2], [5], [8]], featureNames=["two"])
        assert toTest.isIdentical(expEnd)


    def test_extractFeatures_List_trickyOrdering(self):
        data = [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
        toExtract = [6, 5, 3, 9]
        #		toExtract = [3,5,6,9]

        toTest = self.constructor(data)

        ret = toTest.extractFeatures(toExtract)

        expRaw = [0, 0, 1, 0]
        expRet = self.constructor(expRaw)

        expRaw = [0, 1, 1, 0, 0, 1]
        expRem = self.constructor(expRaw)

        assert ret == expRet
        assert toTest == expRem

    def test_extractFeatures_List_reorderingWithFeatureNames(self):
        data = [[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12]]
        fnames = ['a', 'b', 'c', 'd']
        test = self.constructor(data, featureNames=fnames)

        expRetRaw = [[1, 3, 2], [4, 6, 5], [7, 9, 8]]
        expRetNames = ['a', 'c', 'b']
        expRet = self.constructor(expRetRaw, featureNames=expRetNames)

        expTestRaw = [[10], [11], [12]]
        expTestNames = ['d']
        expTest = self.constructor(expTestRaw, featureNames=expTestNames)

        ret = test.extractFeatures(expRetNames)
        assert ret == expRet
        assert test == expTest


    def test_extractFeatures_function_selectionGap(self):
        data = [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
        fnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        extractIndices = [3, 5, 6, 9]

        def sel(feature):
            if int(feature.getFeatureName(0)) in extractIndices:
                return True
            else:
                return False

        toTest = self.constructor(data, featureNames=fnames)

        ret = toTest.extractFeatures(sel)

        expRaw = [1, 0, 0, 0]
        expNames = ['3', '5', '6', '9']
        expRet = self.constructor(expRaw, featureNames=expNames)

        expRaw = [0, 1, 1, 0, 0, 1]
        expNames = ['0', '1', '2', '4', '7', '8']
        expRem = self.constructor(expRaw, featureNames=expNames)

        assert ret == expRet
        assert toTest == expRem


    def test_extractFeatures_functionIntoFEmpty(self):
        """ Test extractFeatures() by removing all featuress using a function """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        expRet = self.constructor(data)

        ret = toTest.extractFeatures(allTrue)
        assert ret.isIdentical(expRet)

        data = [[], [], []]
        data = numpy.array(data)
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    def test_extractFeatures_function_returnPointEmpty(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        exp = self.constructor(data)

        ret = toTest.extractFeatures(allFalse)

        data = [[], [], []]
        data = numpy.array(data)
        expRet = self.constructor(data)

        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(exp)


    def test_extractFeatures_handmadeFunction(self):
        """ Test extractFeatures() against handmade output for function extraction """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data)

        ext = toTest.extractFeatures(absoluteOne)
        exp = self.constructor([[1, -1], [4, -2], [7, -3]])
        assert ext.isIdentical(exp)
        expEnd = self.constructor([[2, 3], [5, 6], [8, 9]])
        assert toTest.isIdentical(expEnd)


    def test_extractFeatures_func_NamePath_preservation(self):
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        ext = toTest.extractFeatures(absoluteOne)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

        assert ext.nameIsDefault()
        assert ext.absolutePath == 'testAbsPath'
        assert ext.relativePath == 'testRelPath'

    def test_extractFeatures_handmadeFunctionWithFeatureName(self):
        """ Test extractFeatures() against handmade output for function extraction with featureNames """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        pointNames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        ext = toTest.extractFeatures(absoluteOne)
        exp = self.constructor([[1, -1], [4, -2], [7, -3]], pointNames=pointNames, featureNames=['one', 'neg'])
        assert ext.isIdentical(exp)
        expEnd = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=["two", "three"])
        assert toTest.isIdentical(expEnd)

    @raises(ArgumentException)
    def test_extractFeatures_exceptionStartInvalid(self):
        """ Test extractFeatures() for ArgumentException when start is not a valid feature index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.extractFeatures(start=1.1, end=2)

    @raises(ArgumentException)
    def test_extractFeatures_exceptionStartInvalidFeatureName(self):
        """ Test extractFeatures() for ArgumentException when start is not a valid feature FeatureName """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.extractFeatures(start="wrong", end=2)

    @raises(ArgumentException)
    def test_extractFeatures_exceptionEndInvalid(self):
        """ Test extractFeatures() for ArgumentException when start is not a valid feature index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.extractFeatures(start=0, end=5)

    @raises(ArgumentException)
    def test_extractFeatures_exceptionEndInvalidFeatureName(self):
        """ Test extractFeatures() for ArgumentException when start is not a valid featureName """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.extractFeatures(start="two", end="five")

    @raises(ArgumentException)
    def test_extractFeatures_exceptionInversion(self):
        """ Test extractFeatures() for ArgumentException when start comes after end """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.extractFeatures(start=2, end=0)

    @raises(ArgumentException)
    def test_extractFeatures_exceptionInversionFeatureName(self):
        """ Test extractFeatures() for ArgumentException when start comes after end as FeatureNames"""
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.extractFeatures(start="two", end="one")


    def test_extractFeatures_rangeIntoFEmpty(self):
        """ Test extractFeatures() removes all Featuress using ranges """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        expRet = self.constructor(data, featureNames=featureNames)
        ret = toTest.extractFeatures(start=0, end=2)

        assert ret.isIdentical(expRet)

        data = [[], [], []]
        data = numpy.array(data)
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    def test_extractFeatures_handmadeRange(self):
        """ Test extractFeatures() against handmade output for range extraction """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.extractFeatures(start=1, end=2)

        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]])
        expectedTest = self.constructor([[1], [4], [7]])

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_extractFeatures_range_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        ret = toTest.extractFeatures(start=1, end=2)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

        assert ret.nameIsDefault()
        assert ret.absolutePath == 'testAbsPath'
        assert ret.relativePath == 'testRelPath'


    def test_extractFeatures_handmadeWithFeatureNames(self):
        """ Test extractFeatures() against handmade output for range extraction with FeatureNames """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures(start=1, end=2)

        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=["two", "three"])
        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=["one"])

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_extractFeatures_handmade_calling_featureNames(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures(start="two", end="three")

        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=["two", "three"])
        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=["one"])

        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_extractFeatures_handmadeString(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['p1', 'p2', 'p3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('p2=5')
        expectedRet = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])
        expectedTest = self.constructor([[1, 3], [4, 6], [7, 9]], pointNames=pointNames,
                                        featureNames=[featureNames[0], featureNames[-1]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('p1==1')
        expectedRet = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])
        expectedTest = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('p3<9')
        expectedRet = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('p3<=8')
        expectedRet = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('p3>8')
        expectedRet = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('p3>8.5')
        expectedRet = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('p1!=1.0')
        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])
        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('p1<1')
        expectedRet = self.constructor([], pointNames=pointNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('p1>0')
        expectedRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expectedTest = self.constructor([], pointNames=pointNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_extractFeatures_handmadeStringWithOperatorWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['p1', 'p2', 'p3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('p2 = 5')
        expectedRet = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])
        expectedTest = self.constructor([[1, 3], [4, 6], [7, 9]], pointNames=pointNames,
                                        featureNames=[featureNames[0], featureNames[-1]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('p1 == 1')
        expectedRet = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])
        expectedTest = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('p3 < 9')
        expectedRet = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('p3 <= 8')
        expectedRet = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('p3 > 8')
        expectedRet = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('p3 > 8.5')
        expectedRet = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('p1 != 1.0')
        expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])
        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('p1 < 1')
        expectedRet = self.constructor([], pointNames=pointNames)
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('p1 > 0')
        expectedRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expectedTest = self.constructor([], pointNames=pointNames)
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_extractFeatures_handmadeStringWithPointWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['pt 1', 'pt 2', 'pt 3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value with no operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('pt 2=5')
        expectedRet = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])
        expectedTest = self.constructor([[1, 3], [4, 6], [7, 9]], pointNames=pointNames,
                                        featureNames=[featureNames[0], featureNames[-1]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

        #test pointName=value with operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('pt 2 = 5')
        expectedRet = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])
        expectedTest = self.constructor([[1, 3], [4, 6], [7, 9]], pointNames=pointNames,
                                        featureNames=[featureNames[0], featureNames[-1]])
        assert expectedRet.isIdentical(ret)
        assert expectedTest.isIdentical(toTest)

    def test_extractFeatures_list_mixed(self):
        """ Test extractFeatures() list input with mixed names and indices """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.extractFeatures([1, "three", -1])
        expRet = self.constructor([[2, 3, -1], [5, 6, -2], [8, 9, -3]], featureNames=["two", "three", "neg"])
        expTest = self.constructor([[1], [4], [7]], featureNames=["one"])
        assert ret.isIdentical(expRet)
        assert toTest.isIdentical(expTest)

    @raises(ArgumentException)
    def test_extractFeatures_handmadeString_pointNotExist(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        ret = toTest.extractFeatures('5=1')

    def test_extractFeatures_numberOnly(self):
        self.back_extract_numberOnly('feature')

    def test_extractFeatures_functionAndNumber(self):
        self.back_extract_functionAndNumber('feature')

    def test_extractFeatures_numberAndRandomizeAllData(self):
        self.back_extract_numberAndRandomizeAllData('feature')

    def test_extractFeatures_numberAndRandomizeSelectedData(self):
        self.back_extract_numberAndRandomizeSelectedData('feature')

    @raises(ArgumentException)
    def test_extractFeatures_randomizeNoNumber(self):
        self.back_structural_randomizeNoNumber('extract', 'feature')

    @raises(ArgumentException)
    def test_extractFeatures_list_numberGreaterThanTargeted(self):
        self.back_structural_list_numberGreaterThanTargeted('extract', 'feature')

    @raises(ArgumentException)
    def test_extractFeatures_function_numberGreaterThanTargeted(self):
        self.back_structural_function_numberGreaterThanTargeted('extract', 'feature')

    @raises(ArgumentException)
    def test_extractFeatures_range_numberGreaterThanTargeted(self):
        self.back_structural_range_numberGreaterThanTargeted('extract', 'feature')

    ################
    # deletePoints #
    ################

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_deletePoints_calls_constructIndicesList(self, mockFunc):
        toTest = self.constructor([[1,2,],[3,4]], pointNames=['a', 'b'])

        toTest.deletePoints(['a', 'b'])

    def test_deletePoints_handmadeSingle(self):
        """ Test deletePoints() against handmade output when deleting one point """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.deletePoints(0)
        expEnd = self.constructor([[4, 5, 6], [7, 8, 9]])
        assert toTest.isIdentical(expEnd)

    def test_deletePoints_index_NamePath_Preserve(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = 'testName'
        toTest._absPath = 'testAbsPath'
        toTest._relPath = 'testRelPath'

        toTest.deletePoints(0)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_deletePoints_ListIntoPEmpty(self):
        """ Test deletePoints() by deleting a list of all points """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        toTest.deletePoints([0, 1, 2, 3])

        data = [[], [], []]
        data = numpy.array(data).T
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)


    def test_deletePoints_handmadeListSequence(self):
        """ Test deletePoints() against handmade output for several list deletions """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        names = ['1', '4', '7', '10']
        toTest = self.constructor(data, pointNames=names)
        toTest.deletePoints('1')
        exp1 = self.constructor([[4, 5, 6], [7, 8, 9], [10, 11, 12]], pointNames=['4', '7', '10'])
        assert toTest.isIdentical(exp1)
        toTest.deletePoints([1, 2])
        exp2 = self.constructor([[4, 5, 6]], pointNames=['4'])
        assert toTest.isIdentical(exp2)

    def test_deletePoints_handmadeListOrdering(self):
        """ Test deletePoints() against handmade output for out of order deletion """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
        names = ['1', '4', '7', '10', '13']
        toTest = self.constructor(data, pointNames=names)
        toTest.deletePoints([3, 4, 1])
        expEnd = self.constructor([[1, 2, 3], [7, 8, 9]], pointNames=['1', '7'])
        assert toTest.isIdentical(expEnd)

    def test_deletePoints_List_trickyOrdering(self):
        data = [[0], [2], [2], [2], [0], [0], [0], [0], [2], [0]]
        toDelete = [6, 5, 3, 9]

        toTest = self.constructor(data)

        toTest.deletePoints(toDelete)

        expRaw = [[0], [2], [2], [0], [0], [2]]
        expRem = self.constructor(expRaw)

        assert toTest == expRem

    def test_deletePoints_function_selectionGap(self):
        data = [[0], [2], [2], [2], [0], [0], [0], [0], [2], [0]]
        deleteIndices = [3, 5, 6, 9]
        pnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        def sel(point):
            if int(point.getPointName(0)) in deleteIndices:
                return True
            else:
                return False

        toTest = self.constructor(data, pointNames=pnames)

        toTest.deletePoints(sel)

        expRaw = [[0], [2], [2], [0], [0], [2]]
        expNames = ['0', '1', '2', '4', '7', '8']
        expRem = self.constructor(expRaw, pointNames=expNames)

        assert toTest == expRem


    def test_deletePoints_functionIntoPEmpty(self):
        """ Test deletePoints() by removing all points using a function """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest.deletePoints(allTrue)

        data = [[], [], []]
        data = numpy.array(data).T
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    def test_deletePoints_function_returnPointEmpty(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        exp = self.constructor(data)

        toTest.deletePoints(allFalse)

        assert toTest.isIdentical(exp)

    def test_deletePoints_handmadeFunction(self):
        """ Test deletePoints() against handmade output for function deletion """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest.deletePoints(oneOrFour)
        expEnd = self.constructor([[7, 8, 9]])
        assert toTest.isIdentical(expEnd)

    def test_deletePoints_func_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.deletePoints(oneOrFour)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_deletePoints_handmadeFuncionWithFeatureNames(self):
        """ Test deletePoints() against handmade output for function deletion with featureNames"""
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)

        toTest.deletePoints(oneOrFour)
        expEnd = self.constructor([[7, 8, 9]], featureNames=featureNames)
        assert toTest.isIdentical(expEnd)


    @raises(ArgumentException)
    def test_deletePoints_exceptionStartInvalid(self):
        """ Test deletePoints() for ArgumentException when start is not a valid point index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.deletePoints(start=1.1, end=2)

    @raises(ArgumentException)
    def test_deletePoints_exceptionEndInvalid(self):
        """ Test deletePoints() for ArgumentException when start is not a valid Point index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.deletePoints(start=1, end=5)

    @raises(ArgumentException)
    def test_deletePoints_exceptionInversion(self):
        """ Test deletePoints() for ArgumentException when start comes after end """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.deletePoints(start=2, end=0)

    def test_deletePoints_handmadeRange(self):
        """ Test deletePoints() against handmade output for range deletion """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.deletePoints(start=1, end=2)

        expectedTest = self.constructor([[1, 2, 3]])

        assert expectedTest.isIdentical(toTest)

    def test_deletePoints_range_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.deletePoints(start=1, end=2)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_deletePoints_rangeIntoPEmpty(self):
        """ Test deletePoints() removes all points using ranges """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints(start=0, end=2)

        data = [[], [], []]
        data = numpy.array(data).T
        exp = self.constructor(data, featureNames=featureNames)

        assert toTest.isIdentical(exp)


    def test_deletePoints_handmadeRangeWithFeatureNames(self):
        """ Test deletePoints() against handmade output for range deletion with featureNames """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints(start=1, end=2)

        expectedTest = self.constructor([[1, 2, 3]], pointNames=['1'], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

    def test_deletePoints_handmadeRangeRand_FM(self):
        """ Test deletePoints() for correct sizes when using randomized range deletion and featureNames """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.deletePoints(start=0, end=2, number=2, randomize=True)

        assert toTest.points == 1

    def test_deletePoints_handmadeRangeDefaults(self):
        """ Test deletePoints uses the correct defaults in the case of range based deletion """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints(end=1)

        expectedTest = self.constructor([[7, 8, 9]], pointNames=['7'], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints(start=1)

        expectedTest = self.constructor([[1, 2, 3]], pointNames=['1'], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

    def test_deletePoints_handmade_calling_pointNames(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints(start='4', end='7')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

    def test_deletePoints_handmadeString(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('one=1')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('one==1')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('one<2')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('one<=1')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('one>4')
        expectedTest = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=pointNames[:-1], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('one>=7')
        expectedTest = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=pointNames[:-1], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('one!=4')
        expectedTest = self.constructor([[4, 5, 6]], pointNames=[pointNames[1]], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('one<1')
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('one>0')
        expectedTest = self.constructor([], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

    def test_deletePoints_handmadeStringWithOperatorWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('one = 1')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('one == 1')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('one < 2')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('one <= 1')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('one > 4')
        expectedTest = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=pointNames[:-1], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('one >= 7')
        expectedTest = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=pointNames[:-1], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('one != 4')
        expectedTest = self.constructor([[4, 5, 6]], pointNames=[pointNames[1]], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('one < 1')
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('one > 0')
        expectedTest = self.constructor([], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

    def test_deletePoints_handmadeStringWithFeatureWhitespace(self):
        featureNames = ["feature one", "feature two", "feature three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('feature one=1')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName=value with operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('feature one = 1')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

    def test_deletePoints_list_mixed(self):
        """ Test deletePoints() list input with mixed names and indices """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        names = ['1', '4', '7', '10']
        toTest = self.constructor(data, pointNames=names)
        toTest.deletePoints(['1',1,-1])
        exp1 = self.constructor([[7, 8, 9]], pointNames=['7'])
        assert toTest.isIdentical(exp1)

    @raises(ArgumentException)
    def test_deletePoints_handmadeString_featureNotExist(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deletePoints('four=1')

    def test_deletePoints_numberOnly(self):
        self.back_delete_numberOnly('point')

    def test_deletePoints_functionAndNumber(self):
        self.back_delete_functionAndNumber('point')

    def test_deletePoints_numberAndRandomizeAllData(self):
        self.back_delete_numberAndRandomizeAllData('point')

    def test_deletePoints_numberAndRandomizeSelectedData(self):
        self.back_delete_numberAndRandomizeSelectedData('point')

    @raises(ArgumentException)
    def test_deletePoints_randomizeNoNumber(self):
        self.back_structural_randomizeNoNumber('delete', 'point')

    @raises(ArgumentException)
    def test_deletePoints_list_numberGreaterThanTargeted(self):
        self.back_structural_list_numberGreaterThanTargeted('delete', 'point')

    @raises(ArgumentException)
    def test_deletePoints_function_numberGreaterThanTargeted(self):
        self.back_structural_function_numberGreaterThanTargeted('delete', 'point')

    @raises(ArgumentException)
    def test_deletePoints_range_numberGreaterThanTargeted(self):
        self.back_structural_range_numberGreaterThanTargeted('delete', 'point')

    #########################
    # delete common backend #
    #########################

    def back_delete_numberOnly(self, axis):
        if axis == 'point':
            toCall = "deletePoints"
        else:
            toCall = "deleteFeatures"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        getattr(toTest, toCall)(number=3)
        if axis == 'point':
            rem = self.constructor(data[3:], pointNames=pnames[3:], featureNames=fnames)
        else:
            rem = self.constructor([p[3:] for p in data], pointNames=pnames, featureNames=fnames[3:])

        assert rem.isIdentical(toTest)

    def back_delete_functionAndNumber(self, axis):
        if axis == 'point':
            toCall = "deletePoints"
        else:
            toCall = "deleteFeatures"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        getattr(toTest, toCall)(allTrue, number=2)
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
            toCall = "deletePoints"
        else:
            toCall = "deleteFeatures"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest1 = self.constructor(data, pointNames=pnames, featureNames=fnames)
        toTest2 = toTest1.copy()
        toTest3 = toTest1.copy()
        toTest4 = toTest1.copy()

        seed = UML.randomness.generateSubsidiarySeed()
        UML.randomness.startAlternateControl(seed)
        getattr(toTest1, toCall)(number=3, randomize=True)
        UML.randomness.endAlternateControl()

        UML.randomness.startAlternateControl(seed)
        getattr(toTest2, toCall)([0, 1, 2, 3], number=3, randomize=True)
        UML.randomness.endAlternateControl()

        UML.randomness.startAlternateControl(seed)
        getattr(toTest3, toCall)(start=0, end=3, number=3, randomize=True)
        UML.randomness.endAlternateControl()

        UML.randomness.startAlternateControl(seed)
        getattr(toTest4, toCall)(allTrue, number=3, randomize=True)
        UML.randomness.endAlternateControl()

        if axis == 'point':
            assert toTest1.points == 1
        else:
            assert toTest1.features == 1

        assert toTest1.isIdentical(toTest2)
        assert toTest1.isIdentical(toTest3)
        assert toTest1.isIdentical(toTest4)

    def back_delete_numberAndRandomizeSelectedData(self, axis):
        """test that randomization occurs after the data has been selected from the user inputs """
        if axis == 'point':
            toCall = "deletePoints"
        else:
            toCall = "deleteFeatures"

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

        seed = UML.randomness.generateSubsidiarySeed()
        UML.randomness.startAlternateControl(seed)
        getattr(toTest1, toCall)([1, 2], number=1, randomize=True)
        UML.randomness.endAlternateControl()

        UML.randomness.startAlternateControl(seed)
        getattr(toTest2, toCall)(start=1, end=2, number=1, randomize=True)
        UML.randomness.endAlternateControl()

        def middleRowsOrCols(value):
            return value[0] in [2, 4, 5, 7]

        UML.randomness.startAlternateControl(seed)
        getattr(toTest3, toCall)(middleRowsOrCols, number=1, randomize=True)
        UML.randomness.endAlternateControl()

        assert toTest1.isIdentical(exp1) or toTest1.isIdentical(exp2)
        assert toTest2.isIdentical(exp1) or toTest2.isIdentical(exp2)
        assert toTest3.isIdentical(exp1) or toTest3.isIdentical(exp2)

    ##################
    # deleteFeatures #
    ##################

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_deleteFeatures_calls_constructIndicesList(self, mockFunc):
        toTest = self.constructor([[1,2,],[3,4]], featureNames=['a', 'b'])

        toTest.deleteFeatures(['a', 'b'])

    def test_deleteFeatures_handmadeSingle(self):
        """ Test deleteFeatures() against handmade output when deleting one feature """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.deleteFeatures(0)

        expEnd = self.constructor([[2, 3], [5, 6], [8, 9]])
        assert toTest.isIdentical(expEnd)

    def test_deleteFeatures_List_NamePath_Preserve(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.deleteFeatures(0)

        assert toTest.path == 'testAbsPath'
        assert toTest.absolutePath == 'testAbsPath'
        assert toTest.relativePath == 'testRelPath'


    def test_deleteFeatures_ListIntoFEmpty(self):
        """ Test deleteFeatures() by removing a list of all features """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        toTest.deleteFeatures([0, 1, 2])

        data = [[], [], [], []]
        data = numpy.array(data)
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    def test_deleteFeatures_ListIntoFEmptyOutOfOrder(self):
        """ Test deleteFeatures() by removing a list of all features """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        toTest.deleteFeatures([2, 0, 1])

        data = [[], [], [], []]
        data = numpy.array(data)
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)


    def test_deleteFeatures_handmadeListSequence(self):
        """ Test deleteFeatures() against handmade output for several deletions by list """
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data, pointNames=pointNames)
        toTest.deleteFeatures([0])
        exp1 = self.constructor([[2, 3, -1], [5, 6, -2], [8, 9, -3]], pointNames=pointNames)
        assert toTest.isIdentical(exp1)
        toTest.deleteFeatures([2, 1])
        expEndData = [[2], [5], [8]]
        exp2 = self.constructor(expEndData, pointNames=pointNames)
        assert toTest.isIdentical(exp2)

    def test_deleteFeatures_handmadeListWithFeatureName(self):
        """ Test deleteFeatures() against handmade output for list deletion when specifying featureNames """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.deleteFeatures(["one"])
        exp1 = self.constructor([[2, 3, -1], [5, 6, -2], [8, 9, -3]], featureNames=["two", "three", "neg"])
        assert toTest.isIdentical(exp1)
        toTest.deleteFeatures(["three", "neg"])
        exp2 = self.constructor([[2], [5], [8]], featureNames=["two"])
        assert toTest.isIdentical(exp2)


    def test_deleteFeatures_List_trickyOrdering(self):
        data = [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
        toDelete = [6, 5, 3, 9]

        toTest = self.constructor(data)
        toTest.deleteFeatures(toDelete)

        expRaw = [0, 1, 1, 0, 0, 1]
        expRem = self.constructor(expRaw)

        assert toTest == expRem

    def test_deleteFeatures_List_reorderingWithFeatureNames(self):
        data = [[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12]]
        fnames = ['a', 'b', 'c', 'd']
        toTest = self.constructor(data, featureNames=fnames)

        toDelete = ['a', 'c', 'b']
        toTest.deleteFeatures(toDelete)
        expTestRaw = [[10], [11], [12]]
        expTestNames = ['d']
        expTest = self.constructor(expTestRaw, featureNames=expTestNames)

        assert toTest == expTest


    def test_deleteFeatures_function_selectionGap(self):
        data = [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
        fnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        deleteIndices = [3, 5, 6, 9]

        def sel(feature):
            if int(feature.getFeatureName(0)) in deleteIndices:
                return True
            else:
                return False

        toTest = self.constructor(data, featureNames=fnames)
        toTest.deleteFeatures(sel)

        expRaw = [0, 1, 1, 0, 0, 1]
        expNames = ['0', '1', '2', '4', '7', '8']
        expRem = self.constructor(expRaw, featureNames=expNames)

        assert toTest == expRem


    def test_deleteFeatures_functionIntoFEmpty(self):
        """ Test deleteFeatures() by removing all featuress using a function """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest.deleteFeatures(allTrue)

        data = [[], [], []]
        data = numpy.array(data)
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    def test_deleteFeatures_function_returnPointEmpty(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        exp = self.constructor(data)

        toTest.deleteFeatures(allFalse)

        assert toTest.isIdentical(exp)


    def test_deleteFeatures_handmadeFunction(self):
        """ Test deleteFeatures() against handmade output for function deletion """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data)

        toTest.deleteFeatures(absoluteOne)

        expEnd = self.constructor([[2, 3], [5, 6], [8, 9]])
        assert toTest.isIdentical(expEnd)


    def test_deleteFeatures_func_NamePath_preservation(self):
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.deleteFeatures(absoluteOne)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_deleteFeatures_handmadeFunctionWithFeatureName(self):
        """ Test deleteFeatures() against handmade output for function deletion with featureNames """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        pointNames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        ext = toTest.deleteFeatures(absoluteOne)
        expEnd = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=["two", "three"])
        assert toTest.isIdentical(expEnd)

    @raises(ArgumentException)
    def test_deleteFeatures_exceptionStartInvalid(self):
        """ Test deleteFeatures() for ArgumentException when start is not a valid feature index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.deleteFeatures(start=1.1, end=2)

    @raises(ArgumentException)
    def test_deleteFeatures_exceptionStartInvalidFeatureName(self):
        """ Test deleteFeatures() for ArgumentException when start is not a valid feature FeatureName """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.deleteFeatures(start="wrong", end=2)

    @raises(ArgumentException)
    def test_deleteFeatures_exceptionEndInvalid(self):
        """ Test deleteFeatures() for ArgumentException when start is not a valid feature index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.deleteFeatures(start=0, end=5)

    @raises(ArgumentException)
    def test_deleteFeatures_exceptionEndInvalidFeatureName(self):
        """ Test deleteFeatures() for ArgumentException when end is not a valid featureName """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.deleteFeatures(start="two", end="five")

    @raises(ArgumentException)
    def test_deleteFeatures_exceptionInversion(self):
        """ Test deleteFeatures() for ArgumentException when start comes after end """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.deleteFeatures(start=2, end=0)

    @raises(ArgumentException)
    def test_deleteFeatures_exceptionInversionFeatureName(self):
        """ Test deleteFeatures() for ArgumentException when start comes after end as FeatureNames"""
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.deleteFeatures(start="two", end="one")

    def test_deleteFeatures_rangeIntoFEmpty(self):
        """ Test deleteFeatures() removes all Featuress using ranges """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.deleteFeatures(start=0, end=2)

        data = [[], [], []]
        data = numpy.array(data)
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)

    def test_deleteFeatures_handmadeRange(self):
        """ Test deleteFeatures() against handmade output for range deletion """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.deleteFeatures(start=1, end=2)

        expectedTest = self.constructor([[1], [4], [7]])

        assert expectedTest.isIdentical(toTest)

    def test_deleteFeatures_range_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.deleteFeatures(start=1, end=2)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_deleteFeatures_handmadeWithFeatureNames(self):
        """ Test deleteFeatures() against handmade output for range deletion with FeatureNames """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures(start=1, end=2)

        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=["one"])

        assert expectedTest.isIdentical(toTest)

    def test_deleteFeatures_handmade_calling_featureNames(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures(start="two", end="three")

        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=["one"])

        assert expectedTest.isIdentical(toTest)

    def test_deleteFeatures_handmadeString(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['p1', 'p2', 'p3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('p2=5')
        expectedTest = self.constructor([[1, 3], [4, 6], [7, 9]], pointNames=pointNames,
                                        featureNames=[featureNames[0], featureNames[-1]])
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('p1==1')
        expectedTest = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('p3<9')
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('p3<=8')
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('p3>8')
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('p3>8.5')
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('p1!=1.0')
        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('p1<1')
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('p1>0')
        expectedTest = self.constructor([[], [], []], pointNames=pointNames)
        assert expectedTest.isIdentical(toTest)

    def test_deleteFeatures_handmadeStringWithOperatorWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['p1', 'p2', 'p3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('p2 = 5')
        expectedTest = self.constructor([[1, 3], [4, 6], [7, 9]], pointNames=pointNames,
                                        featureNames=[featureNames[0], featureNames[-1]])
        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('p1 == 1')
        expectedTest = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])
        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('p3 < 9')
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('p3 <= 8')
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('p3 > 8')
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('p3 > 8.5')
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('p1 != 1.0')
        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('p1 < 1')
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('p1 > 0')
        expectedTest = self.constructor([[],[],[]], pointNames=pointNames)
        assert expectedTest.isIdentical(toTest)

    def test_deleteFeatures_handmadeStringWithPointWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['pt 1', 'pt 2', 'pt 3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value with no operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('pt 2=5')
        expectedTest = self.constructor([[1, 3], [4, 6], [7, 9]], pointNames=pointNames,
                                        featureNames=[featureNames[0], featureNames[-1]])
        assert expectedTest.isIdentical(toTest)

        #test pointName=value with operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('pt 2 = 5')
        expectedTest = self.constructor([[1, 3], [4, 6], [7, 9]], pointNames=pointNames,
                                        featureNames=[featureNames[0], featureNames[-1]])
        assert expectedTest.isIdentical(toTest)

    def test_deleteFeatures_list_mixed(self):
        """ Test deleteFeatures() list input with mixed names and indices """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.deleteFeatures([1, "three", -1])
        exp1 = self.constructor([[1], [4], [7]], featureNames=["one"])
        assert toTest.isIdentical(exp1)

    @raises(ArgumentException)
    def test_deleteFeatures_handmadeString_pointNotExist(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.deleteFeatures('5=1')

    def test_deleteFeatures_numberOnly(self):
        self.back_delete_numberOnly('feature')

    def test_deleteFeatures_functionAndNumber(self):
        self.back_delete_functionAndNumber('feature')

    def test_deleteFeatures_numberAndRandomizeAllData(self):
        self.back_delete_numberAndRandomizeAllData('feature')

    def test_deleteFeatures_numberAndRandomizeSelectedData(self):
        self.back_delete_numberAndRandomizeSelectedData('feature')

    @raises(ArgumentException)
    def test_deleteFeatures_randomizeNoNumber(self):
        self.back_structural_randomizeNoNumber('delete', 'feature')

    @raises(ArgumentException)
    def test_deleteFeatures_list_numberGreaterThanTargeted(self):
        self.back_structural_list_numberGreaterThanTargeted('delete', 'feature')

    @raises(ArgumentException)
    def test_deleteFeatures_function_numberGreaterThanTargeted(self):
        self.back_structural_function_numberGreaterThanTargeted('delete', 'feature')

    @raises(ArgumentException)
    def test_deleteFeatures_range_numberGreaterThanTargeted(self):
        self.back_structural_range_numberGreaterThanTargeted('delete', 'feature')

    ################
    # retainPoints #
    ################

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_retainPoints_calls_constructIndicesList(self, mockFunc):
        """ Test retainPoints calls _constructIndicesList before calling _genericStructuralFrontend"""
        toTest = self.constructor([[1,2,],[3,4]], pointNames=['a', 'b'])
        toTest.retainPoints(['a', 'b'])

    def test_retainPoints_handmadeSingle(self):
        """ Test retainPoints() against handmade output when retaining one point """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.retainPoints(0)
        exp1 = self.constructor([[1, 2, 3]])
        assert toTest.isIdentical(exp1)

    def test_retainPoints_index_NamePath_Preserve(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = 'testName'
        toTest._absPath = 'testAbsPath'
        toTest._relPath = 'testRelPath'

        toTest.retainPoints(0)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_retainPoints_list_retain_all(self):
        """ Test retainPoints() by retaining a list of all points """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        exp = self.constructor(data)
        toTest.retainPoints([0, 1, 2, 3])

        assert toTest.isIdentical(exp)

    def test_retainPoints_list_retain_nothing(self):
        """ Test retainPoints() by retaining an empty list """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        toTest.retainPoints([])

        expData = [[], [], []]
        expData = numpy.array(expData).T
        expTest = self.constructor(expData)
        assert toTest.isIdentical(expTest)

    def test_retainPoints_pythonRange(self):
        """ Test retainPoints() by retaining a python range of points """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        exp = self.constructor([[4, 5, 6], [7, 8, 9]])
        toTest.retainPoints(range(1,3))

        assert toTest.isIdentical(exp)

    def test_retainPoints_handmadeListSequence(self):
        """ Test retainPoints() against handmade output for several list retentions """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        names = ['1', '4', '7', '10']
        toTest = self.constructor(data, pointNames=names)
        toTest.retainPoints(['1','4','10'])
        exp1 = self.constructor([[1, 2, 3], [4, 5, 6], [10, 11, 12]], pointNames=['1','4','10'])
        assert toTest.isIdentical(exp1)
        toTest.retainPoints(1)
        exp2 = self.constructor([4, 5, 6], pointNames=['4'])
        assert toTest.isIdentical(exp2)


    def test_retainPoints_list_mixed(self):
        """ Test retainPoints() list input with mixed names and indices """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        names = ['1', '4', '7', '10']
        toTest = self.constructor(data, pointNames=names)
        toTest.retainPoints(['1',1,-1])
        exp1 = self.constructor([[1, 2, 3], [4, 5, 6], [10, 11, 12]], pointNames=['1','4','10'])
        assert toTest.isIdentical(exp1)


    def test_retainPoints_handmadeListOrdering(self):
        """ Test retainPoints() against handmade output for out of order retention """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
        names = ['1', '4', '7', '10', '13']
        toTest = self.constructor(data, pointNames=names)
        toTest.retainPoints([3, 4, 1])
        exp1 = self.constructor([[10, 11, 12], [13, 14, 15], [4, 5, 6]], pointNames=['10', '13', '4'])
        assert toTest.isIdentical(exp1)


    def test_retainPoints_List_trickyOrdering(self):
        data = [[0], [2], [2], [2], [0], [0], [0], [0], [2], [0]]
        toRetain = [6, 5, 3, 9]

        toTest = self.constructor(data)

        toTest.retainPoints(toRetain)

        expRaw = [[0], [0], [2], [0]]
        expTest = self.constructor(expRaw)

        assert toTest == expTest

    def test_retainPoints_function_selectionGap(self):
        data = [[0], [2], [2], [2], [0], [0], [0], [0], [2], [0]]
        retainIndices = [3, 5, 6, 9]
        pnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        def sel(point):
            if int(point.getPointName(0)) in retainIndices:
                return True
            else:
                return False

        toTest = self.constructor(data, pointNames=pnames)

        toTest.retainPoints(sel)

        expRaw = [[2], [0], [0], [0]]
        expNames = ['3', '5', '6', '9']
        expTest = self.constructor(expRaw, pointNames=expNames)

        assert toTest == expTest


    def test_retainPoints_functionIntoPEmpty(self):
        """ Test retainPoints() by retaining all points using a function """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        expTest = self.constructor(data)

        toTest.retainPoints(allTrue)
        assert toTest.isIdentical(expTest)


    def test_retainPoints_function_returnPointEmpty(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest.retainPoints(allFalse)

        expData = numpy.array([[], [], []])
        expData = expData.T
        expTest = self.constructor(expData)

        assert toTest.isIdentical(expTest)

    def test_retainPoints_function_NumberAndRandomize(self):
        data = [[1], [2], [3], [4], [5], [6], [7], [8]]
        toTest = self.constructor(data)

        toTest.retainPoints(evenOnly, number=3, randomize=True)
        assert toTest.points == 3

    def test_retainPoints_handmadeFunction(self):
        """ Test retainPoints() against handmade output for function retention """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest.retainPoints(oneOrFour)
        exp = self.constructor([[1, 2, 3], [4, 5, 6]])
        assert toTest.isIdentical(exp)


    def test_retainPoints_func_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.retainPoints(oneOrFour)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_retainPoints_handmadeFunctionWithFeatureNames(self):
        """ Test retainPoints() against handmade output for function retention with featureNames"""
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)

        toTest.retainPoints(oneOrFour)
        exp = self.constructor([[1, 2, 3], [4, 5, 6]], featureNames=featureNames)
        assert toTest.isIdentical(exp)


    @raises(ArgumentException)
    def test_retainPoints_exceptionStartInvalid(self):
        """ Test retainPoints() for ArgumentException when start is not a valid point index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.retainPoints(start=1.1, end=2)

    @raises(ArgumentException)
    def test_retainPoints_exceptionEndInvalid(self):
        """ Test retainPoints() for ArgumentException when start is not a valid Point index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.retainPoints(start=1, end=5)

    @raises(ArgumentException)
    def test_retainPoints_exceptionInversion(self):
        """ Test retainPoints() for ArgumentException when start comes after end """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.retainPoints(start=2, end=0)

    def test_retainPoints_handmadeRange(self):
        """ Test retainPoints() against handmade output for range retention """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.retainPoints(start=1, end=2)

        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]])

        assert expectedTest.isIdentical(toTest)

    def test_retainPoints_range_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.retainPoints(start=1, end=2)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_retainPoints_rangeIntoPEmpty(self):
        """ Test retainPoints() retains all points using ranges """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        expRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints(start=0, end=2)

        assert toTest.isIdentical(expRet)


    def test_retainPoints_handmadeRangeWithFeatureNames(self):
        """ Test retainPoints() against handmade output for range retention with featureNames """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints(start=1, end=2)

        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=['4', '7'], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

    def test_retainPoints_handmadeRangeRand_FM(self):
        """ Test retainPoints() for correct sizes when using randomized range retention and featureNames """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.retainPoints(start=0, end=2, number=2, randomize=True)
        assert toTest.points == 2

    def test_retainPoints_handmadeRangeDefaults(self):
        """ Test retainPoints uses the correct defaults in the case of range based retention """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints(end=1)

        expectedTest = self.constructor([[1, 2, 3], [4, 5, 6]], pointNames=['1', '4'], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints(start=1)

        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=['4', '7'], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

    def test_retainPoints_handmade_calling_pointNames(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints(start='4', end='7')
        expectedTest = self.constructor([[4, 5, 6], [7, 8, 9]], pointNames=pointNames[1:], featureNames=featureNames)
        assert expectedTest.isIdentical(toTest)

    def test_retainPoints_handmadeString(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('one=1')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('one==1')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('one<2')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('one<=1')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('one>4')
        expectedTest = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('one>=7')
        expectedTest = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('one!=4')
        expectedTest = self.constructor([[1, 2, 3], [7, 8, 9]], pointNames=[pointNames[0], pointNames[-1]],
                                       featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('one<1')
        expectedTest = self.constructor([], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('one>0')
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

    def test_retainPoints_handmadeStringWithOperatorWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('one = 1')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('one == 1')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('one < 2')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('one <= 1')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('one > 4')
        expectedTest = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('one >= 7')
        expectedTest = self.constructor([[7, 8, 9]], pointNames=pointNames[-1:], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('one != 4')
        expectedTest = self.constructor([[1, 2, 3], [7, 8, 9]], pointNames=[pointNames[0], pointNames[-1]],
                                       featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('one < 1')
        expectedTest = self.constructor([], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        assert expectedTest.isIdentical(toTest)
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('one > 0')
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

    def test_retainPoints_handmadeStringWithFeatureWhitespace(self):
        featureNames = ["feature one", "feature two", "feature three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test featureName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('feature one=1')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName=value with operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('feature one = 1')
        expectedTest = self.constructor([[1, 2, 3]], pointNames=pointNames[:1], featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

    @raises(ArgumentException)
    def test_retainPoints_handmadeString_featureNotExist(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainPoints('four=1')

    def test_retainPoints_numberOnly(self):
        self.back_retain_numberOnly('point')

    def test_retainPoints_functionAndNumber(self):
        self.back_retain_functionAndNumber('point')

    def test_retainPoints_numberAndRandomizeAllData(self):
        self.back_retain_numberAndRandomizeAllData('point')

    def test_retainPoints_numberAndRandomizeSelectedData(self):
        self.back_retain_numberAndRandomizeSelectedData('point')

    @raises(ArgumentException)
    def test_retainPoints_randomizeNoNumber(self):
        self.back_structural_randomizeNoNumber('retain', 'point')

    @raises(ArgumentException)
    def test_retainPoints_list_numberGreaterThanTargeted(self):
        self.back_structural_list_numberGreaterThanTargeted('retain', 'point')

    @raises(ArgumentException)
    def test_retainPoints_function_numberGreaterThanTargeted(self):
        self.back_structural_function_numberGreaterThanTargeted('retain', 'point')

    @raises(ArgumentException)
    def test_retainPoints_range_numberGreaterThanTargeted(self):
        self.back_structural_range_numberGreaterThanTargeted('retain', 'point')

    #########################
    # retain common backend #
    #########################

    def back_retain_numberOnly(self, axis):
        if axis == 'point':
            toCall = "retainPoints"
        else:
            toCall = "retainFeatures"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        getattr(toTest, toCall)(number=3)
        if axis == 'point':
            exp = self.constructor(data[:3], pointNames=pnames[:3], featureNames=fnames)
        else:
            exp = self.constructor([p[:3] for p in data], pointNames=pnames, featureNames=fnames[:3])

        assert exp.isIdentical(toTest)

    def back_retain_functionAndNumber(self, axis):
        if axis == 'point':
            toCall = "retainPoints"
        else:
            toCall = "retainFeatures"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        getattr(toTest, toCall)([0,1,2], number=2)
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
            toCall = "retainPoints"
        else:
            toCall = "retainFeatures"

        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 14]]
        pnames = ['1', '4', '7', '10']
        fnames = ['a', 'b', 'd', 'gg']
        toTest1 = self.constructor(data, pointNames=pnames, featureNames=fnames)
        toTest2 = toTest1.copy()
        toTest3 = toTest1.copy()
        toTest4 = toTest1.copy()

        seed = UML.randomness.generateSubsidiarySeed()
        UML.randomness.startAlternateControl(seed)
        getattr(toTest1, toCall)(number=3, randomize=True)
        UML.randomness.endAlternateControl()

        UML.randomness.startAlternateControl(seed)
        getattr(toTest2, toCall)([0, 1, 2, 3], number=3, randomize=True)
        UML.randomness.endAlternateControl()

        UML.randomness.startAlternateControl(seed)
        getattr(toTest3, toCall)(start=0, end=3, number=3, randomize=True)
        UML.randomness.endAlternateControl()

        UML.randomness.startAlternateControl(seed)
        getattr(toTest4, toCall)(allTrue, number=3, randomize=True)
        UML.randomness.endAlternateControl()

        if axis == 'point':
            assert toTest1.points == 3
        else:
            assert toTest1.features == 3

        assert toTest1.isIdentical(toTest2)
        assert toTest1.isIdentical(toTest3)
        assert toTest1.isIdentical(toTest4)

    def back_retain_numberAndRandomizeSelectedData(self, axis):
        """test that randomization occurs after the data has been selected from the user inputs """
        if axis == 'point':
            toCall = "retainPoints"
        else:
            toCall = "retainFeatures"

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

        seed = UML.randomness.generateSubsidiarySeed()
        UML.randomness.startAlternateControl(seed)
        getattr(toTest1, toCall)([1, 2], number=1, randomize=True)
        UML.randomness.endAlternateControl()

        UML.randomness.startAlternateControl(seed)
        getattr(toTest2, toCall)(start=1, end=2, number=1, randomize=True)
        UML.randomness.endAlternateControl()

        def middleRowsOrCols(value):
            return value[0] in [2, 4, 5, 7]

        UML.randomness.startAlternateControl(seed)
        getattr(toTest3, toCall)(middleRowsOrCols, number=1, randomize=True)
        UML.randomness.endAlternateControl()

        assert toTest1.isIdentical(exp1) or toTest1.isIdentical(exp2)
        assert toTest2.isIdentical(exp1) or toTest2.isIdentical(exp2)
        assert toTest3.isIdentical(exp1) or toTest3.isIdentical(exp2)

    ##################
    # retainFeatures #
    ##################

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_retainFeatures_calls_constructIndicesList(self, mockFunc):
        toTest = self.constructor([[1,2,],[3,4]], featureNames=['a', 'b'])

        toTest.retainFeatures(['a', 'b'])

    def test_retainFeatures_handmadeSingle(self):
        """ Test retainFeatures() against handmade output when retaining one feature """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.retainFeatures(0)
        exp1 = self.constructor([[1], [4], [7]])

        assert toTest.isIdentical(exp1)

    def test_retainFeatures_List_NamePath_Preserve(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.retainFeatures(0)

        assert toTest.path == 'testAbsPath'
        assert toTest.absolutePath == 'testAbsPath'
        assert toTest.relativePath == 'testRelPath'

    def test_retainFeatures_list_retain_all(self):
        """ Test retainFeatures() by retaining a list of all features """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        expTest = self.constructor(data)
        toTest.retainFeatures([0, 1, 2])

        assert toTest.isIdentical(expTest)

    def test_retainFeatures_list_retain_nothing(self):
        """ Test retainFeatures() by retaining an empty list """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        toTest.retainFeatures([])

        expData = [[], [], [], []]
        expData = numpy.array(expData)
        expTest = self.constructor(expData)
        assert toTest.isIdentical(expTest)

    def test_retainFeatures_pythonRange(self):
        """ Test retainFeatures() by retaining a python range of points """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        exp = self.constructor([[2, 3], [5, 6], [8, 9], [11, 12]])
        toTest.retainFeatures(range(1,3))

        assert toTest.isIdentical(exp)

    def test_retainFeatures_ListIntoFEmptyOutOfOrder(self):
        """ Test retainFeatures() by retaining a list of all features """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(data)
        expData = [[3, 1, 2], [6, 4, 5], [9, 7, 8], [12, 10, 11]]
        expTest = self.constructor(expData)
        toTest.retainFeatures([2, 0, 1])

        assert toTest.isIdentical(expTest)

    def test_retainFeatures_handmadeListSequence(self):
        """ Test retainFeatures() against handmade output for several retentions by list """
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data, pointNames=pointNames)
        toTest.retainFeatures([1, 2, 3])
        exp1 = self.constructor([[2, 3, -1], [5, 6, -2], [8, 9, -3]], pointNames=pointNames)
        assert toTest.isIdentical(exp1)
        toTest.retainFeatures([2, 1])
        exp2 = self.constructor([[-1, 3], [-2, 6], [-3, 9]], pointNames=pointNames)
        assert toTest.isIdentical(exp2)

    def test_retainFeatures_handmadeListWithFeatureName(self):
        """ Test retainFeatures() against handmade output for list retention when specifying featureNames """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.retainFeatures(["two", "three", "neg"])
        exp1 = self.constructor([[2, 3, -1], [5, 6, -2], [8, 9, -3]], featureNames=["two", "three", "neg"])
        assert toTest.isIdentical(exp1)
        toTest.retainFeatures(["three", "neg"])
        exp2 = self.constructor([[3, -1], [6, -2], [9, -3]], featureNames=["three", "neg"])
        assert toTest.isIdentical(exp2)


    def test_retainFeatures_list_mixed(self):
        """ Test retainFeatures() list input with mixed names and indices """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.retainFeatures([1, "three", -1])
        exp1 = self.constructor([[2, 3, -1], [5, 6, -2], [8, 9, -3]], featureNames=["two", "three", "neg"])
        assert toTest.isIdentical(exp1)


    def test_retainFeatures_List_trickyOrdering(self):
        data = [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
        toRetain = [6, 5, 3, 9]

        toTest = self.constructor(data)

        toTest.retainFeatures(toRetain)

        expRaw = [0, 0, 1, 0]
        expTest = self.constructor(expRaw)

        assert toTest == expTest

    def test_retainFeatures_List_reorderingWithFeatureNames(self):
        data = [[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12]]
        fnames = ['a', 'b', 'c', 'd']
        test = self.constructor(data, featureNames=fnames)

        expRetRaw = [[1, 3, 2], [4, 6, 5], [7, 9, 8]]
        expRetNames = ['a', 'c', 'b']
        exp = self.constructor(expRetRaw, featureNames=expRetNames)

        test.retainFeatures(expRetNames)
        assert test == exp


    def test_retainFeatures_function_selectionGap(self):
        data = [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
        fnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        retainIndices = [3, 5, 6, 9]

        def sel(feature):
            if int(feature.getFeatureName(0)) in retainIndices:
                return True
            else:
                return False

        toTest = self.constructor(data, featureNames=fnames)

        toTest.retainFeatures(sel)

        expRaw = [1, 0, 0, 0]
        expNames = ['3', '5', '6', '9']
        expTest = self.constructor(expRaw, featureNames=expNames)

        assert toTest == expTest


    def test_retainFeatures_functionIntoFEmpty(self):
        """ Test retainFeatures() by retaining all featuress using a function """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        expTest = self.constructor(data)

        toTest.retainFeatures(allTrue)
        assert toTest.isIdentical(expTest)

    def test_retainFeatures_function_returnPointEmpty(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest.retainFeatures(allFalse)

        data = [[], [], []]
        data = numpy.array(data)
        expTest = self.constructor(data)

        assert toTest.isIdentical(expTest)

    def test_retainFeatures_function_NumberAndRandomize(self):
        data = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        toTest = self.constructor(data)

        toTest.retainFeatures(evenOnly, number=2, randomize=True)
        assert toTest.features == 2

    def test_retainFeatures_handmadeFunction(self):
        """ Test retainFeatures() against handmade output for function retention """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data)

        toTest.retainFeatures(absoluteOne)
        exp = self.constructor([[1, -1], [4, -2], [7, -3]])
        assert toTest.isIdentical(exp)

    def test_retainFeatures_func_NamePath_preservation(self):
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.retainFeatures(absoluteOne)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_retainFeatures_handmadeFunctionWithFeatureName(self):
        """ Test retainFeatures() against handmade output for function retention with featureNames """
        data = [[1, 2, 3, -1], [4, 5, 6, -2], [7, 8, 9, -3]]
        featureNames = ["one", "two", "three", "neg"]
        pointNames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        toTest.retainFeatures(absoluteOne)
        exp = self.constructor([[1, -1], [4, -2], [7, -3]], pointNames=pointNames, featureNames=['one', 'neg'])
        assert toTest.isIdentical(exp)

    @raises(ArgumentException)
    def test_retainFeatures_exceptionStartInvalid(self):
        """ Test retainFeatures() for ArgumentException when start is not a valid feature index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.retainFeatures(start=1.1, end=2)

    @raises(ArgumentException)
    def test_retainFeatures_exceptionStartInvalidFeatureName(self):
        """ Test retainFeatures() for ArgumentException when start is not a valid feature FeatureName """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.retainFeatures(start="wrong", end=2)

    @raises(ArgumentException)
    def test_retainFeatures_exceptionEndInvalid(self):
        """ Test retainFeatures() for ArgumentException when start is not a valid feature index """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.retainFeatures(start=0, end=5)

    @raises(ArgumentException)
    def test_retainFeatures_exceptionEndInvalidFeatureName(self):
        """ Test retainFeatures() for ArgumentException when start is not a valid featureName """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.retainFeatures(start="two", end="five")

    @raises(ArgumentException)
    def test_retainFeatures_exceptionInversion(self):
        """ Test retainFeatures() for ArgumentException when start comes after end """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.retainFeatures(start=2, end=0)

    @raises(ArgumentException)
    def test_retainFeatures_exceptionInversionFeatureName(self):
        """ Test retainFeatures() for ArgumentException when start comes after end as FeatureNames"""
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.retainFeatures(start="two", end="one")


    def test_retainFeatures_rangeIntoFEmpty(self):
        """ Test retainFeatures() retains all features using ranges """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        expTest = self.constructor(data, featureNames=featureNames)
        toTest.retainFeatures(start=0, end=2)

        assert toTest.isIdentical(expTest)


    def test_retainFeatures_handmadeRange(self):
        """ Test retainFeatures() against handmade output for range retention """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        toTest.retainFeatures(start=1, end=2)

        expectedTest = self.constructor([[2, 3], [5, 6], [8, 9]])

        assert expectedTest.isIdentical(toTest)

    def test_retainFeatures_range_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        toTest.retainFeatures(start=1, end=2)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_retainFeatures_handmadeWithFeatureNames(self):
        """ Test retainFeatures() against handmade output for range retention with FeatureNames """
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures(start=1, end=2)

        expectedTest = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=["two", "three"])

        assert expectedTest.isIdentical(toTest)

    def test_retainFeatures_handmade_calling_featureNames(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures(start="two", end="three")

        expectedTest = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=["two", "three"])

        assert expectedTest.isIdentical(toTest)

    def test_retainFeatures_handmadeString(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['p1', 'p2', 'p3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('p2=5')
        expectedTest = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])

        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('p1==1')
        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])

        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('p3<9')
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])

        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('p3<=8')
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])

        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('p3>8')
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])

        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('p3>8.5')
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])

        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('p1!=1.0')
        expectedTest = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])

        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('p1<1')
        expectedTest = self.constructor([[], [], []], pointNames=pointNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('p1>0')
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

    def test_retainFeatures_handmadeStringWithOperatorWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['p1', 'p2', 'p3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('p2 = 5')
        expectedTest = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])

        assert expectedTest.isIdentical(toTest)

        #test featureName==value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('p1 == 1')
        expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])

        assert expectedTest.isIdentical(toTest)

        #test featureName<value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('p3 < 9')
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])

        assert expectedTest.isIdentical(toTest)

        #test featureName<=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('p3 <= 8')
        expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])

        assert expectedTest.isIdentical(toTest)

        #test featureName>value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('p3 > 8')
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])

        assert expectedTest.isIdentical(toTest)

        #test featureName>=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('p3 > 8.5')
        expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])

        assert expectedTest.isIdentical(toTest)

        #test featureName!=value
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('p1 != 1.0')
        expectedTest = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])

        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back an empty
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('p1 < 1')
        expectedTest = self.constructor([[], [], []], pointNames=pointNames)

        assert expectedTest.isIdentical(toTest)

        #test featureName<value and return back all data
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('p1 > 0')
        expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        assert expectedTest.isIdentical(toTest)

    def test_retainFeatures_handmadeStringWithPointWhitespace(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['pt 1', 'pt 2', 'pt 3']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        #test pointName=value with no operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('pt 2=5')
        expectedTest = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])

        assert expectedTest.isIdentical(toTest)

        #test pointName=value with operator whitespace
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('pt 2 = 5')
        expectedTest = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])

        assert expectedTest.isIdentical(toTest)

    @raises(ArgumentException)
    def test_retainFeatures_handmadeString_pointNotExist(self):
        featureNames = ["one", "two", "three"]
        pointNames = ['1', '4', '7']
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.retainFeatures('5=1')

    def test_retainFeatures_numberOnly(self):
        self.back_retain_numberOnly('feature')

    def test_retainFeatures_functionAndNumber(self):
        self.back_retain_functionAndNumber('feature')

    def test_retainFeatures_numberAndRandomizeAllData(self):
        self.back_retain_numberAndRandomizeAllData('feature')

    def test_retainFeatures_numberAndRandomizeSelectedData(self):
        self.back_retain_numberAndRandomizeSelectedData('feature')

    @raises(ArgumentException)
    def test_retainFeatures_randomizeNoNumber(self):
        self.back_structural_randomizeNoNumber('retain', 'feature')

    @raises(ArgumentException)
    def test_retainFeatures_list_numberGreaterThanTargeted(self):
        self.back_structural_list_numberGreaterThanTargeted('retain', 'feature')

    @raises(ArgumentException)
    def test_retainFeatures_function_numberGreaterThanTargeted(self):
        self.back_structural_function_numberGreaterThanTargeted('retain', 'feature')

    @raises(ArgumentException)
    def test_retainFeatures_range_numberGreaterThanTargeted(self):
        self.back_structural_range_numberGreaterThanTargeted('retain', 'feature')

    #####################
    # referenceDataFrom #
    #####################

    @raises(ArgumentException)
    def test_referenceDataFrom_exceptionWrongType(self):
        """ Test referenceDataFrom() throws exception when other is not the same type """
        data1 = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        featureNames = ['one', 'two', 'three']
        pNames = ['1', 'one', '2', '0']
        orig = self.constructor(data1, pointNames=pNames, featureNames=featureNames)

        retType0 = UML.data.available[0]
        retType1 = UML.data.available[1]

        objType0 = UML.createData(retType0, data1, pointNames=pNames, featureNames=featureNames)
        objType1 = UML.createData(retType1, data1, pointNames=pNames, featureNames=featureNames)

        # at least one of these two will be the wrong type
        orig.referenceDataFrom(objType0)
        orig.referenceDataFrom(objType1)


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
        assert '-1' in orig.getPointNames()
        assert '1' in orig.getFeatureNames()
        assert ret is None

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


    def test_referenceDataFrom_allMetadataAttributes(self):
        data1 = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        featureNames = ['one', 'two', 'three']
        pNames = ['1', 'one', '2', '0']
        orig = self.constructor(data1, pointNames=pNames, featureNames=featureNames)

        data2 = [[-1, -2, -3, 4, 5, 3, ], [-1, -2, -3, 4, 5, 3, ]]
        other = self.constructor(data2, )

        orig.referenceDataFrom(other)

        assert orig._pointCount == other.points
        assert orig._featureCount == other.features

        assert orig._nextDefaultValuePoint == other._nextDefaultValuePoint
        assert orig._nextDefaultValueFeature == other._nextDefaultValueFeature


    ########################
    # transformEachPoint() #
    ########################

    @raises(ArgumentException)
    def test_transformEachPoint_exceptionInputNone(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), featureNames=featureNames)
        origObj.transformEachPoint(None)

    @raises(ImproperActionException)
    def test_transformEachPoint_exceptionPEmpty(self):
        data = [[], []]
        data = numpy.array(data).T
        origObj = self.constructor(data)

        def emitLower(point):
            return point[origObj.getFeatureIndex('deci')]

        origObj.transformEachPoint(emitLower)

    @raises(ImproperActionException)
    def test_transformEachPoint_exceptionFEmpty(self):
        data = [[], []]
        data = numpy.array(data)
        origObj = self.constructor(data)

        def emitLower(point):
            return point[origObj.getFeatureIndex('deci')]

        origObj.transformEachPoint(emitLower)

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_transformEachPoint_calls_constructIndicesList(self, mockFunc):
        toTest = self.constructor([[1,2,],[3,4]], pointNames=['a', 'b'])

        toTest.transformEachPoint(noChange, points=['a', 'b'])

    def test_transformEachPoint_Handmade(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitAllDeci(point):
            value = point[origObj.getFeatureIndex('deci')]
            return [value, value, value]

        lowerCounts = origObj.transformEachPoint(emitAllDeci)  # RET CHECK

        expectedOut = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
        exp = self.constructor(expectedOut, pointNames=pointNames, featureNames=featureNames)

        assert lowerCounts is None
        assert origObj.isIdentical(exp)

    def test_transformEachPoint_NamePath_preservation(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitAllDeci(point):
            value = point[toTest.getFeatureIndex('deci')]
            return [value, value, value]

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.transformEachPoint(emitAllDeci)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'

    def test_transformEachPoint_HandmadeLimited(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitAllDeci(point):
            value = point[origObj.getFeatureIndex('deci')]
            return [value, value, value]

        origObj.transformEachPoint(emitAllDeci, points=[3, 'two'])

        expectedOut = [[1, 0.1, 0.01], [1, 0.1, 0.02], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
        exp = self.constructor(expectedOut, pointNames=pointNames, featureNames=featureNames)

        assert origObj.isIdentical(exp)


    def test_transformEachPoint_nonZeroIterAndLen(self):
        origData = [[1, 1, 1], [1, 0, 2], [1, 1, 0], [0, 2, 0]]
        origObj = self.constructor(deepcopy(origData))

        def emitNumNZ(point):
            ret = 0
            assert len(point) == 3
            for value in point.nonZeroIterator():
                ret += 1
            return [ret, ret, ret]

        origObj.transformEachPoint(emitNumNZ)

        expectedOut = [[3, 3, 3], [2, 2, 2], [2, 2, 2], [1, 1, 1]]
        exp = self.constructor(expectedOut)

        assert origObj.isIdentical(exp)


    ##########################
    # transformEachFeature() #
    ##########################

    @raises(ImproperActionException)
    def test_transformEachFeature_exceptionPEmpty(self):
        data = [[], []]
        data = numpy.array(data).T
        origObj = self.constructor(data)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return 0
            return 1

        origObj.transformEachFeature(emitAllEqual)

    @raises(ImproperActionException)
    def test_transformEachFeature_exceptionFEmpty(self):
        data = [[], []]
        data = numpy.array(data)
        origObj = self.constructor(data)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return 0
            return 1

        origObj.transformEachFeature(emitAllEqual)

    @raises(ArgumentException)
    def test_transformEachFeature_exceptionInputNone(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), featureNames=featureNames)
        origObj.transformEachFeature(None)

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_transformEachFeature_calls_constructIndicesList(self, mockFunc):
        toTest = self.constructor([[1,2,],[3,4]], featureNames=['a', 'b'])

        toTest.transformEachFeature(noChange, features=['a', 'b'])

    def test_transformEachFeature_Handmade(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return [0, 0, 0, 0]
            return [1, 1, 1, 1]

        lowerCounts = origObj.transformEachFeature(emitAllEqual)  # RET CHECK
        expectedOut = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]
        exp = self.constructor(expectedOut, pointNames=pointNames, featureNames=featureNames)

        assert lowerCounts is None
        assert origObj.isIdentical(exp)


    def test_transformEachFeature_NamePath_preservation(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return [0, 0, 0, 0]
            return [1, 1, 1, 1]

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.transformEachFeature(emitAllEqual)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_transformEachFeature_HandmadeLimited(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return [0, 0, 0, 0]
            return [1, 1, 1, 1]

        origObj.transformEachFeature(emitAllEqual, features=[0, 'centi'])
        expectedOut = [[1, 0.1, 0], [1, 0.1, 0], [1, 0.1, 0], [1, 0.2, 0]]
        exp = self.constructor(expectedOut, pointNames=pointNames, featureNames=featureNames)

        assert origObj.isIdentical(exp)


    def test_transformEachFeature_nonZeroIterAndLen(self):
        origData = [[1, 1, 1], [1, 0, 2], [1, 1, 0], [0, 2, 0]]
        origObj = self.constructor(deepcopy(origData))

        def emitNumNZ(feature):
            ret = 0
            assert len(feature) == 4
            for value in feature.nonZeroIterator():
                ret += 1
            return [ret, ret, ret, ret]

        origObj.transformEachFeature(emitNumNZ)

        expectedOut = [[3, 3, 2], [3, 3, 2], [3, 3, 2], [3, 3, 2]]
        exp = self.constructor(expectedOut)

        assert origObj.isIdentical(exp)


    ##########################
    # transformEachElement() #
    ##########################

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_transformEachElement_calls_constructIndicesList1(self, mockFunc):
        toTest = self.constructor([[1,2],[3,4]], pointNames=['a', 'b'])

        def noChange(point):
            return point

        toTest.transformEachElement(noChange, points=['a', 'b'])

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_transformEachElement_calls_constructIndicesList2(self, mockFunc):
        toTest = self.constructor([[1,2],[3,4]], featureNames=['a', 'b'])

        def noChange(point):
            return point

        toTest.transformEachElement(noChange, features=['a', 'b'])

    def test_transformEachElement_passthrough(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        ret = toTest.transformEachElement(passThrough)  # RET CHECK
        assert ret is None
        retRaw = toTest.copyAs(format="python list")

        assert [1, 2, 3] in retRaw
        assert [4, 5, 6] in retRaw
        assert [7, 8, 9] in retRaw


    def test_transformEachElement_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.transformEachElement(passThrough)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'


    def test_transformEachElement_plusOnePreserve(self):
        data = [[1, 0, 3], [0, 5, 6], [7, 0, 9]]
        toTest = self.constructor(data)

        toTest.transformEachElement(plusOne, preserveZeros=True)
        retRaw = toTest.copyAs(format="python list")

        assert [2, 0, 4] in retRaw
        assert [0, 6, 7] in retRaw
        assert [8, 0, 10] in retRaw


    def test_transformEachElement_plusOneExclude(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        toTest.transformEachElement(plusOneOnlyEven, skipNoneReturnValues=True)
        retRaw = toTest.copyAs(format="python list")

        assert [1, 3, 3] in retRaw
        assert [5, 5, 7] in retRaw
        assert [7, 9, 9] in retRaw


    def test_transformEachElement_plusOneLimited(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)

        toTest.transformEachElement(plusOneOnlyEven, points=1, features=[1, 'three'], skipNoneReturnValues=True)
        retRaw = toTest.copyAs(format="python list")

        assert [1, 2, 3] in retRaw
        assert [4, 5, 7] in retRaw
        assert [7, 8, 9] in retRaw


    def test_transformEachElement_DictionaryAllMapped(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {1:9, 2:8, 3:7, 4:6, 5:5, 6:4, 7:3, 8:2, 9:1}
        expData = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformEachElement(transformMapping)

        assert toTest.isIdentical(expTest)


    def test_transformEachElement_DictionaryAllMappedStrings(self):
        data = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {"a": 1, "b":2, "c":3, "d":4, "e":5, "f":6, "g":7, "h":8, "i": 9}
        expData = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformEachElement(transformMapping)

        assert toTest.isIdentical(expTest)


    def test_transformEachElement_DictionarySomeMapped(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {2:8, 8:2}
        expData = [[1, 8, 3], [4, 5, 6], [7, 2, 9]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformEachElement(transformMapping)

        assert toTest.isIdentical(expTest)


    def test_transformEachElement_DictionaryMappedNotInPoints(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {2:8, 8:2}
        expData = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformEachElement(transformMapping, points=1)

        assert toTest.isIdentical(expTest)


    def test_transformEachElement_DictionaryMappedNotInFeatures(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {2:8, 8:2}
        expData = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformEachElement(transformMapping, features=0)

        assert toTest.isIdentical(expTest)


    def test_transformEachElement_DictionaryPreserveZerosNoZeroMap(self):
        data = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {1:2}
        expData = [[0, 0, 0], [2, 2, 2], [0, 0, 0]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformEachElement(transformMapping, preserveZeros=True)

        assert toTest.isIdentical(expTest)


    def test_transformEachElement_DictionaryPreserveZerosZeroMapZero(self):
        data = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {0:0, 1:2}
        expData = [[0, 0, 0], [2, 2, 2], [0, 0, 0]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformEachElement(transformMapping, preserveZeros=True)

        assert toTest.isIdentical(expTest)


    def test_transformEachElement_DictionaryPreserveZerosZeroMapNonZero(self):
        data = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {0:100, 1:2}
        expData = [[0, 0, 0], [2, 2, 2], [0, 0, 0]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformEachElement(transformMapping, preserveZeros=True)

        assert toTest.isIdentical(expTest)


    def test_transformEachElement_DictionaryDoNotPreserveZerosZeroMapNonZero(self):
        data = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {0:100}
        expData = [[100, 100, 100], [1, 1, 1], [100, 100, 100]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformEachElement(transformMapping, preserveZeros=False)

        assert toTest.isIdentical(expTest)


    def test_transformEachElement_DictionarySkipNoneReturn(self):
        data = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {1:None}
        expData = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        expTest = self.constructor(expData, pointNames=pnames, featureNames=names)

        toTest.transformEachElement(transformMapping, skipNoneReturnValues=True)

        assert toTest.isIdentical(expTest)


    def test_transformEachElement_DictionaryDoNotSkipNoneReturn(self):
        data = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)
        transformMapping = {1:None}
        if self.returnType == "Sparse":
            # Sparse cannot contain None values
            expData = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
            expTest = self.constructor(expData, pointNames=pnames, featureNames=names)
        else:
            expData = [[0, 0, 0], [None, None, None], [0, 0, 0]]
            expTest = self.constructor(expData, pointNames=pnames, featureNames=names, treatAsMissing=None)
        toTest.transformEachElement(transformMapping, skipNoneReturnValues=False)

        assert toTest.isIdentical(expTest)

    ####################
    #### fillWith() ####
    ####################

    # fillWith(self, values, pointStart, featureStart, pointEnd, featureEnd)

    def test_fillWith_acceptableValues(self):
        raw = [[1, 2], [3, 4]]
        toTest = self.constructor(raw)

        try:
            toTest.fillWith(set([1, 3]), 0, 0, 0, 1)
            assert False  # expected ArgumentExcpetion
        except ArgumentException as ae:
            print(ae)
        except Exception:
            assert False  # expected ArgumentException

        try:
            toTest.fillWith(lambda x: x * x, 0, 0, 0, 1)
            assert False  # expected ArgumentExcpetion
        except ArgumentException as ae:
            print(ae)
        except Exception:
            assert False  # expected ArgumentException


    def test_fillWith_sizeMismatch(self):
        raw = [[1, 2], [3, 4]]
        toTest = self.constructor(raw)

        raw = [[-1, -2]]
        val = self.constructor(raw)

        try:
            toTest.fillWith(val, 0, 0, 1, 1)
            assert False  # expected ArgumentExcpetion
        except ArgumentException as ae:
            print(ae)
        except Exception:
            assert False  # expected ArgumentException

        val.transpose()

        try:
            toTest.fillWith(val, 0, 0, 1, 1)
            assert False  # expected ArgumentExcpetion
        except ArgumentException as ae:
            print(ae)
        except Exception:
            assert False  # expected ArgumentException


    def test_fillWith_invalidID(self):
        raw = [[1, 2], [3, 4]]
        toTest = self.constructor(raw)

        val = 1

        try:
            toTest.fillWith(val, "hello", 0, 1, 1)
            assert False  # expected ArgumentExcpetion
        except ArgumentException as ae:
            print(ae)
        except Exception:
            assert False  # expected ArgumentException
        try:
            toTest.fillWith(val, 0, "Wrong", 1, 1)
            assert False  # expected ArgumentExcpetion
        except ArgumentException as ae:
            print(ae)
        except Exception:
            assert False  # expected ArgumentException
        try:
            toTest.fillWith(val, 0, 0, 2, 1)
            assert False  # expected ArgumentExcpetion
        except ArgumentException as ae:
            print(ae)
        except Exception:
            assert False  # expected ArgumentException
        try:
            toTest.fillWith(val, 0, 0, 1, -12)
            assert False  # expected ArgumentExcpetion
        except ArgumentException as ae:
            print(ae)
        except Exception:
            assert False  # expected ArgumentException


    def test_fillWith_start_lessThan_end(self):
        raw = [[1, 2], [3, 4]]
        toTest = self.constructor(raw)

        val = 1

        try:
            toTest.fillWith(val, 1, 0, 0, 1)
            assert False  # expected ArgumentExcpetion
        except ArgumentException as ae:
            print(ae)
        except Exception:
            assert False  # expected ArgumentException
        try:
            toTest.fillWith(val, 0, 1, 1, 0)
            assert False  # expected ArgumentExcpetion
        except ArgumentException as ae:
            print(ae)
        except Exception:
            assert False  # expected ArgumentException


    def test_fillWith_fullObjectFill(self):
        raw = [[1, 2], [3, 4]]
        toTest = self.constructor(raw)

        arg = [[-1, -2], [-3, -4]]
        arg = self.constructor(arg)
        exp = arg.copy()

        ret = toTest.fillWith(arg, 0, 0, toTest.points - 1, toTest.features - 1)
        assert ret is None

        arg *= 10

        assert toTest == exp
        assert toTest != arg


    def test_fillWith_vectorFill(self):
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

        toTestP.fillWith(valP, 0, 0, 0, 1)
        assert toTestP == expP

        toTestF.fillWith(valF, 0, 0, 1, 0)
        assert toTestF == expF


    def test_fillWith_offsetSquare(self):
        raw = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
        base = self.constructor(raw)
        trialRaw = [[0, 0], [0, 0]]
        trial = self.constructor(trialRaw)

        leftCorner = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for p, f in leftCorner:
            toTest = base.copy()

            toTest.fillWith(trial, p, f, p + 1, f + 1)
            assert toTest[p, f] == 0
            assert toTest[p + 1, f] == 0
            assert toTest[p, f + 1] == 0
            assert toTest[p + 1, f + 1] == 0


    def test_fillWith_constants(self):
        toTest0 = self.constructor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        exp0 = self.constructor([[0, 1, 1], [0, 1, 1], [0, 0, 0]])
        toTest0.fillWith(1, 0, 1, 1, 2)
        assert toTest0 == exp0

        toTest1 = self.constructor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        exp1 = self.constructor([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
        toTest1.fillWith(0, 0, 1, 2, 1)
        assert toTest1 == exp1

        toTestI = self.constructor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        expi = self.constructor([[1, 0, 2], [0, 1, 0], [2, 0, 1]])
        toTestI.fillWith(2, 0, 2, 0, 2)
        toTestI.fillWith(2, 2, 0, 2, 0)
        assert toTestI == expi


    def test_fillWIth_differentType(self):
        raw = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
        fill = [[0, 0], [0, 0]]
        exp = [[0, 0, 13], [0, 0, 23], [31, 32, 33]]
        exp = self.constructor(exp)
        for t in UML.data.available:
            toTest = self.constructor(raw)
            arg = UML.createData(t, fill)
            toTest.fillWith(arg, 0, 0, 1, 1)
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
        checkMsg = True
        target = "flattenToOnePoint" if axis == 'point' else "flattenToOneFeature"

        pempty = self.constructor(numpy.empty((0,2)))
        exceptionHelper(pempty, target, [], ImproperActionException, checkMsg)

        fempty = self.constructor(numpy.empty((4,0)))
        exceptionHelper(fempty, target, [], ImproperActionException, checkMsg)

        trueEmpty = self.constructor(numpy.empty((0,0)))
        exceptionHelper(trueEmpty, target, [], ImproperActionException, checkMsg)


    # flatten single p/f - see name changes
    def test_flattenToOnePoint_vector(self):
        self.back_flatten_vector('point')

    def test_flattenToOneFeature_vector(self):
        self.back_flatten_vector('feature')

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
            testObj.transpose()
            expObj.transpose()

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
        keptAxisNames = copyObj.getFeatureNames() if axis == 'point' else copyObj.getPointNames()
        discardAxisNames = copyObj.getPointNames() if axis == 'point' else copyObj.getFeatureNames()
        check = testObj.getFeatureNames() if axis == 'point' else testObj.getPointNames()

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

    # excpetion: either axis empty
    def test_unflattenFromOnePoint_empty(self):
        self.back_unflatten_empty('point')

    def test_unflattenFromOneFeature_empty(self):
        self.back_unflatten_empty('feature')

    def back_unflatten_empty(self, axis):
        checkMsg = True
        target = "unflattenFromOnePoint" if axis == 'point' else "unflattenFromOneFeature"
        single = (0,2) if axis == 'point' else (2,0)

        singleEmpty = self.constructor(numpy.empty(single))
        exceptionHelper(singleEmpty, target, [2], ImproperActionException, checkMsg)

        trueEmpty = self.constructor(numpy.empty((0,0)))
        exceptionHelper(trueEmpty, target, [2], ImproperActionException, checkMsg)


    # exceptions: opposite vector, 2d data
    def test_unflattenFromOnePoint_wrongShape(self):
        self.back_unflatten_wrongShape('point')

    def test_unflattenFromOneFeature_wrongShape(self):
        self.back_unflatten_wrongShape('feature')

    def back_unflatten_wrongShape(self, axis):
        checkMsg = True
        target = "unflattenFromOnePoint" if axis == 'point' else "unflattenFromOneFeature"
        vecShape = (4,1) if axis == 'point' else (1,4)

        wrongVector = self.constructor(numpyRandom.rand(*vecShape))
        exceptionHelper(wrongVector, target, [2], ImproperActionException, checkMsg)

        rectangle = self.constructor(numpyRandom.rand(4,4))
        exceptionHelper(rectangle, target, [2], ImproperActionException, checkMsg)


    # excpetion: numPoints / numFeatures does not divide length of mega P/F
    def test_unflattenFromOnePoint_doesNotDivide(self):
        self.back_unflatten_doesNotDivide('point')

    def test_unflattenFromOneFeature_doesNotDivide(self):
        self.back_unflatten_doesNotDivide('feature')

    def back_unflatten_doesNotDivide(self, axis):
        checkMsg = True
        target = "unflattenFromOnePoint" if axis == 'point' else "unflattenFromOneFeature"
        primeLength = (1,7) if axis == 'point' else (7,1)
        divisableLength = (1,8) if axis == 'point' else (8,1)

        undivisable = self.constructor(numpyRandom.rand(*primeLength))
        exceptionHelper(undivisable, target, [2], ArgumentException, checkMsg)

        divisable = self.constructor(numpyRandom.rand(*divisableLength))
        exceptionHelper(divisable, target, [5], ArgumentException, checkMsg)


    # exception: unflattening would destroy an axis name
    def test_unflattenFromOnePoint_nameDestroyed(self):
        self.back_unflatten_nameDestroyed('point')

    def test_unflattenFromOneFeature_nameDestroyed(self):
        self.back_unflatten_nameDestroyed('feature')

    def back_unflatten_nameDestroyed(self, axis):
        checkMsg = True
        target = "unflattenFromOnePoint" if axis == 'point' else "unflattenFromOneFeature"
        vecShape = (1,4) if axis == 'point' else (4,1)
        data = numpyRandom.rand(*vecShape)

        # non-default name, flattened axis
        args = {"pointNames":["non-default"]} if axis == 'point' else {"featureNames":["non-default"]}
        testObj = self.constructor(data, **args)
        exceptionHelper(testObj, target, [2], ImproperActionException, checkMsg)

        # all non-default names, unflattened axis
        names = ["a", "b", "c", "d"]
        args = {"featureNames":names} if axis == 'point' else {"pointNames":names}
        testObj = self.constructor(data, **args)
        exceptionHelper(testObj, target, [2], ImproperActionException, checkMsg)

        # single non-default name, unflattened axis
        testObj = self.constructor(data)
        if axis == 'point':
            testObj.setFeatureName(1, "non-default")
        else:
            testObj.setPointName(2, "non-default")
        exceptionHelper(testObj, target, [2], ImproperActionException, checkMsg)

    # exception: unflattening would destroy an axis name
    def test_unflattenFromOnePoint_nameFormatInconsistent(self):
        self.back_unflatten_nameFormatInconsistent('point')

    def test_unflattenFromOneFeature_nameFormatInconsistent(self):
        self.back_unflatten_nameFormatInconsistent('feature')

    def back_unflatten_nameFormatInconsistent(self, axis):
        checkMsg = True
        target = "unflattenFromOnePoint" if axis == 'point' else "unflattenFromOneFeature"
        vecShape = (1,4) if axis == 'point' else (4,1)
        data = numpyRandom.rand(*vecShape)

        # unflattend axis, mix of default names and correctly formatted
        names = ["a | 1", "b | 1", "a | 2", "b | 2"]
        args = {"featureNames":names} if axis == 'point' else {"pointNames":names}
        testObj = self.constructor(data, **args)
        if axis == 'point':
            testObj.setFeatureName(1, None)
        else:
            testObj.setPointName(1, None)
        exceptionHelper(testObj, target, [2], ImproperActionException, checkMsg)

        # unflattened axis, inconsistent along original unflattened axis
        names = ["a | 1", "b | 1", "a | 2", "c | 2"]
        args = {"featureNames":names} if axis == 'point' else {"pointNames":names}
        testObj = self.constructor(data, **args)
        exceptionHelper(testObj, target, [2], ImproperActionException, checkMsg)

        # unflattened axis, inconsistent along original flattened axis
        names = ["a | 1", "b | 2", "a | 2", "b | 3"]
        args = {"featureNames":names} if axis == 'point' else {"pointNames":names}
        testObj = self.constructor(data, **args)
        exceptionHelper(testObj, target, [2], ImproperActionException, checkMsg)


    # unflatten something that was flattened - include name transformation
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

    # unflatten something that was flattend - include name transformation
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

    def back_unflatten_handmadeDefaultNames(self, axis):
        target = "unflattenFromOnePoint" if axis == 'point' else "unflattenFromOneFeature"
        raw = [[1, 10, 20, 2]]
        toTest = self.constructor(raw)
        expData = numpy.array([[1,10],[20,2]])

        if axis == 'point':
            exp = self.constructor(expData)
        else:
            toTest.transpose()
            exp = self.constructor(expData.T)

        getattr(toTest, target)(2)
        assert toTest == exp

        # check that the name conforms to the standards of how UML objects assign
        # default names
        def checkName(n):
            assert n.startswith(DEFAULT_PREFIX)
            assert int(n[len(DEFAULT_PREFIX):]) >= 0

        list(map(checkName, toTest.getPointNames()))
        list(map(checkName, toTest.getFeatureNames()))


    # random round trip
    def test_flatten_to_unflatten_point_roundTrip(self):
        self.back_flatten_to_unflatten_roundTrip('point')

    def test_flatten_to_unflatten_feature_roundTrip(self):
        self.back_flatten_to_unflatten_roundTrip('feature')

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


def exceptionHelper(testObj, target, args, wanted, checkMsg):
    try:
        getattr(testObj, target)(*args)
        assert False  # expected an exception
    except wanted as check:
        if checkMsg:
            print(check)


class StructureAll(StructureDataSafe, StructureModifying):
    pass
