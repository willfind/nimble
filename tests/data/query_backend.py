"""
Methods tested in this file (none modify the data):

pointCount, featureCount, isIdentical, writeFile, __getitem__,
pointView, featureView, view, containsZero, __eq__, __ne__, toString,
__repr__, points.similarities, features.similarities, points.statistics,
features.statistics, points.__iter__, features.__iter__,
iterateElements, inverse, solveLinearSystem, report, features.report
"""

import math
import tempfile
import os
import os.path
from functools import reduce
from copy import deepcopy
import re
import textwrap

import numpy as np
import pytest

import nimble
from nimble import match
from nimble.match import QueryString
from nimble.core.data import BaseView
from nimble.core.data._dataHelpers import formatIfNeeded
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.exceptions import ImproperObjectAction

from tests.helpers import raises
from tests.helpers import noLogEntryExpected, oneLogEntryExpected
from tests.helpers import assertNoNamesGenerated
from tests.helpers import assertCalled
from .baseObject import DataTestObject


preserveName = "PreserveTestName"
preserveAPath = os.path.join(os.getcwd(), "correct", "looking", "path")
preserveRPath = os.path.relpath(preserveAPath)
preservePair = (preserveAPath, preserveRPath)


def _fnames(num):
    ret = []
    for i in range(num):
        ret.append('f' + str(i))
    return ret


def _pnames(num):
    ret = []
    for i in range(num):
        ret.append('p' + str(i))
    return ret


class QueryBackend(DataTestObject):
    ##############
    # pointCount #
    ##############


    def test_pointCount_empty(self):
        """ test pointCount when given different kinds of emptiness """
        data = [[], []]
        dataPEmpty = np.array(data).T
        dataFEmpty = np.array(data)

        objPEmpty = self.constructor(dataPEmpty)
        objFEmpty = self.constructor(dataFEmpty)

        assert len(objPEmpty.points) == 0
        assert len(objFEmpty.points) == 2


    def test_pointCount_vectorTest(self):
        """ Test pointCount when we only have row or column vectors of data """
        dataR = [[1, 2, 3]]
        dataC = [[1], [2], [3]]

        toTestR = self.constructor(dataR)
        toTestC = self.constructor(dataC)

        rPoints = len(toTestR.points)
        cPoints = len(toTestC.points)

        assert rPoints == 1
        assert cPoints == 3
        assertNoNamesGenerated(toTestR)
        assertNoNamesGenerated(toTestC)


    #################
    # featureCount #
    #################


    def test_featureCount_empty(self):
        """ test featureCount when given different kinds of emptiness """
        data = [[], []]
        dataPEmpty = np.array(data).T
        dataFEmpty = np.array(data)

        pEmpty = self.constructor(dataPEmpty)
        fEmpty = self.constructor(dataFEmpty)

        assert len(pEmpty.features) == 2
        assert len(fEmpty.features) == 0


    def test_featureCount_vectorTest(self):
        """ Test featureCount when we only have row or column vectors of data """
        dataR = [[1, 2, 3]]
        dataC = [[1], [2], [3]]

        toTestR = self.constructor(dataR)
        toTestC = self.constructor(dataC)

        rFeatures = len(toTestR.features)
        cFeatures = len(toTestC.features)

        assert rFeatures == 3
        assert cFeatures == 1
        assertNoNamesGenerated(toTestR)
        assertNoNamesGenerated(toTestC)


    #################
    # isIdentical() #
    #################
    @noLogEntryExpected
    def test_isIdentical_False(self):
        """ Test isIdentical() against some non-equal input """
        toTest = self.constructor([[4, 5]])
        assert not toTest.isIdentical(self.constructor([[1, 1], [2, 2]]))
        assert not toTest.isIdentical(self.constructor([[1, 2, 3]]))
        assert not toTest.isIdentical(self.constructor([[1, 2]]))
        assertNoNamesGenerated(toTest)

    @noLogEntryExpected
    def test_isIdentical_FalseBozoTypes(self):
        """ Test isIdentical() against some non-equal input of crazy types """
        toTest = self.constructor([[4, 5]])
        assert not toTest.isIdentical(np.array([[1, 1], [2, 2]]))
        assert not toTest.isIdentical('self.constructor([[1,2,3]])')
        assert not toTest.isIdentical(toTest.isIdentical)
        assertNoNamesGenerated(toTest)

    @noLogEntryExpected
    def test_isIdentical_True(self):
        """ Test isIdentical() against some actually equal input """
        toTest1 = self.constructor([[4, 5]])
        toTest2 = self.constructor(deepcopy([[4, 5]]))
        assert toTest1.isIdentical(toTest2)
        assert toTest2.isIdentical(toTest1)
        assertNoNamesGenerated(toTest1)
        assertNoNamesGenerated(toTest2)

    def test_isIdentical_FalseWithNaN(self):
        """ Test isIdentical() against some non-equal input with nan"""
        toTest1 = self.constructor([[1, np.nan, 5]])
        toTest2 = self.constructor(deepcopy([[1, np.nan, 3]]))
        assert not toTest1.isIdentical(toTest2)
        assert not toTest2.isIdentical(toTest1)
        assertNoNamesGenerated(toTest1)
        assertNoNamesGenerated(toTest2)

    def test_isIdentical_TrueWithNaN(self):
        """ Test isIdentical() against some actually equal input with nan """
        toTest1 = self.constructor([[1, np.nan, 5]])
        toTest2 = self.constructor(deepcopy([[1, np.nan, 5]]))
        assert toTest1.isIdentical(toTest2)
        assert toTest2.isIdentical(toTest1)
        assertNoNamesGenerated(toTest1)
        assertNoNamesGenerated(toTest2)


    #############
    # writeFile #
    #############

    @noLogEntryExpected
    def test_writeFile_CSVhandmade(self):
        """ Test writeFile() for csv extension with both data and featureNames """
        # instantiate object
        data = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        pointNames = ['1', 'one', '2', '0']
        featureNames = ['one', 'two', 'three']
        toWrite = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        orig = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        with tempfile.NamedTemporaryFile(suffix=".csv") as tmpFile:
            toWrite.writeFile(tmpFile.name, fileFormat='csv', includeNames=True)

            # read it back into a different object, then test equality
            readObj = self.constructor(source=tmpFile.name)

        assert readObj.isIdentical(toWrite)
        assert toWrite.isIdentical(readObj)

        assert toWrite == orig

    def test_writeFile_CSVhandmade_output(self):
        # instantiate object
        data = [[1., 2., 3.], [0., 2., 4.], [0., 0., 0.]]
        pointNames = ['one', '2', '0']
        featureNames = ['one', 'two', 'three']
        toWrite = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        orig = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        # should be no leading blank lines and data values should be floats
        exp = "pointNames,one,two,three\none,1.0,2.0,3.0\n2,0.0,2.0,4.0\n0,0.0,0.0,0.0\n"
        with tempfile.NamedTemporaryFile(mode='w+', suffix=".csv") as tmpFile:
            toWrite.writeFile(tmpFile.name, fileFormat='csv', includeNames=True)
            tmpFile.seek(0)
            assert tmpFile.read() == exp


    def test_writeFile_CSVhandmade_lazyNameGeneration(self):
        # instantiate object
        data = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        toWrite = self.constructor(data)

        with tempfile.NamedTemporaryFile(suffix=".csv") as tmpFile:
            toWrite.writeFile(tmpFile.name, fileFormat='csv', includeNames=False)
            assertNoNamesGenerated(toWrite)

            toWrite.writeFile(tmpFile.name, fileFormat='csv')
            assertNoNamesGenerated(toWrite)

    def test_writeFile_CSV_excludeDefaultNames(self):
        data = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        pointNames = ['1', 'one', '2', '0']
        featureNames = ['one', 'two', 'three']

        def excludeAxis(axis):
            if axis == 'point':
                exclude = self.constructor(data, featureNames=featureNames)
                if isinstance(exclude, nimble.core.data.BaseView):
                    setter = exclude._source.points.setNames
                else:
                    setter = exclude.points.setNames
                count = len(exclude.points)
            else:
                exclude = self.constructor(data, pointNames=pointNames)
                if isinstance(exclude, nimble.core.data.BaseView):
                    setter = exclude._source.features.setNames
                else:
                    setter = exclude.features.setNames
                count = len(exclude.features)

            # increase the index of the default point name so that it will be
            # recognizable when we read in from the file.
            axisExclude = getattr(exclude, axis + 's')

            with tempfile.NamedTemporaryFile(suffix=".csv") as tmpFile:
                exclude.writeFile(tmpFile.name, fileFormat='csv', includeNames=True)

                # read it back into a different object, then test equality
                if axis == 'point':
                    readObj = self.constructor(source=tmpFile.name, featureNames=True)
                else:
                    readObj = self.constructor(source=tmpFile.name, pointNames=True)
            axisRead = getattr(readObj, axis + 's')
            # isIdentical will ignore default names, but we still want to
            # ensure everything else is a match
            assert readObj.isIdentical(exclude)
            assert exclude.isIdentical(readObj)
            assert axisRead.names is None

        excludeAxis('point')
        excludeAxis('feature')

    @noLogEntryExpected
    def test_writeFile_CSVhandmade_extraCommas(self):
        """ Test writeFile() when data and names contain commas """
        # instantiate object
        data = [[1, 2, 'a'], [1, 2, 'a,b'], [2, 4, 'a,b,c'], [0, 0, 'd']]
        pointNames = ['1', 'one,1', '2', '0,zero']
        featureNames = ['one,1', 'two', '3,three']
        toWrite = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        orig = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        with tempfile.NamedTemporaryFile(suffix=".csv") as tmpFile:
            toWrite.writeFile(tmpFile.name, fileFormat='csv', includeNames=True)

            # read it back into a different object, then test equality
            # must specify featureNames=True because 'automatic' will not detect
            readObj = self.constructor(source=tmpFile.name, featureNames=True)

        assert readObj.isIdentical(toWrite)
        assert toWrite.isIdentical(readObj)

        assert toWrite == orig

    @noLogEntryExpected
    def test_writeFile_CSVhandmade_extraQuotes(self):
        """ Test writeFile() when data and names contain commas """
        # instantiate object
        data = [[1, 2, 'with "quote"'], [1, 2, '"quotes","and", "commas"'],
                [2, 4, 'includes"quote"'], [0, 0, 'd']]
        pointNames = ['1', '1"quote"', '2', '0,zero']
        featureNames = ['"quote",1', 'two', '3,three']
        toWrite = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        orig = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        with tempfile.NamedTemporaryFile(suffix=".csv") as tmpFile:
            toWrite.writeFile(tmpFile.name, fileFormat='csv', includeNames=True)
            # read it back into a different object, then test equality
            # must specify featureNames=True because 'automatic' will not detect
            readObj = self.constructor(source=tmpFile.name, featureNames=True)
        assert readObj.isIdentical(toWrite)
        assert toWrite.isIdentical(readObj)

        assert toWrite == orig

    @noLogEntryExpected
    def test_writeFile_MTXhandmade(self):
        """ Test writeFile() for mtx extension with both data and featureNames """
        # instantiate object
        data = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        featureNames = ['one', 'two', 'three']
        pointNames = ['1', 'one', '2', '0']
        toWrite = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        with tempfile.NamedTemporaryFile(suffix=".mtx") as tmpFile:
            toWrite.writeFile(tmpFile.name, fileFormat='mtx', includeNames=True)

            # read it back into a different object, then test equality
            readObj = self.constructor(source=tmpFile.name)

        assert readObj.isIdentical(toWrite)
        assert toWrite.isIdentical(readObj)

    def test_writeFile_MTXhandmade_lazyNameGeneration(self):
        # instantiate object
        data = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        toWrite = self.constructor(data)

        with tempfile.NamedTemporaryFile(suffix=".mtx") as tmpFile:
            toWrite.writeFile(tmpFile.name, fileFormat='mtx', includeNames=False)

        assertNoNamesGenerated(toWrite)

    #################
    # save/LoadData #
    #################

    def test_save(self):
        data = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        featureNames = ['one', 'two', 'three']
        pointNames = ['1', 'one', '2', '0']
        toSave = self.constructor(data, pointNames=pointNames,
                                  featureNames=featureNames)

        with tempfile.NamedTemporaryFile(suffix=".pickle") as tmpFile:
            toSave.save(tmpFile.name)
            load1 = nimble.data(None, tmpFile.name)

        assert toSave.isIdentical(load1)
        assert load1.isIdentical(toSave)

        with tempfile.NamedTemporaryFile(suffix=".p") as tmpFile:
            toSave.save(tmpFile.name)
            load2 = nimble.data(None, tmpFile.name)

        assert toSave.isIdentical(load2)
        assert load2.isIdentical(toSave)

        with tempfile.NamedTemporaryFile() as tmpFile:
            toSave.save(tmpFile.name)
            load3 = nimble.data(None, tmpFile.name + ".pickle")

        assert toSave.isIdentical(load3)
        assert load3.isIdentical(toSave)

    def test_save_lazyNameGeneration(self):
        data = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        toSave = self.constructor(data)

        with tempfile.NamedTemporaryFile(suffix=".pickle") as tmpFile:
            toSave.save(tmpFile.name)

        assertNoNamesGenerated(toSave)

    def test_save_extensionHandling(self):
        data = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        featureNames = ['one', 'two', 'three']
        pointNames = ['1', 'one', '2', '0']
        toSave = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        with tempfile.NamedTemporaryFile(suffix=".pickle") as tmpFile:
            # save without extension to ensure the .pickle extension is added
            fileNameWithoutExtension = tmpFile.name[:-7]
            # it is important that the saved object has the same name as the
            # tmpFile otherwise a new file will be created and saved which will
            # not be cleaned up by tempfile.
            assert fileNameWithoutExtension + '.pickle' == tmpFile.name

            toSave.save(fileNameWithoutExtension)
            LoadObj = nimble.data(None, tmpFile.name)
            assert isinstance(LoadObj, nimble.core.data.Base)

            with raises(InvalidArgumentValue):
                LoadObj = nimble.data(None, fileNameWithoutExtension)

    @oneLogEntryExpected
    def test_saveAndLoad_logCount(self):
        data = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        featureNames = ['one', 'two', 'three']
        pointNames = ['1', 'one', '2', '0']
        toSave = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        with tempfile.NamedTemporaryFile(suffix=".pickle") as tmpFile:
            toSave.save(tmpFile.name)
            LoadObj = nimble.data(None, tmpFile.name)

    ##############
    # __getitem__#
    ##############
    @raises(KeyError)
    def test_getitem_exception_duplicateValuesPoint(self):
        raw = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        obj = self.constructor(raw)

        obj[[0, 1, 0], :]

    @raises(KeyError)
    def test_getitem_exception_duplicateValuesFeature(self):
        raw = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        obj = self.constructor(raw)

        obj[:, [1, 0, 1]]

    @raises(KeyError)
    def test_getitem_exception_mixedTypesPoint(self):
        raw = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        obj = self.constructor(raw)

        obj[[False, True, 2], :]

    @raises(KeyError)
    def test_getitem_exception_mixedTypesFeature(self):
        raw = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        obj = self.constructor(raw)

        obj[:, [False, True, 2]]

    @noLogEntryExpected
    def test_getitem_allExamples(self):
        """

        """
        featureNames = ["one", "two", "three", "zero", "gender"]
        pnames = ['1', '4', '7', '0']
        data = [[1, 2, 3, 0, 'f'], [4, 5, 0, 0, 'm'], [7, 0, 9, 0, 'f'], [0, 0, 0, 0, 'm']]

        toTest = self.constructor(data, pointNames=pnames, featureNames=featureNames)

        tmp1 = self.constructor(data[1], featureNames=featureNames, pointNames=[pnames[1]])
        assert toTest[1, :] == tmp1
        assert toTest['4', :] == tmp1
        assert toTest[0, 'gender'] == 'f'
        assert toTest[0, 4] == 'f'
        assert toTest['1', 'gender'] == 'f'

        tmp2 = self.constructor(data[1:], featureNames=featureNames, pointNames=pnames[1:])
        assert toTest[1:, :] == tmp2
        assert toTest[1:3, :] == tmp2
        assert toTest["4":"0", :] == tmp2
        assert toTest[[1,2,3], :] == tmp2
        assert toTest[['4', '7', '0'], :] == tmp2

        tmp3 = self.constructor([['f'], ['m'], ['f'], ['m']], featureNames=['gender'], pointNames=pnames)
        assert toTest[:, 4] == tmp3
        assert toTest[:, 'gender'] == tmp3

        tmp4 = self.constructor([['f', 0], ['m', 0], ['f', 0], ['m', 0]], featureNames=['gender', 'zero'], pointNames=pnames)
        assert toTest[:, 4:3:-1] == tmp4
        assert toTest[:, "gender":"zero":-1] == tmp4
        assert toTest[:, [4,3]] == tmp4
        assert toTest[:, ['gender', 'zero']] == tmp4

        tmp5 = self.constructor([['f', 0], ['m', 0]], featureNames=['gender', 'zero'], pointNames=pnames[:2])
        assert toTest[:1, 4:3:-1] == tmp5
        assert toTest[:"4", "gender":"zero":-1] == tmp5
        assert toTest[['1', '4'], [4,3]] == tmp5
        assert toTest[[0,1], ['gender', 'zero']] == tmp5


    def test_getitem_simpleExampleWithZeroes(self):
        """ Test __getitem__ returns the correct output for a number of simple queries """
        featureNames = ["one", "two", "three", "zero"]
        pnames = ['1', '4', '7', '0']
        data = [[1, 2, 3, 0], [4, 5, 0, 0], [7, 0, 9, 0], [0, 0, 0, 0]]

        toTestInt = self.constructor(data, pointNames=pnames, featureNames=featureNames)

        assert toTestInt[0, 0] == 1
        assert isinstance(toTestInt[0, 0], (int, np.integer))
        assert toTestInt[1, 3] == 0
        assert isinstance(toTestInt[1, 3], (int, np.integer))
        assert toTestInt['7', 2] == 9
        assert isinstance(toTestInt['7', 2], (int, np.integer))
        assert toTestInt[3, 'zero'] == 0
        assert isinstance(toTestInt[3, 'zero'], (int, np.integer))
        assert toTestInt[1, 'one'] == 4
        assert isinstance(toTestInt[1, 'one'], (int, np.integer))

        data = [[1., 2., 3., 0.], [4., 5., 0., 0.], [7., 0., 9., 0.], [0., 0., 0., 0.]]

        toTestFloat = self.constructor(data, pointNames=pnames, featureNames=featureNames)

        assert toTestFloat[0, 0] == 1
        assert isinstance(toTestFloat[0, 0], (float, np.floating))
        assert toTestFloat[1, 3] == 0
        assert isinstance(toTestFloat[1, 3], (float, np.floating))
        assert toTestFloat['7', 2] == 9
        assert isinstance(toTestFloat['7', 2], (float, np.floating))
        assert toTestFloat[3, 'zero'] == 0
        assert isinstance(toTestFloat[3, 'zero'], (float, np.floating))
        assert toTestFloat[1, 'one'] == 4
        assert isinstance(toTestFloat[1, 'one'], (float, np.floating))

    @raises(KeyError)
    def test_getitem_nonIntConvertableFloatSingleKey(self):
        data = [[0, 1, 2, 3]]
        toTest = self.constructor(data)

        assert toTest[0.1] == 0

    @raises(KeyError)
    def test_getitem_nonIntConvertableFloatTupleKey(self):
        data = [[0, 1], [2, 3]]
        toTest = self.constructor(data)

        assert toTest[0, 1.1] == 1

    def test_getitem_floatKeys(self):
        """ Test __getitem__ correctly interprets float valued keys """
        featureNames = ["one", "two", "three", "zero"]
        pnames = ['1', '4', '7', '0']
        data = [[1, 2, 3, 0], [4, 5, 0, 0], [7, 0, 9, 0], [0, 0, 0, 0]]

        toTest = self.constructor(data, pointNames=pnames, featureNames=featureNames)

        assert toTest[0.0, 0] == 1
        assert toTest[1.0, 3.0] == 0

        data = [[0, 1, 2, 3]]
        toTest = self.constructor(data)

        assert toTest[0.0] == 0
        assert toTest[1.0] == 1

    def test_getitem_floatKeysInList(self):
        """ Test __getitem__ correctly interprets a list of float valued keys """
        featureNames = ["one", "two", "three", "zero"]
        pnames = ['1', '4', '7', '0']
        data = [[1, 2, 3, 0], [4, 5, 0, 0], [7, 0, 9, 0], [0, 0, 0, 0]]

        toTest = self.constructor(data, pointNames=pnames, featureNames=featureNames)
        exp = self.constructor([[2, 0], [0, 0]], pointNames=['1', '7'], featureNames=['two', 'zero'])
        assert toTest[[0.0, 2.0], [1.0, 3.0]] == exp

        data = [[0, 1, 2, 3]]
        toTest = self.constructor(data)
        exp = self.constructor([[0, 1]])
        assert toTest[[0.0, 1.0]] == exp

    def test_getitem_SinglePoint(self):
        """ Test __getitem__ has vector style access for one point object """
        pnames = ['single']
        fnames = ['a', 'b', 'c', 'd', 'e']
        data = [[0, 1, 2, 3, 10]]
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)

        assert toTest[0] == 0
        assert toTest['a'] == 0
        assert toTest[1] == 1
        assert toTest['b'] == 1
        assert toTest[2] == 2
        assert toTest['c'] == 2
        assert toTest[3] == 3
        assert toTest['d'] == 3
        assert toTest[4] == 10
        assert toTest['e'] == 10

    def test_getitem_SingleFeature(self):
        """ Test __getitem__ has vector style access for one feature object """
        fnames = ['single']
        pnames = ['a', 'b', 'c', 'd', 'e']
        data = [[0], [1], [2], [3], [10]]
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)

        assert toTest[0] == 0
        assert toTest['a'] == 0
        assert toTest[1] == 1
        assert toTest['b'] == 1
        assert toTest[2] == 2
        assert toTest['c'] == 2
        assert toTest[3] == 3
        assert toTest['d'] == 3
        assert toTest[4] == 10
        assert toTest['e'] == 10

    def test_getitem_vectorsPlayNiceWithPythonKeywords(self):
        dataPV = [[-1, 2, 3, 4, 5]]
        dataFV = [[-1], [2], [3], [4], [5]]

        pv = self.constructor(dataPV)
        fv = self.constructor(dataFV)

        assert all(pv)
        assert all(fv)

        assert any(pv)
        assert any(fv)

        assert [x for x in pv if x > 0] == [2, 3, 4, 5]
        assert [x for x in fv if x > 0] == [2, 3, 4, 5]

        assert list(map(abs, pv)) == [1, 2, 3, 4, 5]
        assert list(map(abs, fv)) == [1, 2, 3, 4, 5]

        assert max(pv) == 5
        assert max(fv) == 5

        assert min(pv) == -1
        assert min(fv) == -1

        assert reduce(lambda x, y: x + y, pv) == 13
        assert reduce(lambda x, y: x + y, fv) == 13

        assert set(pv) == set([-1, 2, 3, 4, 5])
        assert set(fv) == set([-1, 2, 3, 4, 5])

        assert sorted(pv, reverse=True) == [5, 4, 3, 2, -1]
        assert sorted(fv, reverse=True) == [5, 4, 3, 2, -1]

        assert sum(pv) == 13
        assert sum(fv) == 13

        assert tuple(pv) == (-1, 2, 3, 4, 5)
        assert tuple(fv) == (-1, 2, 3, 4, 5)

        assert list(zip(pv, fv)) == [(-1, -1), (2, 2), (3, 3), (4, 4), (5, 5)]


    @raises(InvalidArgumentType)
    def test_getitem_exception_nimble2D(self):
        raw = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        obj = self.constructor(raw)

        rawIdx = [[1, 2, 3], [4, 5, 6]]
        idxObj = self.constructor(rawIdx)

        obj[idxObj]

    @raises(InvalidArgumentType)
    def test_getitem_exception_nimbleBoolean2D(self):
        raw = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        obj = self.constructor(raw)

        idxObj = obj.matchingElements(lambda x: x % 2 == 0)

        obj[idxObj]

    @raises(KeyError)
    def test_getitem_exception_nimbleBooleanPointVectorShape(self):
        raw = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        obj = self.constructor(raw)

        idxObj = obj.points.matching(lambda pt: sum(pt) > 6)
        idxObjShort = idxObj[:1]

        obj[idxObjShort, :]

    @raises(KeyError)
    def test_getitem_exception_nimbleBooleanFeatureVectorShape(self):
        raw = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        obj = self.constructor(raw)

        idxObj = obj.features.matching(lambda ft: sum(ft) > 12)
        idxObjShort = idxObj[:1]

        obj[:, idxObjShort]

    def test_getitem_nimbleBooleanPointVector(self):
        raw = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        obj = self.constructor(raw)

        idxObj = obj.points.matching(lambda pt: sum(pt) > 6)
        assert isinstance(idxObj[0], (bool, np.bool_))
        ret = obj[idxObj, :]

        expData = [[4, 5, 6], [7, 8, 9]]
        exp = self.constructor(expData)

        assert ret == exp

    def test_getitem_nimbleBooleanFeatureVector(self):
        raw = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        obj = self.constructor(raw)

        idxObj = obj.features.matching(lambda ft: sum(ft) < 18)
        assert isinstance(idxObj[0], (bool, np.bool_))
        ret = obj[:, idxObj]

        expData = [[1, 2], [4, 5], [7, 8]]
        exp = self.constructor(expData)

        assert ret == exp


    ###############################
    # points/features.__getitem__ #
    ###############################
    @noLogEntryExpected
    def test_pf_getitem(self):
        featureNames = ["one", "two", "three", "zero", "gender"]
        pnames = ['1', '4', '7', '0']
        data = [[1, 2, 3, 0, 'f'], [4, 5, 0, 0, 'm'], [7, 0, 9, 0, 'f'], [0, 0, 0, 0, 'm']]

        toTest = self.constructor(data, pointNames=pnames, featureNames=featureNames)

        tmp1 = self.constructor(data[1], featureNames=featureNames, pointNames=[pnames[1]])
        assert toTest.points[1] == tmp1
        assert toTest.points['4'] == tmp1

        tmp2 = self.constructor(data[1:], featureNames=featureNames, pointNames=pnames[1:])
        assert toTest.points[1:] == tmp2
        assert toTest.points[1:3] == tmp2
        assert toTest.points["4":"0"] == tmp2
        assert toTest.points[[1,2,3]] == tmp2
        assert toTest.points[['4', '7', '0']] == tmp2
        assert toTest.points[0:3] == toTest

        tmp3 = self.constructor([['f'], ['m'], ['f'], ['m']], featureNames=['gender'], pointNames=pnames)
        assert toTest.features[4] == tmp3
        assert toTest.features['gender'] == tmp3

        tmp4 = self.constructor([['f', 0], ['m', 0], ['f', 0], ['m', 0]], featureNames=['gender', 'zero'], pointNames=pnames)
        assert toTest.features[4:3:-1] == tmp4
        assert toTest.features["gender":"zero":-1] == tmp4
        assert toTest.features[[4,3]] == tmp4
        assert toTest.features[['gender', 'zero']] == tmp4

        tmp5 = self.constructor([['f', 0], ['m', 0]], featureNames=['gender', 'zero'], pointNames=pnames[:2])
        assert toTest.points[:1].features[4:3:-1] == tmp5
        assert toTest.points[:"4"].features["gender":"zero":-1] == tmp5
        assert toTest.points[['1', '4']].features[[4,3]] == tmp5
        assert toTest.points[[0,1]].features[['gender', 'zero']] == tmp5

    @raises(KeyError)
    def test_points_getitem_exception_duplicateValues(self):
        raw = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        obj = self.constructor(raw)

        obj.points[[0, 1, 0]]

    @raises(KeyError)
    def test_features_getitem_exception_duplicateValues(self):
        raw = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        obj = self.constructor(raw)

        obj.features[[1, 0, 1]]

    def test_points_getitem_nimbleBooleanPointVector(self):
        raw = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        obj = self.constructor(raw)

        idxObj = obj.points.matching(lambda pt: sum(pt) > 6)
        assert isinstance(idxObj[0], (bool, np.bool_))
        ret = obj.points[idxObj]

        expData = [[4, 5, 6], [7, 8, 9]]
        exp = self.constructor(expData)

        assert ret == exp

    def test_points_getitem_nimbleBooleanFeatureVector(self):
        raw = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        obj = self.constructor(raw)

        idxObj = obj.features.matching(lambda ft: sum(ft) < 18)
        assert isinstance(idxObj[0], (bool, np.bool_))
        ret = obj.points[idxObj]

        expData = [[1, 2, 3], [4, 5, 6]]
        exp = self.constructor(expData)

        assert ret == exp

    def test_features_getitem_nimbleBooleanPointVector(self):
        raw = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        obj = self.constructor(raw)

        idxObj = obj.points.matching(lambda pt: sum(pt) > 6)
        assert isinstance(idxObj[0], (bool, np.bool_))
        ret = obj.features[idxObj]

        expData = [[2, 3], [5, 6], [8, 9]]
        exp = self.constructor(expData)

        assert ret == exp

    def test_features_getitem_nimbleBooleanFeatureVector(self):
        raw = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        obj = self.constructor(raw)

        idxObj = obj.features.matching(lambda ft: sum(ft) < 18)
        assert isinstance(idxObj[0], (bool, np.bool_))
        ret = obj.features[idxObj]

        expData = [[1, 2], [4, 5], [7, 8]]
        exp = self.constructor(expData)

        assert ret == exp

    ################
    # pointView #
    ################
    @noLogEntryExpected
    def test_pointView_FEmpty(self):
        """ Test pointView() when accessing a feature empty object """
        data = [[], []]
        data = np.array(data)
        toTest = self.constructor(data)

        v = toTest.pointView(0)

        assert len(v.features) == 0

    @noLogEntryExpected
    def test_pointView_isinstance(self):
        pointNames = ['1', '4', '7']
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        pView = toTest.pointView(0)

        assert isinstance(pView, BaseView)
        assert pView.name is None
        assert len(pView.points) == 1
        assert len(pView.features) == 3
        assert len(pView) == len(toTest.features)
        assert pView[0] == 1
        assert pView['two'] == 2
        assert pView['three'] == 3

    ##################
    # featureView #
    ##################
    @noLogEntryExpected
    def test_featureView_FEmpty(self):
        """ Test featureView() when accessing a point empty object """
        data = [[], []]
        data = np.array(data).T
        toTest = self.constructor(data)

        v = toTest.featureView(0)

        assert len(v.points) == 0

    @noLogEntryExpected
    def test_featureView_isinstance(self):
        """ Test featureView() returns an instance of the BaseView """
        pointNames = ['1', '4', '7']
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        fView = toTest.featureView('one')

        assert isinstance(fView, BaseView)
        assert fView.name is None
        assert len(fView.points) == 3
        assert len(fView.features) == 1
        assert len(fView) == len(toTest.points)
        assert fView[0] == 1
        assert fView['4'] == 4
        assert fView['7'] == 7

    ########
    # view #
    ########

    # This only checks the api for view creation does the intended validation.
    # That the returned View object then acts like everything else in the
    # data hierarchy is tested using test objects, in the same style as the
    # concrete data types.

    def test_view_pointStart_pointEnd_validation(self):
        pointNames = ['1', '4', '7']
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        textCheck = False
        with raises(InvalidArgumentType) as exc:
            # pointStart is non-ID didn't raise exception
            toTest.view(pointStart=1.5)
        if textCheck:
            print(exc)

        with raises(IndexError) as exc:
            # pointEnd > pointCount didn't raise exception
            toTest.view(pointEnd=5)
        if textCheck:
            print(exc)

        with raises(InvalidArgumentType) as exc:
            # pointEnd is non-ID didn't raise exception
            toTest.view(pointEnd=1.4)
        if textCheck:
            print(exc)

        with raises(InvalidArgumentValueCombination) as exc:
            # pointStart > pointEnd didn't raise exception
            toTest.view(pointStart='7', pointEnd='4')
        if textCheck:
            print(exc)

    def test_view_featureStart_featureEnd_validation(self):
        pointNames = ['1', '4', '7']
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        textCheck = False
        with raises(InvalidArgumentType) as exc:
            # featureStart is non-ID didn't raise exception
            toTest.view(featureStart=1.5)
        if textCheck:
            print(exc)

        with raises(IndexError) as exc:
            # featureEnd > featureCount didn't raise exception
            toTest.view(featureEnd=4)
        if textCheck:
            print(exc)

        with raises(InvalidArgumentType) as exc:
            # featureEnd is non-ID didn't raise exception
            toTest.view(featureEnd=1.4)
        if textCheck:
            print(exc)

        with raises(InvalidArgumentValueCombination) as exc:
            # featureStart > featureEnd didn't raise exception
            toTest.view(featureStart='three', featureEnd='two')
        if textCheck:
            print(exc)

    @noLogEntryExpected
    def test_ViewAccess_AllLimits(self):
        data = [[11, 12, 13, 14], [0, 0, 0, 0], [21, 22, 23, 24], [0, 0, 0, 0], [31, 32, 33, 34]]
        pnames = ['p1', 'p2', 'p3', 'p4', 'p5']
        fnames = ['f1', 'f2', 'f3', 'f4']

        origPLen = len(data)
        origFLen = len(data[0])

        orig = self.constructor(data, pointNames=pnames, featureNames=fnames)

        def checkAccess(v, pStart, pEnd, fStart, fEnd):
            pSize = pEnd - pStart
            fSize = fEnd - fStart
            for i in range(pSize):
                for j in range(fSize):
                    assert v[i, j] == orig[i + pStart, j + fStart]

            # points.getNames
            # +1 since pEnd inclusive when calling .view, array splices are exclusive
            assert v.points.getNames() == pnames[pStart:pEnd + 1]
            # points.getIndex, points.getName
            for i in range(pStart, pEnd):
                origName = pnames[i]
                assert v.points.getName(i - pStart) == origName
                assert v.points.getIndex(origName) == i - pStart

            # features.getName
            # +1 since fEnd inclusive when calling .view, array splices are exclusive
            assert v.features.getNames() == fnames[fStart:fEnd + 1]

            # features.getIndex, features.getName
            for i in range(fStart, fEnd):
                origName = fnames[i]
                assert v.features.getName(i - fStart) == origName
                assert v.features.getIndex(origName) == i - fStart

        for pStart in range(origPLen):
            for pEnd in range(pStart, origPLen):
                for fStart in range(origFLen):
                    for fEnd in range(fStart, origFLen):
                        testView = orig.view(pointStart=pStart, pointEnd=pEnd,
                                             featureStart=fStart, featureEnd=fEnd)
                        checkAccess(testView, pStart, pEnd, fStart, fEnd)


    ################
    # containsZero #
    ################

    @noLogEntryExpected
    def test_containsZero_simple(self):
        """ Test containsZero works as expected on simple numerical data """
        dataAll = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        dataSome = [[1, 1, 1], [0.0, 1, -4], [2, 2, 2]]
        dataNone = [[1.1, 2, 12], [-2, -3, -4], [.0001, .000003, .00000004]]

        dAll = self.constructor(dataAll)
        dSome = self.constructor(dataSome)
        dNone = self.constructor(dataNone)

        assert dAll.containsZero() is True
        assert dSome.containsZero() is True
        assert dNone.containsZero() is False
        assertNoNamesGenerated(dAll)
        assertNoNamesGenerated(dSome)
        assertNoNamesGenerated(dNone)

    ##########
    # __eq__ #
    ##########

    def test_eq__exactlyisIdentical(self):
        """ Test that __eq__ relies on isIdentical """

        class FlagWrap(object):
            flag = False

        flag1 = FlagWrap()
        flag2 = FlagWrap()

        def fake(other):
            if flag1.flag:
                flag2.flag = True
            flag1.flag = True
            return True

        toTest1 = self.constructor([[4, 5]])
        toTest2 = self.constructor(deepcopy([[4, 5]]))

        toTest1.isIdentical = fake
        toTest2.isIdentical = fake

        assert toTest1 == toTest2
        assert toTest2 == toTest1
        assert flag1.flag
        assert flag2.flag

    ##########
    # __ne__ #
    ##########

    def test_ne__exactly__eq__(self):
        """ Test that __ne__ relies on __eq__ """

        class FlagWrap(object):
            flag = False

        flag1 = FlagWrap()
        flag2 = FlagWrap()

        def fake(other):
            if flag1.flag:
                flag2.flag = True
            flag1.flag = True
            return False

        toTest1 = self.constructor([[4, 5]])
        toTest2 = self.constructor(deepcopy([[4, 5]]))

        toTest1.__eq__ = fake
        toTest2.__eq__ = fake

        assert toTest1 != toTest2
        assert toTest2 != toTest1
        assert flag1.flag
        assert flag2.flag


    ############
    # toString #
    ############

    @pytest.mark.slow
    def test_toString_nameAndValRecreation_randomized(self):
        """ Regression test with random data and limits. Recreates expected results """
        for pNum in [3, 9]:
            for fNum in [2, 5, 8, 15]:
                randGen = nimble.random.data("List", pNum, fNum, 0)
                raw = randGen._data

                fnames = ['fn0', 'fn1', 'fn2', 'fn3', 'fn4', 'fn5', 'fn6', 'fn7', 'fn8', 'fn9', 'fna', 'fnb', 'fnc',
                          'fnd', 'fne']
                #				fnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e']
                pnames = ['pn0', 'pn1', 'pn2', 'pn3', 'pn4', 'pn5', 'pn6', 'pn7', 'pn8', 'pn9', 'pna', 'pnb', 'pnc',
                          'pnd', 'pne']
                data = self.constructor(raw, pointNames=pnames[:pNum], featureNames=fnames[:fNum])

                for mw in [40, 60, 80, None]:
                    for mh in [5, 7, 10, None]:
                        for inc in [False, True]:
                            ret = data.toString(includeNames=inc, maxWidth=mw, maxHeight=mh)
                            checkToStringRet(ret, data, inc, mw, mh)

    def test_toString_lazyNameGeneration(self):
        data = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        ret = data.toString(includeNames=False)
        assertNoNamesGenerated(data)
        ret = data.toString(includeNames=True)
        assertNoNamesGenerated(data)

        empty = self.constructor([])
        ret = empty.toString()
        assertNoNamesGenerated(empty)

    def test_toString_nameAndValRecreation_randomized_longNames(self):
        """ Test long point and feature names do not exceed max width"""
        randGen = nimble.random.data("List", 9, 9, 0)
        raw = randGen._data

        suffix = [1, 22, 333, 4444, 55555, 666666, 7777777, 88888888, 999999999]
        fnames = ['feature_' + str(x) for x in suffix]
        pnames = ['point_' + str(x) for x in suffix]
        data = self.constructor(raw, pointNames=pnames, featureNames=fnames)

        for mw in [40, 60, 80, None]:
            for mh in [5, 7, 10, None]:
                for inc in [False, True]:
                    ret = data.toString(includeNames=inc, maxWidth=mw, maxHeight=mh)
                    checkToStringRet(ret, data, inc, mw, mh)

    def test_toString_knownWidths(self):
        """ Test max string length reaches but does not exceed max width """
        raw = [['a', 'bbb', 'cc'], ['a', 'bbb', 'cc'], ['a', 'bbb', 'cc']]
        ftNames = ['fa', 'fb', 'fc']

        data = self.constructor(raw, featureNames=ftNames)
        # width of 5 to 7 will return first feature and colHold ('a  --')
        for mw in range(5, 8):
            ret = data.toString(maxWidth=mw, includeNames=True)
            # ignore last index since ret always ends with \n
            retSplit = ret.split('\n')[:-1]
            for i, line in enumerate(retSplit):
                if i == 0:
                    assert line == 'fa --'
                elif i == 1:
                    # separator for ftNames and data
                    assert line == ''
                else:
                    assert line == 'a  --'

        # width of 8 will return first ft, colHold, last ft ('a  -- cc')
        # 2 for first column based on feature name 'fa', 2 for colHold,
        # 2 for last column (both feature name and data have same width),
        # and 2 separating spaces
        ret = data.toString(maxWidth=8, includeNames=True)
        # ignore last index since ret always ends with \n
        retSplit = ret.split('\n')[:-1]
        for i, line in enumerate(retSplit):
            if i == 0:
                assert line == 'fa -- fc'
            elif i == 1:
                # separator for ftNames and data
                assert line == ''
            else:
                assert line == 'a  -- cc'

        # width of 9 can accommodate all data ('a  bbb cc')
        # 2 for first column based on feature name 'fa', 3 for second column
        # based on width of 'bbb', 2 for third column (both feature name and
        # data have same width) and 2 separating spaces
        ret = data.toString(maxWidth=9, includeNames=True)
        # ignore last index since ret always ends with \n
        retSplit = ret.split('\n')[:-1]
        for i, line in enumerate(retSplit):
            if i == 0:
                assert line == 'fa  fb fc'
            elif i == 1:
                # separator for ftNames and data
                assert line == ''
            else:
                assert line == 'a  bbb cc'

    def test_toString_knownHeights(self):
        """ Test max string height reaches but does not exceed max height """
        randGen = nimble.random.data("List", 9, 3, 0)

        data = self.constructor(randGen._data)

        # for height in range 3 to 8, row separator should be present
        for mh in range(3, 9):
            ret = data.toString(maxHeight=mh, includeNames=False)
            # ignore last index since ret always ends with \n
            retSplit = ret.split('\n')[:-1]
            sepRow = int(mh / 2)
            assert len(retSplit) == mh
            rowSepPattern = re.compile(r'[\s\|] +')
            assert re.match(rowSepPattern, retSplit[sepRow])
        # height 10 can accommodate all data
        ret = data.toString(maxHeight=10, includeNames=False)
        # ignore last index since ret always ends with \n
        retSplit = ret.split('\n')[:-1]
        for line in retSplit:
            assert '|' not in line


    def test_toString_emptyObjects(self):
        # no checks, but this at least confirms that it is runnable
        names = ['n1', 'n2', 'n3']

        rawPEmpty = np.zeros((0, 3))
        objPEmpty = self.constructor(rawPEmpty, featureNames=names)

        assert objPEmpty.toString() == ""

        rawFEmpty = np.zeros((3, 0))
        objFEmpty = self.constructor(rawFEmpty, pointNames=names)

        assert objFEmpty.toString() == ""


    # _arrangePointNames(self, maxRows, nameLength, rowHolder, nameHold)
    def test_arrangePointNames_correctSplit(self):
        rowHolder = '|'
        nameHold = '...'

        raw = [[300, 310, 320], [301, 311, 321], [302, 312, 312], [303, 313, 313], [303, 313, 313]]
        initnames = ['zero', 'one', 'two', 'three', 'four']
        obj = self.constructor(raw, pointNames=initnames)

        pnames, bound = obj._arrangePointNames(2, 11, rowHolder, nameHold)
        assert pnames == ['zero', rowHolder]
        assert bound == len('zero')

        pnames, bound = obj._arrangePointNames(3, 11, rowHolder, nameHold)
        assert pnames == ['zero', rowHolder, 'four']
        assert bound == len('four')

        pnames, bound = obj._arrangePointNames(4, 11, rowHolder, nameHold)
        assert pnames == ['zero', 'one', rowHolder, 'four']
        assert bound == len('four')

        pnames, bound = obj._arrangePointNames(5, 11, rowHolder, nameHold)
        assert pnames == ['zero', 'one', 'two', 'three', 'four']
        assert bound == len('three')


    def test_arrangePointNames_correctTruncation(self):
        rowHolder = '|'
        nameHold = '...'

        raw = [[300, 310, 320], [301, 311, 321], [302, 312, 312], [303, 313, 313]]
        initnames = ['zerooo', 'one', 'two', 'threee']
        obj = self.constructor(raw, pointNames=initnames)

        pnames, bound = obj._arrangePointNames(4, 3, rowHolder, nameHold)
        assert pnames == ['...', 'one', 'two', '...']
        assert bound == 3


    def test_arrangePointNames_omitDefault(self):
        rowHolder = '|'
        nameHold = '...'

        raw = [[300, 310, 320], [301, 311, 321], [302, 312, 312], [303, 313, 313]]
        initnames = [None, 'one', None, 'three']
        obj = self.constructor(raw, pointNames=initnames)

        pnames, bound = obj._arrangePointNames(4, 11, rowHolder, nameHold)
        assert pnames == ['', 'one', '', 'three']
        assert bound == len('three')

    @raises(InvalidArgumentValue)
    def test_arrangeDataWithLimits_exception_maxH(self):
        randGen = nimble.random.data("List", 5, 5, 0, elementType='int')
        randGen._arrangeDataWithLimits(maxHeight=1, maxWidth=120)

    @pytest.mark.slow
    def test_arrangeDataWithLimits(self):
        def makeUniformLength(rType, p, f, l):
            raw = []
            if l is not None:
                val = 10 ** (l - 1)
            else:
                val = 1
            for i in range(p):
                raw.append([])
                for j in range(f):
                    raw[i].append(val)

            return nimble.data(rType, raw)

        def runTrial(pNum, fNum, valLen, maxW, maxH, colSep, includeFNames):
            if pNum == 0 and fNum == 0:
                return
            elif pNum == 0:
                data = makeUniformLength("List", 1, fNum, valLen)
                data.points.extract(0)
                if includeFNames and fNum > 0:
                    fNames = ['ft' + str(i) for i in range(fNum)]
                    data.features.setNames(fNames)
            elif fNum == 0:
                data = makeUniformLength("List", pNum, 1, valLen)
                data.features.extract(0)
            else:
                if valLen is None:
                    data = nimble.random.data("List", pNum, fNum, .25, elementType='int')
                else:
                    data = makeUniformLength("List", pNum, fNum, valLen)
                if includeFNames:
                    fNames = ['ft' + str(i) for i in range(fNum)]
                    data.features.setNames(fNames)
                #			raw = data.data
            ret, widths, fNames = data._arrangeDataWithLimits(
                maxW, maxH, includeFNames=includeFNames, colSep=colSep)

            if includeFNames:
                assert len(fNames) == len(widths)
                for name, width in zip(fNames, widths):
                    assert len(name) <= width
                sepLen = len(colSep) * (len(fNames) - 1)
                assert sum(len(fn) for fn in fNames) <= maxW - sepLen

            assert len(ret) <= maxH

            for pRep in ret:
                assert len(pRep) == len(widths)
                lenSum = 0
                for val in pRep:
                    lenSum += len(val)
                assert lenSum <= (maxW - ((len(pRep) - 1) * len(colSep)))

            if len(ret) > 0:
                # the longest string value, which could be the feature name,
                # should be equal to the value in widths at that index
                for fIndex in range(len(ret[0])):
                    widthBound = len(fNames[fIndex]) if fNames else 0
                    for pRep in ret:
                        val = pRep[fIndex]
                        if len(val) > widthBound:
                            widthBound = len(val)
                    assert widths[fIndex] == widthBound

        for pNum in [0, 1, 2, 4, 5, 7, 10]:
            for fNum in [0, 1, 2, 4, 5, 7, 10]:
                for valLen in [1, 2, 4, 5, None]:
                    for maxW in [10, 20, 40, 80]:
                        for maxH in [2, 5, 10]:
                            for colSep in ['', ' ', '  ']:
                                for inc in [False, True]:
                                    runTrial(pNum, fNum, valLen, maxW, maxH, colSep, inc)

    ############
    # __repr__ #
    ############

    def back_reprOutput(self, numPts, numFts, truncated=False, defaults='none'):
        randGen = nimble.random.data("List", numPts, numFts, 0)
        if defaults in ['some', 'none']:
            pNames = ['pt' + str(i) for i in range(numPts)]
            fNames = ['ft' + str(i) for i in range(numFts)]
            if defaults == 'some':
                pNames = [name if i % 2 != 0 else None for i, name in enumerate(pNames)]
                fNames = [name if i % 2 != 0 else None for i, name in enumerate(fNames)]
            data = self.constructor(randGen._data, pointNames=pNames, featureNames=fNames)
        else:
            data = self.constructor(randGen._data)
        ret = repr(data)
        retSplit = ret.split('\n')

        # width check
        assert all(len(line) <= 79 for line in retSplit)

        # first line is type string
        assert retSplit[0] == data.getTypeString() + "("

        numPoints = min(30, len(data.points)) # truncated if over 30
        # data lines formatted correctly, truncated will still pass
        firstDataLinePattern = re.compile(r'\s{4}\[\[[0-9\.\s\-\|]+\]$')
        midDataLinesPattern = re.compile(r'\s{5}\[[0-9\.\s\-\|]+\]$')
        lastDataLinePattern = re.compile(r'\s{5}\[[0-9\.\s\-\|]+\]\]$')
        assert re.match(firstDataLinePattern, retSplit[1])
        for line in retSplit[2:numPoints]:
            assert re.match(midDataLinesPattern, line)
        assert re.match(lastDataLinePattern, retSplit[numPoints])

        for line in retSplit[1:numPoints]:
            if truncated:
                assert '--' in line
            else:
                assert '--' not in line

        if truncated:
            rowHoldPattern = re.compile(r'\s{5}\[[\s\|\-]+\]$')
            assert re.match(rowHoldPattern, retSplit[16])

        if defaults == 'all':
            assert len(retSplit) == numPoints + 2
        else:
            # pointNames
            if truncated:
                pNames = pNames[:15] + ['...'] + pNames[-14:]
            if defaults == 'some':
                pNamesString = "    pointNames=["
                toJoin = [name if name == '...' or name is None
                          else "'{}'".format(name) for name in pNames]
                pNamesString += ", ".join(map(str, toJoin))
                pNamesString += ']'
            else:
                pNamesString = "    pointNames={"
                toJoin = [name if name == '...'
                          else "'{}':{}".format(name, name[2:]) for name in pNames]
                pNamesString += ", ".join(toJoin)
                pNamesString += '}'
            wrappedPNames = textwrap.wrap(pNamesString, width=79, subsequent_indent=' '*8)
            fNameIndexStart = numPoints + len(wrappedPNames) + 1
            for line, wrapped in zip(retSplit[numPoints + 1:fNameIndexStart], wrappedPNames):
                # remove trailing whitespace which may differ between the two
                line = line.rstrip()
                wrapped = wrapped.rstrip()
                assert line == wrapped

            # featureNames
            if truncated:
                # get only the data between brackets
                firstDataLine = retSplit[1][6:-1]
                # could still have trailing whitespace
                firstDataLine = firstDataLine.rstrip()
                # split at column separator
                lCols, rCols = firstDataLine.split(' -- ')

                # split at whitespace to find the number of cols on left and right
                lColsLen = len(lCols.split())
                rColsLen = len(rCols.split())
                fNames = fNames[:lColsLen] + ['...'] + fNames[-rColsLen:]

            if defaults == 'some':
                fNamesString = "    featureNames=["
                toJoin = [name if name == '...' or name is None
                          else "'{}'".format(name) for name in fNames]
                fNamesString += ", ".join(map(str, toJoin))
                fNamesString += ']'
            else:
                fNamesString = "    featureNames={"
                toJoin = [name if name == '...'
                          else "'{}':{}".format(name, name[2:]) for name in fNames]
                fNamesString += ", ".join(toJoin)
                fNamesString += '}'
            wrappedFNames = textwrap.wrap(fNamesString, width=79, subsequent_indent=' '*8)
            fNameIndexEnd = fNameIndexStart + len(wrappedFNames)
            for line, wrapped in zip(retSplit[fNameIndexStart:fNameIndexEnd + 1], wrappedFNames):
                # remove trailing whitespace which may differ between the two
                line = line.rstrip()
                wrapped = wrapped.rstrip()
                assert line == wrapped

    def test_repr_notTruncated(self):
        self.back_reprOutput(9, 9)

    def test_repr_truncated(self):
        self.back_reprOutput(40, 20, truncated=True)

    def test_repr_notTruncated_withDefaults(self):
        self.back_reprOutput(9, 9, defaults='some')

    def test_repr_truncated_withDefaults(self):
        self.back_reprOutput(40, 20, truncated=True, defaults='some')

    def test_repr_notTruncated_allDefaults(self):
        self.back_reprOutput(9, 9, defaults='all')

    def test_repr_truncated_allDefaults(self):
        self.back_reprOutput(40, 20, truncated=True, defaults='all')

    def test_repr_lazyNameGeneration(self):
        data = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        repr(data)
        assertNoNamesGenerated(data)

        # test an empty object w/ more than 0 features and names reset to None
        empty = self.constructor([], featureNames=['a', 'b', 'c'])
        try:
            empty.features.setNames(None, useLog=False)
        except TypeError:
            # need to change names in views manually
            empty.features.names = None
            empty.features.namesInverse = None
        repr(data)
        assertNoNamesGenerated(empty)

    ###############################################
    # points.similarities / features.similarities #
    ###############################################

    # similarities calls the correlation and covariance in the calculate module
    # First we will test that similarities calls each calculate module function
    def test_points_similarities_callsCalculateFunction(self):
        simFuncs = {'correlation': 'correlation', 'covariance': 'covariance',
                    'sample covariance': 'covariance',
                    'population covariance' : 'covariance'}
        for simFunc in simFuncs:
            calcFunc = simFuncs[simFunc]
            self.backend_sim_callsFunctions(simFunc, calcFunc, 'point')

    def test_features_similarities_callsCalculateFunction(self):
        simFuncs = {'correlation': 'correlation', 'covariance': 'covariance',
                    'sample covariance': 'covariance',
                    'population covariance' : 'covariance'}
        for simFunc in simFuncs:
            calcFunc = simFuncs[simFunc]
            self.backend_sim_callsFunctions(simFunc, calcFunc, 'feature')

    def backend_sim_callsFunctions(self, objFunc, calcFunc, axis):
        with assertCalled(nimble.calculate, calcFunc):
            if axis == 'point':
                data = [[3, 0, 3], [0, 0, 3], [3, 0, 0]]
                obj = self.constructor(data)
                obj.points.similarities(objFunc)
            else:
                data = [[3, 0, 3], [0, 0, 0], [3, 3, 0]]
                obj = self.constructor(data)
                obj.features.similarities(objFunc)

    @raises(InvalidArgumentType)
    def test_points_similarities_InvalidParamType(self):
        """ Test points.similarities raise exception for unexpected param type """
        self.backend_Sim_InvalidParamType(True)

    @raises(InvalidArgumentType)
    def test_features_similarities_InvalidParamType(self):
        """ Test features.similarities raise exception for unexpected param type """
        self.backend_Sim_InvalidParamType(False)

    def backend_Sim_InvalidParamType(self, axis):
        data = [[1, 2], [3, 4]]
        obj = self.constructor(data)

        if axis:
            obj.points.similarities({"hello": 5})
        else:
            obj.features.similarities({"hello": 5})

    @raises(InvalidArgumentValue)
    def test_points_similarities_UnexpectedString(self):
        """ Test points.similarities raise exception for unexpected string value """
        self.backend_Sim_UnexpectedString(True)

    @raises(InvalidArgumentValue)
    def test_features_similarities_UnexpectedString(self):
        """ Test features.similarities raise exception for unexpected string value """
        self.backend_Sim_UnexpectedString(False)

    def backend_Sim_UnexpectedString(self, axis):
        data = [[1, 2], [3, 4]]
        obj = self.constructor(data)

        if axis:
            obj.points.similarities("foo")
        else:
            obj.features.similarities("foo")

    # test results covariance
    def test_points_similarities_SampleCovarianceResult(self):
        """ Test points.similarities returns correct sample covariance results """
        self.backend_Sim_SampleCovarianceResult(True)

    def test_features_similarities_SampleCovarianceResult(self):
        """ Test features.similarities returns correct sample covariance results """
        self.backend_Sim_SampleCovarianceResult(False)

    @noLogEntryExpected
    def backend_Sim_SampleCovarianceResult(self, axis):
        data = [[3, 0, 3], [0, 0, 3], [3, 0, 0]]
        dataT = np.array(data).T.tolist()
        orig = self.constructor(data)
        trans = self.constructor(dataT)
        sameAsOrig = self.constructor(data)
        sameAsOrigT = self.constructor(dataT)

        if axis:
            ret = orig.points.similarities("covariance ")
        else:
            ret = trans.features.similarities("sample\tcovariance")
            ret.transpose(useLog=False)

        # hand computed results
        expRow0 = [3, 1.5, 1.5]
        expRow1 = [1.5, 3, -1.5]
        expRow2 = [1.5, -1.5, 3]
        expData = [expRow0, expRow1, expRow2]
        expObj = self.constructor(expData)

        # numpy computted result -- bias=0 -> divisor of n-1
        npExpRaw = np.cov(data, bias=0)
        npExpObj = self.constructor(npExpRaw)
        assert ret.isApproximatelyEqual(npExpObj)

        assert expObj.isApproximatelyEqual(ret)
        assert sameAsOrig == orig
        assert sameAsOrigT == trans

    def test_points_similarities_PopulationCovarianceResult(self):
        """ Test points.similarities returns correct population covariance results """
        self.backend_Sim_populationCovarianceResult(True)

    def test_features_similarities_PopulationCovarianceResult(self):
        """ Test features.similarities returns correct population covariance results """
        self.backend_Sim_populationCovarianceResult(False)

    @noLogEntryExpected
    def backend_Sim_populationCovarianceResult(self, axis):
        data = [[3, 0, 3], [0, 0, 3], [3, 0, 0]]
        dataT = np.array(data).T.tolist()
        orig = self.constructor(data)
        trans = self.constructor(dataT)
        sameAsOrig = self.constructor(data)
        sameAsOrigT = self.constructor(dataT)

        if axis:
            ret = orig.points.similarities("population COvariance")
        else:
            ret = trans.features.similarities("populationcovariance")
            ret.transpose(useLog=False)

        # hand computed results
        expRow0 = [2, 1, 1]
        expRow1 = [1, 2, -1]
        expRow2 = [1, -1, 2]
        expData = [expRow0, expRow1, expRow2]
        expObj = self.constructor(expData)

        # numpy computted result -- bias=1 -> divisor of n
        npExpRaw = np.cov(data, bias=1)
        npExpObj = self.constructor(npExpRaw)
        assert ret.isApproximatelyEqual(npExpObj)

        assert expObj.isApproximatelyEqual(ret)
        assert sameAsOrig == orig
        assert sameAsOrigT == trans

    def test_points_similarities_STDandVarianceIdentity(self):
        """ Test identity between population covariance and population std of points """
        self.backend_Sim_STDandVarianceIdentity(True)

    def test_features_similarities_STDandVarianceIdentity(self):
        """ Test identity between population covariance and population std of features """
        self.backend_Sim_STDandVarianceIdentity(False)

    @noLogEntryExpected
    def backend_Sim_STDandVarianceIdentity(self, axis):
        data = [[3, 0, 3], [0, 0, 3], [3, 0, 0]]
        dataT = np.array(data).T.tolist()
        orig = self.constructor(data)
        trans = self.constructor(dataT)

        if axis:
            ret = orig.points.similarities(" populationcovariance")
            stdVector = orig.points.statistics("population std")
        else:
            ret = trans.features.similarities("populationcovariance")
            stdVector = trans.features.statistics("\npopulationstd")
            ret.transpose(useLog=False)

        np.testing.assert_approx_equal(ret[0, 0], stdVector[0] * stdVector[0])
        np.testing.assert_approx_equal(ret[1, 1], stdVector[1] * stdVector[1])
        np.testing.assert_approx_equal(ret[2, 2], stdVector[2] * stdVector[2])


    # test results correlation
    def test_points_similarities_CorrelationResult(self):
        """ Test points.similarities returns correct correlation results """
        self.backend_Sim_CorrelationResult(True)

    def test_features_similarities_CorrelationResult(self):
        """ Test features.similarities returns correct correlation results """
        self.backend_Sim_CorrelationResult(False)

    @noLogEntryExpected
    def backend_Sim_CorrelationResult(self, axis):
        data = [[3, 0, 3], [0, 0, 3], [3, 0, 0]]
        dataT = np.array(data).T.tolist()
        orig = self.constructor(data)
        trans = self.constructor(dataT)
        sameAsOrig = self.constructor(data)
        sameAsOrigT = self.constructor(dataT)

        if axis:
            ret = orig.points.similarities("correlation")
        else:
            ret = trans.features.similarities("corre lation")
            ret.transpose(useLog=False)

        expRow0 = [1, (1. / 2), (1. / 2)]
        expRow1 = [(1. / 2), 1, (-1. / 2)]
        expRow2 = [(1. / 2), (-1. / 2), 1]
        expData = [expRow0, expRow1, expRow2]
        expObj = self.constructor(expData)

        npExpRaw = np.corrcoef(data)
        npExpObj = self.constructor(npExpRaw)

        assert ret.isApproximatelyEqual(npExpObj)
        assert expObj.isApproximatelyEqual(ret)
        assert sameAsOrig == orig
        assert sameAsOrigT == trans

    def test_points_similarities_CorrelationHelpersEquiv(self):
        """ Compare points.similarities correlation using the various possible helpers """
        self.backend_Sim_CorrelationHelpersEquiv(True)

    def test_features_similarities_CorrelationHelpersEquiv(self):
        """ Compare features.similarities correlation using the various possible helpers """
        self.backend_Sim_CorrelationHelpersEquiv(False)

    def backend_Sim_CorrelationHelpersEquiv(self, axis):
        data = [[3, 0, 3], [0, 0, 3], [3, 0, 0]]
        dataT = np.array(data).T.tolist()
        orig = self.constructor(data)
        trans = self.constructor(dataT)
        sameAsOrig = self.constructor(data)

        def explicitCorr(X, sample=True):
            sampleStdVector = X.points.statistics('samplestd')
            popStdVector = X.points.statistics('populationstd')
            stdVector = sampleStdVector if sample else popStdVector

            stdVector_T = stdVector.copy()
            stdVector_T.transpose()

            if sample:
                cov = X.points.similarities('sample covariance')
            else:
                cov = X.points.similarities('population Covariance')

            stdMatrix = stdVector.matrixMultiply(stdVector_T)
            ret = cov / stdMatrix

            return ret

        if axis:
            ret = orig.points.similarities("correlation")
            sampRet = explicitCorr(orig, True)
            popRet = explicitCorr(orig, False)
        else:
            ret = trans.features.similarities("correlation")
            # helper only calls pointStatistics, so have to make sure
            # that in this case, we are calling with the transpose of
            # the object used to test features.similarities
            sampRet = explicitCorr(orig, True)
            popRet = explicitCorr(orig, False)
            ret.transpose()

        npExpRawB0 = np.corrcoef(data, bias=0)
        npExpRawB1 = np.corrcoef(data, bias=1)
        npExpB0 = self.constructor(npExpRawB0)
        npExpB1 = self.constructor(npExpRawB1)

        assert ret.isApproximatelyEqual(sampRet)
        assert ret.isApproximatelyEqual(popRet)
        assert sampRet.isApproximatelyEqual(popRet)
        assert ret.isApproximatelyEqual(npExpB0)
        assert ret.isApproximatelyEqual(npExpB1)


    # test results dot product
    def test_points_similarities_DotProductResult(self):
        """ Test points.similarities returns correct dot product results """
        self.backend_Sim_DotProductResult(True)

    def test_features_similarities_DotProductResult(self):
        """ Test features.similarities returns correct dot product results """
        self.backend_Sim_DotProductResult(False)

    @noLogEntryExpected
    def backend_Sim_DotProductResult(self, axis):
        data = [[1, 1, 1], [0, 1, 1], [1, 0, 0]]
        dataT = np.array(data).T.tolist()
        orig = self.constructor(data)
        trans = self.constructor(dataT)
        sameAsOrig = self.constructor(data)
        sameAsOrigT = self.constructor(dataT)

        if axis:
            ret = orig.points.similarities("Dot Product")
        else:
            ret = trans.features.similarities("dotproduct\n")
            ret.transpose(useLog=False)

        expData = [[3, 2, 1], [2, 2, 0], [1, 0, 1]]
        expObj = self.constructor(expData)

        assert expObj == ret
        assert sameAsOrig == orig
        assert sameAsOrigT == trans

    # test input function validation
    @raises(InvalidArgumentValue)
    def todotest_points_similarities_FuncValidation(self):
        """ Test points.similarities raises exception for invalid functions """
        self.backend_Sim_FuncValidation(True)

    @raises(InvalidArgumentValue)
    def todotest_features_similarities_FuncValidation(self):
        """ Test features.similarities raises exception for invalid functions """
        self.backend_Sim_FuncValidation(False)

    def backend_Sim_FuncValidation(self, axis):
        assert False
        data = [[1, 2], [3, 4]]
        obj = self.constructor(data)

        def singleArg(one):
            return one

        if axis:
            obj.points.similarities(singleArg)
        else:
            obj.features.similarities(singleArg)

    # test results passed function
    def todotest_pointSimilariteGivenFuncResults(self):
        """ Test points.similarities returns correct results for given function """
        self.backend_Sim_GivenFuncResults(True)

    def todotest_features_similarities_GivenFuncResults(self):
        """ Test features.similarities returns correct results for given function """
        self.backend_Sim_GivenFuncResults(False)

    def backend_Sim_GivenFuncResults(self, axis):
        assert False
        data = [[1, 2], [3, 4]]
        obj = self.constructor(data)

        def euclideanDistance(left, right):
            assert False

        if axis:
            obj.points.similarities(euclideanDistance)
        else:
            obj.features.similarities(euclideanDistance)

    def backend_Sim_NamePath_Preservation(self, axis):
        data = [[3, 0, 3], [0, 0, 3], [3, 0, 0]]
        dataT = np.array(data).T.tolist()
        orig = self.constructor(data, name=preserveName, paths=preservePair)
        trans = self.constructor(dataT, name=preserveName, paths=preservePair)

        possible = [
            'correlation', 'covariance', 'dotproduct', 'samplecovariance',
            'populationcovariance'
        ]

        for curr in possible:
            if axis:
                ret = orig.points.similarities(curr)
            else:
                ret = trans.features.similarities(curr)

            assert orig.name == preserveName
            assert orig.absolutePath == preserveAPath
            assert orig.relativePath == preserveRPath

            assert ret.name is None
            assert ret.absolutePath == preserveAPath
            assert ret.relativePath == preserveRPath

    def test_points_similarities_Dot_NamePath_preservation(self):
        self.backend_Sim_NamePath_Preservation(True)

    def test_features_similarities__NamePath_preservation(self):
        self.backend_Sim_NamePath_Preservation(False)


    ################### ####################
    # pointStatistics # #featureStatistics #
    ################### ####################

    # # statistics calls several functions in the calculate module
    # # First we will test that statistics calls each calculate module function
    def test_points_statistics_callsCalculateFunction(self):
        statFuncs = {'max': 'maximum', 'mean': 'mean', 'median': 'median',
                     'min': 'minimum', 'population std': 'standardDeviation',
                     'population standard deviation': 'standardDeviation',
                     'proportion missing': 'proportionMissing',
                     'proportion zero': 'proportionZero',
                     'sample standard deviation': 'standardDeviation',
                     'sample std': 'standardDeviation',
                     'standard deviation': 'standardDeviation',
                     'std': 'standardDeviation', 'unique count': 'uniqueCount'}
        for statFunc in statFuncs:
            calcFunc = statFuncs[statFunc]
            self.backend_stat_callsFunctions(statFunc, calcFunc, 'point')

    def test_features_statistics_callsCalculateFunction(self):
        statFuncs = {'max': 'maximum', 'mean': 'mean', 'median': 'median',
                     'min': 'minimum', 'population std': 'standardDeviation',
                     'population standard deviation': 'standardDeviation',
                     'proportion missing': 'proportionMissing',
                     'proportion zero': 'proportionZero',
                     'sample standard deviation': 'standardDeviation',
                     'sample std': 'standardDeviation',
                     'standard deviation': 'standardDeviation',
                     'std': 'standardDeviation', 'unique count': 'uniqueCount'}
        for statFunc in statFuncs:
            calcFunc = statFuncs[statFunc]
            self.backend_stat_callsFunctions(statFunc, calcFunc, 'feature')

    def backend_stat_callsFunctions(self, objFunc, calcFunc, axis):
        with assertCalled(nimble.calculate, calcFunc):
            if axis == 'point':
                data = [[3, 0, 3], [0, 0, 3], [3, 0, 0]]
                obj = self.constructor(data)
                obj.points.statistics(objFunc)
            else:
                data = [[3, 0, 3], [0, 0, 0], [3, 3, 0]]
                obj = self.constructor(data)
                obj.features.statistics(objFunc)

    def test_pointStatistics_max(self):
        """ Test pointStatistics returns correct max results """
        self.backend_Stat_max(True)

    def test_featureStatistics_max(self):
        """ Test featureStatistics returns correct max results """
        self.backend_Stat_max(False)

    @noLogEntryExpected
    def backend_Stat_max(self, axis):
        data = [[1, 2, 1], [-10, -1, -21], [-1, 0, 0]]
        dataT = np.array(data).T.tolist()
        fnames = _fnames(3)
        pnames = _pnames(3)
        orig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        trans = self.constructor(dataT, featureNames=pnames, pointNames=fnames)
        sameAsOrig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        sameAsOrigT = self.constructor(dataT, featureNames=pnames, pointNames=fnames)

        if axis:
            ret = orig.points.statistics("MAx")

        else:
            ret = trans.features.statistics("max ")
            ret.transpose(useLog=False)

        assert len(ret.points) == 3
        assert len(ret.features) == 1

        expRaw = [[2], [-1], [0]]
        expObj = self.constructor(expRaw, featureNames=["max"], pointNames=pnames)

        assert expObj == ret
        assert sameAsOrig == orig
        assert sameAsOrigT == trans

    def test_pointStatistics_mean(self):
        """ Test pointStatistics returns correct mean results """
        self.backend_Stat_mean(True)

    def test_featureStatistics_mean(self):
        """ Test featureStatistics returns correct mean results """
        self.backend_Stat_mean(False)

    @noLogEntryExpected
    def test_featureStatistics_groupbyfeature(self):
        orig = self.constructor([[1,2,3,'f'], [4,5,6,'m'], [7,8,9,'f'], [10,11,12,'m']], featureNames=['a','b', 'c', 'gender'])
        if isinstance(orig, nimble.core.data.BaseView):
            return
        #don't test view.
        res = orig.features.statistics('mean', groupByFeature='gender')
        expObjf = self.constructor([4,5,6], featureNames=['a','b', 'c'], pointNames=['mean'])
        expObjm = self.constructor([7,8,9], featureNames=['a','b', 'c'], pointNames=['mean'])
        assert expObjf == res['f']
        assert expObjm == res['m']

    @noLogEntryExpected
    def backend_Stat_mean(self, axis):
        data = [[1, 1, 1], [0, 1, 1], [1, 0, 0]]
        dataT = np.array(data).T.tolist()
        fnames = _fnames(3)
        pnames = _pnames(3)
        orig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        trans = self.constructor(dataT, featureNames=pnames, pointNames=fnames)
        sameAsOrig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        sameAsOrigT = self.constructor(dataT, featureNames=pnames, pointNames=fnames)

        if axis:
            ret = orig.points.statistics("Mean")
        else:
            ret = trans.features.statistics(" MEAN")
            ret.transpose(useLog=False)

        assert len(ret.points) == 3
        assert len(ret.features) == 1

        expRaw = [[1], [2. / 3], [1. / 3]]
        expObj = self.constructor(expRaw, featureNames=["mean"], pointNames=pnames)

        assert expObj == ret
        assert sameAsOrig == orig
        assert sameAsOrigT == trans

    def test_pointStatistics_median(self):
        """ Test pointStatistics returns correct median results """
        self.backend_Stat_median(True)

    def test_featureStatistics_median(self):
        """ Test featureStatistics returns correct median results """
        self.backend_Stat_median(False)

    @noLogEntryExpected
    def backend_Stat_median(self, axis):
        data = [[1, 1, 1], [0, 1, 1], [1, 0, 0]]
        dataT = np.array(data).T.tolist()
        fnames = _fnames(3)
        pnames = _pnames(3)
        orig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        trans = self.constructor(dataT, featureNames=pnames, pointNames=fnames)
        sameAsOrig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        sameAsOrigT = self.constructor(dataT, featureNames=pnames, pointNames=fnames)

        if axis:
            ret = orig.points.statistics("MeDian")
        else:
            ret = trans.features.statistics("median")
            ret.transpose(useLog=False)

        assert len(ret.points) == 3
        assert len(ret.features) == 1

        expRaw = [[1], [1], [0]]
        expObj = self.constructor(expRaw, featureNames=["median"], pointNames=pnames)

        assert expObj == ret
        assert sameAsOrig == orig
        assert sameAsOrigT == trans


    def test_pointStatistics_min(self):
        """ Test pointStatistics returns correct min results """
        self.backend_Stat_min(True)

    def test_featureStatistics_min(self):
        """ Test featureStatistics returns correct min results """
        self.backend_Stat_min(False)

    @noLogEntryExpected
    def backend_Stat_min(self, axis):
        data = [[1, 2, 1], [-10, -1, -21], [-1, 0, 0]]
        dataT = np.array(data).T.tolist()
        fnames = _fnames(3)
        pnames = _pnames(3)
        orig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        trans = self.constructor(dataT, featureNames=pnames, pointNames=fnames)
        sameAsOrig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        sameAsOrigT = self.constructor(dataT, featureNames=pnames, pointNames=fnames)

        if axis:
            ret = orig.points.statistics("mIN")
        else:
            ret = trans.features.statistics("min")
            ret.transpose(useLog=False)

        assert len(ret.points) == 3
        assert len(ret.features) == 1

        expRaw = [[1], [-21], [-1]]
        expObj = self.constructor(expRaw, featureNames=['min'], pointNames=pnames)

        assert expObj == ret
        assert sameAsOrig == orig
        assert sameAsOrigT == trans

    def test_pointStatistics_uniqueCount(self):
        """ Test pointStatistics returns correct uniqueCount results """
        self.backend_Stat_uniqueCount(True)

    def test_featureStatistics_uniqueCount(self):
        """ Test featureStatistics returns correct uniqueCount results """
        self.backend_Stat_uniqueCount(False)

    @noLogEntryExpected
    def backend_Stat_uniqueCount(self, axis):
        data = [[1, 1, 1], [0, 1, 1], [1, 0, -1]]
        dataT = np.array(data).T.tolist()
        fnames = _fnames(3)
        pnames = _pnames(3)
        orig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        trans = self.constructor(dataT, featureNames=pnames, pointNames=fnames)
        sameAsOrig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        sameAsOrigT = self.constructor(dataT, featureNames=pnames, pointNames=fnames)

        if axis:
            ret = orig.points.statistics("unique Count")
        else:
            ret = trans.features.statistics("UniqueCount")
            ret.transpose(useLog=False)

        assert len(ret.points) == 3
        assert len(ret.features) == 1

        expRaw = [[1], [2], [3]]
        expObj = self.constructor(expRaw, featureNames=['uniquecount'], pointNames=pnames)

        assert expObj == ret
        assert sameAsOrig == orig
        assert sameAsOrigT == trans

    def todotest_pointStatistics_proportionMissing(self):
        """ Test pointStatistics returns correct proportionMissing results """
        self.backend_Stat_proportionMissing(True)

    def todotest_featureStatistics_proportionMissing(self):
        """ Test featureStatistics returns correct proportionMissing results """
        self.backend_Stat_proportionMissing(False)

    @noLogEntryExpected
    def backend_Stat_proportionMissing(self, axis):
        data = [[1, None, 1], [0, 1, float('nan')], [1, float('nan'), None]]
        dataT = np.array(data).T.tolist()
        fnames = _fnames(3)
        pnames = _pnames(3)
        orig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        trans = self.constructor(dataT, featureNames=pnames, pointNames=fnames)
        sameAsOrig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        sameAsOrigT = self.constructor(dataT, featureNames=pnames, pointNames=fnames)

        if axis:
            ret = orig.points.statistics("Proportion Missing ")
        else:
            ret = trans.features.statistics("proportionmissing")
            ret.transpose(useLog=False)

        assert len(ret.points) == 3
        assert len(ret.features) == 1

        expRaw = [[1. / 3], [1. / 3], [2. / 3]]
        expObj = self.constructor(expRaw, featureNames=['proportionmissing'], pointNames=pnames)

        assert expObj == ret
        assert sameAsOrig == orig
        assert sameAsOrigT == trans

    def test_pointStatistics_proportionZero(self):
        """ Test pointStatistics returns correct proportionZero results """
        self.backend_Stat_proportionZero(True)

    def test_featureStatistics_proportionZero(self):
        """ Test featureStatistics returns correct proportionZero results """
        self.backend_Stat_proportionZero(False)

    @noLogEntryExpected
    def backend_Stat_proportionZero(self, axis):
        data = [[1, 1, 1], [0, 1, 1], [1, 0, 0]]
        dataT = np.array(data).T.tolist()
        fnames = _fnames(3)
        pnames = _pnames(3)
        orig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        trans = self.constructor(dataT, featureNames=pnames, pointNames=fnames)
        sameAsOrig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        sameAsOrigT = self.constructor(dataT, featureNames=pnames, pointNames=fnames)

        if axis:
            ret = orig.points.statistics("proportionZero")
        else:
            ret = trans.features.statistics("proportion Zero")
            assert len(ret.points) == 1
            assert len(ret.features) == 3
            ret.transpose(useLog=False)

        assert len(ret.points) == 3
        assert len(ret.features) == 1

        expRaw = [[0], [1. / 3], [2. / 3]]
        expObj = self.constructor(expRaw, featureNames=['proportionzero'], pointNames=pnames)

        assert expObj == ret
        assert sameAsOrig == orig
        assert sameAsOrigT == trans

    def test_pointStatistics_samplestd(self):
        """ Test pointStatistics returns correct sample std results """
        self.backend_Stat_sampleStandardDeviation(True)

    def test_featureStatistics_samplestd(self):
        """ Test featureStatistics returns correct sample std results """
        self.backend_Stat_sampleStandardDeviation(False)

    @noLogEntryExpected
    def backend_Stat_sampleStandardDeviation(self, axis):
        data = [[1, 1, 1], [0, 1, 1], [1, 0, 0]]
        dataT = np.array(data).T.tolist()
        fnames = _fnames(3)
        pnames = _pnames(3)
        orig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        trans = self.constructor(dataT, featureNames=pnames, pointNames=fnames)
        sameAsOrig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        sameAsOrigT = self.constructor(dataT, featureNames=pnames, pointNames=fnames)

        if axis:
            ret = orig.points.statistics("samplestd  ")
        else:
            ret = trans.features.statistics("standard deviation")
            ret.transpose(useLog=False)

        assert len(ret.points) == 3
        assert len(ret.features) == 1

        npExpRaw = np.std(data, axis=1, ddof=1, keepdims=True)
        npExpObj = self.constructor(npExpRaw)

        assert npExpObj.isApproximatelyEqual(ret)

        expRaw = [[0], [math.sqrt(3. / 9)], [math.sqrt(3. / 9)]]
        expObj = self.constructor(expRaw, pointNames=pnames)
        expPossibleFNames = ['samplestd', 'samplestandarddeviation', 'std', 'standarddeviation']

        assert expObj.isApproximatelyEqual(ret)
        assert expObj.points.getNames() == ret.points.getNames()
        assert len(ret.features.getNames()) == 1
        assert ret.features.getNames()[0] in expPossibleFNames
        assert sameAsOrig == orig
        assert sameAsOrigT == trans


    def test_pointStatistics_populationstd(self):
        """ Test pointStatistics returns correct population std results """
        self.backend_Stat_populationStandardDeviation(True)

    def test_featureStatistics_populationstd(self):
        """ Test featureStatistics returns correct population std results """
        self.backend_Stat_populationStandardDeviation(False)

    @noLogEntryExpected
    def backend_Stat_populationStandardDeviation(self, axis):
        data = [[1, 1, 1], [0, 1, 1], [1, 0, 0]]
        dataT = np.array(data).T.tolist()
        fnames = _fnames(3)
        pnames = _pnames(3)
        orig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        trans = self.constructor(dataT, featureNames=pnames, pointNames=fnames)
        sameAsOrig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        sameAsOrigT = self.constructor(dataT, featureNames=pnames, pointNames=fnames)

        if axis:
            ret = orig.points.statistics("popu  lationstd")
        else:
            ret = trans.features.statistics("population standarddeviation")
            ret.transpose(useLog=False)

        assert len(ret.points) == 3
        assert len(ret.features) == 1

        npExpRaw = np.std(data, axis=1, ddof=0, keepdims=True)
        npExpObj = self.constructor(npExpRaw)

        assert npExpObj.isApproximatelyEqual(ret)

        expRaw = [[0], [math.sqrt(2. / 9)], [math.sqrt(2. / 9)]]
        expObj = self.constructor(expRaw, pointNames=pnames)
        expPossiblePNames = ['populationstd', 'populationstandarddeviation']

        assert expObj.isApproximatelyEqual(ret)
        assert expObj.points.getNames() == ret.points.getNames()
        assert len(ret.features.getNames()) == 1
        assert ret.features.getNames()[0] in expPossiblePNames
        assert sameAsOrig == orig
        assert sameAsOrigT == trans

    @raises(InvalidArgumentValue)
    def test_pointStatistics_unexpectedString(self):
        """ Test pointStatistics returns correct std results """
        self.backend_Stat_unexpectedString(True)

    @raises(InvalidArgumentValue)
    def test_featureStatistics_unexpectedString(self):
        """ Test featureStatistics returns correct std results """
        self.backend_Stat_unexpectedString(False)

    def backend_Stat_unexpectedString(self, axis):
        data = [[1, 1, 1], [0, 1, 1], [1, 0, 0]]
        orig = self.constructor(data)
        sameAsOrig = self.constructor(data)

        if axis:
            ret = orig.points.statistics("hello")
        else:
            ret = orig.features.statistics("meanie")

    def backend_Stat_NamePath_preservation(self, axis):
        data = [[1, 2, 1], [-10, -1, -21], [-1, 0, 0]]
        orig = self.constructor(data, name=preserveName, paths=preservePair)

        accepted = [
            'max', 'mean', 'median', 'min', 'uniquecount', 'proportionmissing',
            'proportionzero', 'standarddeviation', 'std', 'populationstd',
            'populationstandarddeviation', 'samplestd',
            'samplestandarddeviation'
        ]

        for curr in accepted:
            if axis:
                ret = orig.points.statistics(curr)
            else:
                ret = orig.features.statistics(curr)

            assert orig.name == preserveName
            assert orig.absolutePath == preserveAPath
            assert orig.relativePath == preserveRPath

            assert ret.name is None
            assert ret.absolutePath == preserveAPath
            assert ret.relativePath == preserveRPath

    def test_pointStatistics_NamePath_preservations(self):
        self.backend_Stat_NamePath_preservation(True)

    def test_featureStatistics_NamePath_preservations(self):
        self.backend_Stat_NamePath_preservation(False)


    ########
    # plot #
    ########

    @pytest.mark.slow
    @noLogEntryExpected
    def test_plot_fileOutput(self):
        with tempfile.NamedTemporaryFile(suffix='.png') as outFile:
            path = outFile.name
            startSize = os.path.getsize(path)
            assert startSize == 0

            randGenerated = nimble.random.data("List", 10, 10, 0, useLog=False)
            raw = randGenerated.copy(to='pythonlist')
            obj = self.constructor(raw)
            obj.plotHeatMap(outPath=path, show=False)

            endSize = os.path.getsize(path)
            assert startSize < endSize
            assertNoNamesGenerated(obj)

    ###########################
    # plotFeatureDistribution #
    ###########################

    @pytest.mark.slow
    @noLogEntryExpected
    def test_plotFeatureDistribution_fileOutput(self):
        with tempfile.NamedTemporaryFile(suffix='.png') as outFile:
            path = outFile.name
            startSize = os.path.getsize(path)
            assert startSize == 0

            randGenerated = nimble.random.data("List", 10, 10, 0, useLog=False)
            raw = randGenerated.copy(to='pythonlist')
            obj = self.constructor(raw)
            obj.plotFeatureDistribution(feature=0, outPath=path, show=False)

            endSize = os.path.getsize(path)
            assert startSize < endSize
            assertNoNamesGenerated(obj)


    #############################
    # plotFeatureAgainstFeature #
    #############################

    @pytest.mark.slow
    @noLogEntryExpected
    def test_plotFeatureAgainstFeature_fileOutput(self):
        with tempfile.NamedTemporaryFile(suffix='.png') as outFile:
            path = outFile.name
            startSize = os.path.getsize(path)
            assert startSize == 0

            randGenerated = nimble.random.data("List", 10, 10, 0, useLog=False)
            raw = randGenerated.copy(to='pythonlist')
            obj = self.constructor(raw)
            obj.plotFeatureAgainstFeature(x=0, y=1, outPath=path, show=False)

            endSize = os.path.getsize(path)
            assert startSize < endSize
            assertNoNamesGenerated(obj)

    ##################
    # features.plot #
    #################

    @pytest.mark.slow
    @noLogEntryExpected
    def test_features_plot_fileOutput(self):
        with tempfile.NamedTemporaryFile(suffix='.png') as outFile:
            path = outFile.name
            startSize = os.path.getsize(path)
            assert startSize == 0

            obj = nimble.random.data("List", 3, 3, 0, useLog=False)
            obj.features.plot(outPath=path, show=False)

            endSize = os.path.getsize(path)
            assert startSize < endSize
            assertNoNamesGenerated(obj)

    ######################
    # features.plotMeans #
    ######################

    @pytest.mark.slow
    @noLogEntryExpected
    def test_features_plotMeans_fileOutput(self):
        with tempfile.NamedTemporaryFile(suffix='.png') as outFile:
            path = outFile.name
            startSize = os.path.getsize(path)
            assert startSize == 0

            obj = nimble.random.data("List", 10, 10, 0, useLog=False)
            obj.features.plotMeans(outPath=path, show=False)

            endSize = os.path.getsize(path)
            assert startSize < endSize
            assertNoNamesGenerated(obj)

    ###########################
    # features.plotStatistics #
    ###########################

    @pytest.mark.slow
    @noLogEntryExpected
    def test_features_plotStatistics_fileOutput(self):
        with tempfile.NamedTemporaryFile(suffix='.png') as outFile:
            path = outFile.name
            startSize = os.path.getsize(path)
            assert startSize == 0

            obj = nimble.random.data("List", 10, 10, 0, useLog=False)
            obj.features.plotStatistics(sum, outPath=path, show=False)

            endSize = os.path.getsize(path)
            assert startSize < endSize
            assertNoNamesGenerated(obj)

    #########################
    # plotFeatureGroupMeans #
    #########################

    @pytest.mark.slow
    @noLogEntryExpected
    def test_plotFeatureGroupMeans_fileOutput(self):
        with tempfile.NamedTemporaryFile(suffix='.png') as outFile:
            path = outFile.name
            startSize = os.path.getsize(path)
            assert startSize == 0

            groups = [['a'], ['a'], ['a'], ['b'], ['b'], ['b'],
                      ['c'], ['c'], ['c'], ['d']]
            groupObj = nimble.data('List', groups, useLog=False)
            obj = nimble.random.data("List", 10, 1, 0, useLog=False)
            obj.features.append(groupObj, useLog=False)

            obj.plotFeatureGroupMeans(0, 1, outPath=path, show=False)

            endSize = os.path.getsize(path)
            assert startSize < endSize
            assertNoNamesGenerated(obj)

    ##############################
    # plotFeatureGroupStatistics #
    ##############################

    @pytest.mark.slow
    @noLogEntryExpected
    def test_plotGroupStatistics_fileOutput(self):
        with tempfile.NamedTemporaryFile(suffix='.png') as outFile:
            path = outFile.name
            startSize = os.path.getsize(path)
            assert startSize == 0

            groups = [['a'], ['a'], ['a'], ['b'], ['b'], ['b'],
                      ['c'], ['c'], ['c'], ['d']]
            groupObj = nimble.data('List', groups, useLog=False)
            obj = nimble.random.data("List", 10, 1, 0, useLog=False)
            obj.features.append(groupObj, useLog=False)

            obj.plotFeatureGroupStatistics(sum, 0, 1, outPath=path, show=False)

            endSize = os.path.getsize(path)
            assert startSize < endSize
            assertNoNamesGenerated(obj)

    ################
    # points.plot #
    ###############

    @pytest.mark.slow
    @noLogEntryExpected
    def test_points_plot_fileOutput(self):
        with tempfile.NamedTemporaryFile(suffix='.png') as outFile:
            path = outFile.name
            startSize = os.path.getsize(path)
            assert startSize == 0

            obj = nimble.random.data("List", 3, 3, 0, useLog=False)
            obj.points.plot(outPath=path, show=False)

            endSize = os.path.getsize(path)
            assert startSize < endSize
            assertNoNamesGenerated(obj)

    ####################
    # points.plotMeans #
    ####################

    @pytest.mark.slow
    @noLogEntryExpected
    def test_points_plotMeans_fileOutput(self):
        with tempfile.NamedTemporaryFile(suffix='.png') as outFile:
            path = outFile.name
            startSize = os.path.getsize(path)
            assert startSize == 0

            obj = nimble.random.data("List", 10, 10, 0, useLog=False)
            obj.points.plotMeans(outPath=path, show=False)

            endSize = os.path.getsize(path)
            assert startSize < endSize
            assertNoNamesGenerated(obj)

    #########################
    # points.plotStatistics #
    #########################

    @pytest.mark.slow
    @noLogEntryExpected
    def test_points_plotStatistics_fileOutput(self):
        with tempfile.NamedTemporaryFile(suffix='.png') as outFile:
            path = outFile.name
            startSize = os.path.getsize(path)
            assert startSize == 0

            obj = nimble.random.data("List", 10, 10, 0, useLog=False)
            obj.points.plotStatistics(sum, outPath=path, show=False)

            endSize = os.path.getsize(path)
            assert startSize < endSize
            assertNoNamesGenerated(obj)

    ###################
    # points.__iter__ #
    ###################
    @noLogEntryExpected
    def test_points_iter_FemptyCorrectness(self):
        data = [[], []]
        data = np.array(data)
        toTest = self.constructor(data)
        pIter = iter(toTest.points)

        pView = next(pIter)
        assert len(pView) == 0
        pView = next(pIter)
        assert len(pView) == 0

        with raises(StopIteration):
            next(pIter)

    def test_points_iter_noNextPempty(self):
        """ test .points() has no next value when object is point empty """
        data = [[], []]
        data = np.array(data).T
        toTest = self.constructor(data)
        viewIter = iter(toTest.points)
        with raises(StopIteration):
            next(viewIter)

    @noLogEntryExpected
    def test_points_iter_exactValueViaFor(self):
        """ Test .points() gives views that contain exactly the correct data """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)

        viewIter = iter(toTest.points)

        toCheck = []
        for v in viewIter:
            toCheck.append(v)

        assert toCheck[0][0] == 1
        assert toCheck[0][1] == 2
        assert toCheck[0][2] == 3
        assert toCheck[1][0] == 4
        assert toCheck[1][1] == 5
        assert toCheck[1][2] == 6
        assert toCheck[2][0] == 7
        assert toCheck[2][1] == 8
        assert toCheck[2][2] == 9

    def test_points_iter_allZeroVectors(self):
        """ Test .points() works when there are all zero points """
        data = [[0, 0, 0], [4, 5, 6], [0, 0, 0], [7, 8, 9], [0, 0, 0], [0, 0, 0]]
        toTest = self.constructor(data)

        viewIter = iter(toTest.points)
        toCheck = []
        for v in viewIter:
            toCheck.append(v)

        assert len(toCheck) == len(toTest.points)

        assert toCheck[0][0] == 0
        assert toCheck[0][1] == 0
        assert toCheck[0][2] == 0

        assert toCheck[1][0] == 4
        assert toCheck[1][1] == 5
        assert toCheck[1][2] == 6

        assert toCheck[2][0] == 0
        assert toCheck[2][1] == 0
        assert toCheck[2][2] == 0

        assert toCheck[3][0] == 7
        assert toCheck[3][1] == 8
        assert toCheck[3][2] == 9

        assert toCheck[4][0] == 0
        assert toCheck[4][1] == 0
        assert toCheck[4][2] == 0

        assert toCheck[5][0] == 0
        assert toCheck[5][1] == 0
        assert toCheck[5][2] == 0

    #####################
    # features.__iter__ #
    #####################
    @noLogEntryExpected
    def test_features_iter_PemptyCorrectness(self):
        data = [[], []]
        data = np.array(data).T
        toTest = self.constructor(data)
        fIter = iter(toTest.features)

        fView = next(fIter)
        assert len(fView) == 0
        fView = next(fIter)
        assert len(fView) == 0

        with raises(StopIteration):
            next(fIter)

    def test_features_iter_noNextFempty(self):
        """ test .features() has no next value when object is feature empty """
        data = [[], []]
        data = np.array(data)
        toTest = self.constructor(data)
        viewIter = iter(toTest.features)
        with raises(StopIteration):
            next(viewIter)

    @noLogEntryExpected
    def test_features_iter_exactValueViaFor(self):
        """ Test .features() gives views that contain exactly the correct data """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)

        viewIter = iter(toTest.features)

        toCheck = []
        for v in viewIter:
            toCheck.append(v)

        assert toCheck[0][0] == 1
        assert toCheck[0][1] == 4
        assert toCheck[0][2] == 7
        assert toCheck[1][0] == 2
        assert toCheck[1][1] == 5
        assert toCheck[1][2] == 8
        assert toCheck[2][0] == 3
        assert toCheck[2][1] == 6
        assert toCheck[2][2] == 9


    def test_features_iter_allZeroVectors(self):
        """ Test .features() works when there are all zero features """
        data = [[0, 1, 0, 2, 0, 3, 0, 0], [0, 4, 0, 5, 0, 6, 0, 0], [0, 7, 0, 8, 0, 9, 0, 0]]
        toTest = self.constructor(data)

        viewIter = iter(toTest.features)
        toCheck = []
        for v in viewIter:
            toCheck.append(v)

        assert len(toCheck) == len(toTest.features)
        assert toCheck[0][0] == 0
        assert toCheck[0][1] == 0
        assert toCheck[0][2] == 0

        assert toCheck[1][0] == 1
        assert toCheck[1][1] == 4
        assert toCheck[1][2] == 7

        assert toCheck[2][0] == 0
        assert toCheck[2][1] == 0
        assert toCheck[2][2] == 0

        assert toCheck[3][0] == 2
        assert toCheck[3][1] == 5
        assert toCheck[3][2] == 8

        assert toCheck[4][0] == 0
        assert toCheck[4][1] == 0
        assert toCheck[4][2] == 0

        assert toCheck[5][0] == 3
        assert toCheck[5][1] == 6
        assert toCheck[5][2] == 9

        assert toCheck[6][0] == 0
        assert toCheck[6][1] == 0
        assert toCheck[6][2] == 0

        assert toCheck[7][0] == 0
        assert toCheck[7][1] == 0
        assert toCheck[7][2] == 0

    ############
    # __iter__ #
    ############
    @noLogEntryExpected
    def test_iter_noNextPempty(self):
        data = [[], []]
        data = np.array(data).T
        toTest = self.constructor(data)
        viewIter = iter(toTest)
        with raises(StopIteration):
            next(viewIter)

    @noLogEntryExpected
    def test_iter_noNextFempty(self):
        data = [[], []]
        data = np.array(data)
        toTest = self.constructor(data)
        viewIter = iter(toTest)
        with raises(StopIteration):
            next(viewIter)

    @noLogEntryExpected
    def test_iter_exactValueViaFor_pt(self):
        data = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
        toTest = self.constructor(data)

        toCheck = []
        for v in toTest:
            toCheck.append(v)

        assert toCheck == list(range(1, 10))

    @noLogEntryExpected
    def test_iter_exactValueViaFor_ft(self):
        data = [[1], [2], [3], [4], [5], [6], [7], [8], [9]]
        toTest = self.constructor(data)

        toCheck = []
        for v in toTest:
            toCheck.append(v)

        assert toCheck == list(range(1, 10))

    @raises(ImproperObjectAction)
    def test_iter_exactValueViaFor_2D(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)

        for v in toTest:
            pass

    ###################
    # iterateElements #
    ###################

    @noLogEntryExpected
    def test_iterateElements_noNextPempty(self):
        """ test iterateElements() has no next value when object is point empty """
        data = [[], []]
        data = np.array(data).T
        toTest = self.constructor(data)
        viewIter = iter(toTest.iterateElements())
        with raises(StopIteration):
            next(viewIter)

    def test_iterateElements_noNextFempty(self):
        """ test iterateElements() has no next value when object is feature empty """
        data = [[], []]
        data = np.array(data)
        toTest = self.constructor(data)
        viewIter = iter(toTest.iterateElements())
        with raises(StopIteration):
            next(viewIter)

    @raises(InvalidArgumentValue)
    def test_iterateElements_exception_orderInvalidString(self):
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)

        it = toTest.iterateElements(order='foo')

    @raises(InvalidArgumentType)
    def test_iterateElements_exception_orderInvalidType(self):
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)

        it = toTest.iterateElements(order=1)

    @raises(InvalidArgumentType)
    def test_iterateElements_exception_onlyInvalidType(self):
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)

        it = toTest.iterateElements(only=1)

    @noLogEntryExpected
    def test_iterateElements_exactValueViaFor(self):
        """ Test iterateElements() gives views that contain exactly the correct data """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)

        toCheck = []
        for v in toTest.iterateElements():
            toCheck.append(v)

        assert toCheck == list(range(1, 10))

    def test_iterateElements_allZeroPoints(self):
        """ Test iterateElements() works when there are all zero points """
        data = [[0, 0, 0], [4, 5, 6], [0, 0, 0], [7, 8, 9], [0, 0, 0], [0, 0, 0]]
        toTest = self.constructor(data)

        toCheck = []
        for v in toTest.iterateElements():
            toCheck.append(v)

        assert len(toCheck) == len(toTest.points) * len(toTest.features)

        assert toCheck[0] == 0
        assert toCheck[1] == 0
        assert toCheck[2] == 0

        assert toCheck[3] == 4
        assert toCheck[4] == 5
        assert toCheck[5] == 6

        assert toCheck[6] == 0
        assert toCheck[7] == 0
        assert toCheck[8] == 0

        assert toCheck[9] == 7
        assert toCheck[10] == 8
        assert toCheck[11] == 9

        assert toCheck[12] == 0
        assert toCheck[13] == 0
        assert toCheck[14] == 0

        assert toCheck[15] == 0
        assert toCheck[16] == 0
        assert toCheck[17] == 0

    def test_iterateElements_allZeroVectors(self):
        """ Test iterateElements() works when there are all zero features """
        data = [[0, 1, 0, 2, 0, 3, 0, 0], [0, 4, 0, 5, 0, 6, 0, 0], [0, 7, 0, 8, 0, 9, 0, 0]]
        toTest = self.constructor(data)

        toCheck = []
        for v in toTest.iterateElements():
            toCheck.append(v)

        assert len(toCheck) == len(toTest.features) * len(toTest.points)
        assert toCheck[0] == 0
        assert toCheck[1] == 1
        assert toCheck[2] == 0
        assert toCheck[3] == 2
        assert toCheck[4] == 0
        assert toCheck[5] == 3
        assert toCheck[6] == 0
        assert toCheck[7] == 0

        assert toCheck[8] == 0
        assert toCheck[9] == 4
        assert toCheck[10] == 0
        assert toCheck[11] == 5
        assert toCheck[12] == 0
        assert toCheck[13] == 6
        assert toCheck[14] == 0
        assert toCheck[15] == 0

        assert toCheck[16] == 0
        assert toCheck[17] == 7
        assert toCheck[18] == 0
        assert toCheck[19] == 8
        assert toCheck[20] == 0
        assert toCheck[21] == 9
        assert toCheck[22] == 0
        assert toCheck[23] == 0

    @noLogEntryExpected
    def test_iterateElements_orderPt_onlyNonZero(self):
        data = [[0, 1, 2], [0, 4, 0], [0, 0, 5], [0, 0, 0]]
        obj = self.constructor(data)

        ret = []
        for val in obj.iterateElements(only=match.nonZero):
            ret.append(val)

        assert ret == [1, 2, 4, 5]
        assertNoNamesGenerated(obj)

    def test_iterateElements_orderPt_onlyNonZero_empty(self):
        data = []
        obj = self.constructor(data)

        ret = []
        for val in obj.iterateElements(only=match.nonZero):
            ret.append(val)

        assert ret == []

    @noLogEntryExpected
    def test_iterateElements_orderFt_onlyNonZero(self):
        data = [[0, 1, 2], [0, 4, 0], [0, 0, 5], [0, 0, 0]]
        obj = self.constructor(data)

        ret = []
        for val in obj.iterateElements(order='feature', only=match.nonZero):
            ret.append(val)

        assert ret == [1, 4, 2, 5]
        assertNoNamesGenerated(obj)

    def test_iterateElements_orderFt_onlyNonZero_empty(self):
        data = []
        obj = self.constructor(data)

        ret = []
        for val in obj.iterateElements(order='feature', only=match.nonZero):
            ret.append(val)

        assert ret == []

    ###########
    # inverse #
    ###########
    @noLogEntryExpected
    def test_inverse_multiplicative(self):
        """ Test computation of multiplicative inverse."""
        from scipy import linalg
        data = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]])
        pointNames =  ['1', 'one', '2']
        featureNames = ['one', 'two', 'three']

        data_inv = linalg.inv(data)
        resObj =  self.constructor(data_inv)

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        orig = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        invObj = toTest.inverse()

        assert invObj == resObj
        assert toTest == orig

    @noLogEntryExpected
    def test_inverse_pseudoInverse(self):
        """ Test computation of pseudo-inverse using singular-value decomposition. """
        from scipy import linalg
        data = np.array([[3, 2, 1], [2, 2, 0], [1, 0, 1]])
        pointNames =  ['1', 'one', '2']
        featureNames = ['one', 'two', 'three']

        data_pinv = linalg.pinv2(data)
        resObj = self.constructor(data_pinv)

        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        orig = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        pinvObj = toTest.inverse(pseudoInverse=True)

        assert pinvObj == resObj
        assert toTest == orig

    #####################
    # solveLinearSystem #
    #####################

    def test_solveLinearSystem_solve(self):
        """ Test solveLinearSystem using solve method. """
        self.backend_solveLinearSystem(solveFunction='solve')

    def test_solveLinearSystem_leastSquares(self):
        """ Test solveLinearSystem using least squares method. """
        self.backend_solveLinearSystem(solveFunction='least squares')

    @noLogEntryExpected
    def backend_solveLinearSystem(self, solveFunction):
        from scipy import linalg
        A = np.array([[1, 20], [-30, 4]])
        b = np.array([[-30], [4]])

        x = np.transpose(linalg.solve(A, b))

        pointNames =  ['1', 'one']
        featureNames = ['one', 'two']

        resObj = self.constructor(x, pointNames=['b'], featureNames=featureNames)

        Aobj = self.constructor(A, pointNames=pointNames, featureNames=featureNames)
        origA = self.constructor(A, pointNames=pointNames, featureNames=featureNames)
        bobj = self.constructor(b)

        xobj = Aobj.solveLinearSystem(bobj, solveFunction=solveFunction)

        assert xobj.isApproximatelyEqual(resObj)
        assert Aobj == origA


    @raises(InvalidArgumentType)
    def test_solveLinearSystem_b_InvalidType(self):
        A = np.array([[1, 20], [-30, 4]])
        b = np.array([[-30], [4]])

        Aobj = self.constructor(A)
        Aobj.solveLinearSystem(b)

    @raises(InvalidArgumentType)
    def test_solveLinearSystem_InvalidParamType(self):
        A = np.array([[1, 20], [-30, 4]])
        b = np.array([[-30], [4]])

        Aobj = self.constructor(A)
        bobj = self.constructor(b)

        Aobj.solveLinearSystem(bobj, solveFunction=['solve'])

    @raises(InvalidArgumentValue)
    def test_solveLinearSystem_InvalidParamValue(self):
        A = np.array([[1, 20], [-30, 4]])
        b = np.array([[-30], [4]])

        Aobj = self.constructor(A)
        bobj = self.constructor(b)

        Aobj.solveLinearSystem(bobj, solveFunction='foo')


    ###################
    # features.report #
    ###################

    def test_features_report_allNumeric(self):
        fnames = ['one', 'two', 'three']
        obj = self.constructor([[1, 4, 9], [2, 2, 9.2], [3, 0, 8.8]],
                               featureNames=fnames)

        ret = obj.features.report()

        heads = ['index', 'mean', 'mode', 'minimum', 'Q1', 'median', 'Q3',
                 'maximum', 'uniqueCount', 'count', 'standardDeviation']

        rep = [[0, 2, 2, 1, 1.5, 2, 2.5, 3, 3, 3, 1],
               [1, 2, 2, 0, 1, 2, 3, 4, 3, 3, 2],
               [2, 9, 9, 8.8, 8.9, 9, 9.1, 9.2, 3, 3, 0.2]]

        exp = nimble.data("List", rep, fnames, heads)

        assert ret.isApproximatelyEqual(exp)

    def test_features_report_withNonNumeric(self):
        fnames = ['one', 'two', 'three']
        obj = self.constructor([[1, 2, 'a'], [2, 1, 'b'], [3, 3, 'c']],
                               featureNames=fnames)

        ret = obj.features.report()

        heads = ['index', 'mean', 'mode', 'minimum', 'Q1', 'median', 'Q3',
                 'maximum', 'uniqueCount', 'count', 'standardDeviation']
        rep = [[0, 2, 2, 1, 1.5, 2, 2.5, 3, 3, 3, 1],
               [1, 2, 2, 1, 1.5, 2, 2.5, 3, 3, 3, 1],
               [2, np.nan, 'b', np.nan, np.nan, np.nan, np.nan, np.nan, 3, 3, np.nan]]
        exp = nimble.data("List", rep, fnames, heads)

        assert ret == exp

    def test_features_report_limited(self):
        fnames = ['one', 'two', 'three']
        obj = self.constructor([[1, 4, 9], [2, 2, 9.2], [3, 0, 8.8]],
                               featureNames=fnames)

        selection = ['minimum', 'Q1', 'median', 'Q3', 'maximum', 'mean']

        ret = obj.features.report(selection)

        heads = ['index'] + selection

        rep = [[0, 1, 1.5, 2, 2.5, 3, 2],
               [1, 0, 1, 2, 3, 4, 2],
               [2, 8.8, 8.9, 9, 9.1, 9.2, 9]]

        exp = nimble.data("List", rep, fnames, heads)

        assert ret == exp

    def test_features_report_extraFunctions(self):
        fnames = ['one', 'two', 'three']
        obj = self.constructor([[1, 4, 9], [2, 2, 9.2], [3, 0, 8.8]],
                               featureNames=fnames)

        def minMaxAvg(ft):
            max = nimble.calculate.maximum(ft)
            min = nimble.calculate.minimum(ft)
            return (min + max) / 2

        funcs = [nimble.calculate.sum, minMaxAvg, lambda _: 1, lambda _: 42]

        ret = obj.features.report(basicStatistics=False,
                                  extraStatisticFunctions=funcs)

        heads = ['index', 'sum', 'minMaxAvg', '<lambda> (0)', '<lambda> (1)']

        rep = [[0, 6, 2, 1, 42],
               [1, 6, 2, 1, 42],
               [2, 27, 9, 1, 42]]

        exp = nimble.data("List", rep, fnames, heads)

        assert ret == exp

    @raises(InvalidArgumentValue, match='Invalid value found in basicStatistics')
    def test_features_report_limited_exception(self):
        fnames = ['one', 'two', 'three']
        obj = self.constructor([[1, 4, 9], [2, 2, 9.2], [3, 0, 8.8]],
                               featureNames=fnames)

        selection = ['minimum', 'Q1', 'meedian', 'Q3', 'maximum', 'mean']

        ret = obj.features.report(selection)

    ##########
    # report #
    ##########

    def test_report(self):
        fnames = ['one', 'two', 'three']
        obj = self.constructor([[0, 2, 'a'], [2, None, 'b'], [0, 3, 'c']],
                               featureNames=fnames)

        ret = obj.report()
        heads = ['Values', 'Points', 'Features', 'proportionZero', 'proportionMissing']
        rep =  [9, 3, 3, (2 / 9), (1 / 9)]
        exp = nimble.data("List", rep, featureNames=heads)

        assert ret == exp

        obj = self.constructor([[[[0, 2, 'a']]], [[[2, None, 'b']]], [[[0, 3, 'c']]]])

        ret = obj.report()
        heads = ['Values', 'Dimensions', 'proportionZero', 'proportionMissing']
        rep = [9, '3 x 1 x 1 x 3', (2 / 9), (1 / 9)]
        exp = nimble.data("List", rep, featureNames=heads)

        assert ret == exp

    ######################
    # _axisQueryFunction #
    ######################
    def test_axisQueryString(self):
        """tests both axes for QueryString"""

        def constructObjAndGetAxis(axis, data, offAxisNames):
            if axis == 'points':
                obj = self.constructor(data, featureNames=offAxisNames)
            else:
                obj = self.constructor(np.array(data).T,
                                       pointNames=offAxisNames)
            return getattr(obj, axis)

        def operatorAssertions(axisObj, query, optr):
            equal = axisObj[0]
            notEqual1 = axisObj[1]
            notEqual2 = axisObj[2]
            # axis uses _axisQueryFunction to support more specific exception
            # messages when the QueryString will not work.
            func = axisObj._axisQueryFunction(query)
            if '=' in optr:
                if '==' in optr:
                    assert func(equal)
                    assert not func(notEqual2)
                    assert not func(notEqual1)
                elif '!' in optr:
                    assert not func(equal)
                    assert func(notEqual2)
                    assert func(notEqual1)
                else:
                    assert func(equal)
            if '<' in optr:
                assert func(notEqual2)
                assert not func(notEqual1)
            elif '>' in optr:
                assert func(notEqual1)
                assert not func(notEqual2)
            if optr == 'is':
                assert func(equal)
                assert not func(notEqual2)
                assert not func(notEqual1)

        for axis in ['points', 'features']:
            # Success #
            data = [[0, 1, 2], [3, 4, 5], [-1, -2, -3]]
            offNames = ['one', 'two', 'three']
            primaryAxis = constructObjAndGetAxis(axis, data, offNames)
            for optr in [' == ', ' != ', ' < ', ' <= ', ' > ', ' >= ']:
                query = 'one' + optr + '0'
                operatorAssertions(primaryAxis, query, optr)

            offNames = ['vec one', 'vec two', 'vec three']
            primaryAxis = constructObjAndGetAxis(axis, data, offNames)
            for optr in [' == ', ' != ', ' < ', ' <= ', ' > ', ' >= ']:
                query = 'vec two' + optr + '1'
                operatorAssertions(primaryAxis, query, optr)

            offNames = ['<one>', '<two>', '<three>']
            primaryAxis = constructObjAndGetAxis(axis, data, offNames)
            for optr in [' == ', ' != ', ' < ', ' <= ', ' > ', ' >= ']:
                query = '<three>' + optr + '2'
                operatorAssertions(primaryAxis, query, optr)

            data = [['a', 'a b', '>=2'], ['b', 'b c', '<2'], ['c', 'c d', '<2']]
            offNames = ['one', 'two', 'three']
            primaryAxis = constructObjAndGetAxis(axis, data, offNames)
            for optr in [' == ', ' != ']:
                query = 'one' + optr + 'a'
                operatorAssertions(primaryAxis, query, optr)

                query = 'two' + optr + 'a b'
                operatorAssertions(primaryAxis, query, optr)

                query = 'three' + optr + '>=2'
                operatorAssertions(primaryAxis, query, optr)

            offNames = ['vec one', 'vec two', 'vec three']
            primaryAxis = constructObjAndGetAxis(axis, data, offNames)
            for optr in [' == ', ' != ']:
                query = 'vec one' + optr + 'a'
                operatorAssertions(primaryAxis, query, optr)

            offNames = ['<one>', '<two>', '<three>']
            primaryAxis = constructObjAndGetAxis(axis, data, offNames)
            for optr in [' == ', ' != ']:
                query = '<one>' + optr + 'a'
                operatorAssertions(primaryAxis, query, optr)

                query = '<three>' + optr + '>=2'
                operatorAssertions(primaryAxis, query, optr)

            data = [[None, 1, -2], [-3, -4, 5], [6, -7, 8]]
            offNames = ['one', 'two', 'three']
            primaryAxis = constructObjAndGetAxis(axis, data, offNames)
            for query in ['one is missing', 'three is not positive', 'two is positive']:
                operatorAssertions(primaryAxis, query, 'is')

            # Exceptions #
            data = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
            offNames = ['one', 'two', 'three']
            primaryAxis = constructObjAndGetAxis(axis, data, offNames)
            func = primaryAxis._axisQueryFunction
            # bad whitespace padding on operator
            with raises(InvalidArgumentValue, match='nor a valid query'):
                func('one== 6')
            with raises(InvalidArgumentValue, match='nor a valid query'):
                func('two!=4')
            with raises(InvalidArgumentValue, match='nor a valid query'):
                func('three >7')
            # not a feature name
            with raises(InvalidArgumentValue, match='does not exist'):
                func('four == 4')
            with raises(InvalidArgumentValue, match='does not exist'):
                func(' == 4')
            # no operator
            with raises(InvalidArgumentValue, match='nor a valid query'):
                func('two = 4')
            with raises(InvalidArgumentValue, match='nor a valid query'):
                func('hello')
            # invalid is
            with raises(InvalidArgumentValue, match='nor a valid query'):
                func('one is something')
            with raises(InvalidArgumentValue, match='does not exist'):
                func('two   is missing')

            data = [[0, '> 250k', '== 2'], [3, '> 250k', '!= 2'], [6, '< 250k', '!= 2']]
            offNames = ['one', 'two', 'three']
            primaryAxis = constructObjAndGetAxis(axis, data, offNames)
            func = primaryAxis._axisQueryFunction
            # invalid query value
            with raises(InvalidArgumentValue, match='Multiple operators'):
                func('two == > 250k')
            with raises(InvalidArgumentValue, match='Multiple operators'):
                func('three != == 2')
            # invalid query feature name
            offNames = ['< one >', '< two >', '< three >']
            primaryAxis = constructObjAndGetAxis(axis, data, offNames)
            func = primaryAxis._axisQueryFunction
            with raises(InvalidArgumentValue, match='Multiple operators'):
                func('< one > < 4')
            with raises(InvalidArgumentValue, match='Multiple operators'):
                func('< one > == 4')
            # invalid name and value
            with raises(InvalidArgumentValue, match='Multiple operators'):
                func('< two > == > 250k')
            with raises(InvalidArgumentValue, match='Multiple operators'):
                func('< three > != == 2')

###########
# Helpers #
###########

def checkToStringRet(ret, data, includeNames, maxWidth, maxHeight):
    cHold = '--'
    rHold = '|'
    pnameSep = '   '
    colSep = ' '
    sigDigits = 3
    rows = ret.split('\n')
    rows = rows[:(len(rows) - 1)]

    splitRetLines = ret.split('\n')[:-1] # ends with newline; will ignore
    maxWidth = maxWidth if maxWidth else max(len(line) for line in splitRetLines)
    maxHeight = maxHeight if maxHeight else len(splitRetLines)

    assert len(splitRetLines) <= maxHeight
    assert all(len(line) <= maxWidth for line in splitRetLines)

    negRow = False

    if includeNames:
        rowOffset = 2
        fnamesRaw = rows[0]
        fnamesSplit = fnamesRaw.split(colSep)
        fnames = []
        for val in fnamesSplit:
            if len(val) != 0:
                fnames.append(val)
        # -1 for the fnames,  -1 for the blank row
        assert len(rows) - 2 <= len(data.points)
    else:
        rowOffset = 0
        assert len(rows) <= len(data.points)

    for r in range(rowOffset, len(rows)):
        row = rows[r]
        if includeNames:
            # remove leading whitespace to correctly extract point name
            row = row.strip()
            namesSplit = row.split(pnameSep, 1)
            pname = namesSplit[0]
            row = namesSplit[1]
        spaceSplit = row.split(colSep)
        vals = []
        for possible in spaceSplit:
            if possible != '':
                vals.append(possible)
        if vals[0] == rHold:
            negRow = True
            continue

        rDataIndex = r - rowOffset
        if negRow:
            rDataIndex = -(len(rows) - r)

        negCol = False
        assert len(vals) <= len(data.features)
        if includeNames:
            assert len(fnames) == len(vals)
        for c in range(len(vals)):
            if vals[c] == cHold:
                negCol = True
                continue

            cDataIndex = c
            if negCol:
                cDataIndex = -(len(vals) - c)

            wanted = data[rDataIndex, cDataIndex]
            wantedS = formatIfNeeded(wanted, sigDigits)
            have = vals[c]
            assert wantedS == have

            if includeNames:
                # generate name from indices
                offset = len(data.points) if negRow else 0
                fromIndexPname = data.points.getName(offset + rDataIndex)
                assert fromIndexPname == pname

                offset = len(data.features) if negCol else 0
                fromIndexFname = data.features.getName(offset + cDataIndex)

                # long feature names may have been truncated
                if fnames[cDataIndex].endswith('...'):
                    truncatedLen = len(fnames[cDataIndex]) - 3
                    fromIndexFname = fromIndexFname[:truncatedLen]
                assert fromIndexFname == fnames[cDataIndex]


def test_elementQueryString():
    for optr in ['==', '!=', '<', '<=', '>', '>=']:
        func1 = QueryString(optr + '0')
        func2 = QueryString(optr + ' ' + '0')
        if '=' in optr:
            if '!' in optr:
                assert not func1(0)
                assert not func2(0)
            else:
                assert func1(0)
                assert func2(0)
        if '>' in optr:
            assert func1(1)
            assert func2(1)
        if '<' in optr:
            assert func1(-1)
            assert func2(-1)

        if optr in ['==', '!=']:
            func3 = QueryString(optr + 'hi')
            func4 = QueryString(optr + ' ' + 'hi')
            if '!' in optr:
                assert func3('hello')
                assert not func3('hi')
                assert func4('hello')
                assert not func4('hi')
            else:
                assert func3('hi')
                assert not func3('hello')
                assert func4('hi')
                assert not func4('hello')

            func5 = QueryString(optr + 'hi hello')
            func6 = QueryString(optr + ' ' + 'hi hello')
            if '!' in optr:
                assert func5('hello hi')
                assert not func5('hi hello')
                assert func6('hello hi')
                assert not func6('hi hello')
            else:
                assert func5('hi hello')
                assert not func5('hello hi')
                assert func6('hi hello')
                assert not func6('hello hi')


    func7 = QueryString('is missing')
    assert func7(None)
    assert func7(np.nan)
    assert not func7(1)
    assert not func7('None')

    func8 = QueryString('is not negative')
    assert func8(4.0)
    assert func8(0)
    assert not func8(-1)
    assert not func8(-0.1)

    func9 = QueryString('is None')
    assert func9(None)
    assert not func9(np.nan)
    assert not func9(1)
    assert not func9('None')

    func10 = QueryString('is True')
    assert func10(True)
    assert not func10('True')
    assert not func10(1)

    # invalid query strings should return None
    for invalid in [lambda x: 5, '= 0', '=0', 'is something']:
        with raises(InvalidArgumentValue):
            QueryString(invalid)

def convertFeatureReportValues(val):
    try:
        flt = float(val)
        try:
            itgr = int(val)
            if flt != itgr:
                return flt
            return itgr
        except ValueError:
            if flt != flt:
                return val
            return flt
    except ValueError:
        return val
