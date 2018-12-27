"""
Methods tested in this file (none modify the data):

pointCount, featureCount, isIdentical, writeFile, __getitem__,
pointView, featureView, view, containsZero, __eq__, __ne__, toString,
pointSimilarities, featureSimilarities, pointStatistics,
featureStatistics, points.__iter__, features.__iter__,
points.nonZeroIterator, features.nonZeroIterator
"""

from __future__ import absolute_import
from __future__ import print_function
import math
import tempfile
import numpy
import os
import os.path
from nose.tools import *
from nose.plugins.attrib import attr

from copy import deepcopy

import UML
from UML import loadData
from UML.data import BaseView
from UML.data.tests.baseObject import DataTestObject
from UML.data.dataHelpers import formatIfNeeded
from UML.data.dataHelpers import makeConsistentFNamesAndData
from UML.data.dataHelpers import DEFAULT_PREFIX
from UML.exceptions import ArgumentException
from six.moves import map
from six.moves import range
from six.moves import zip
from functools import reduce

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
        dataPEmpty = numpy.array(data).T
        dataFEmpty = numpy.array(data)

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


    #################
    # featureCount #
    #################


    def test_featureCount_empty(self):
        """ test featureCount when given different kinds of emptiness """
        data = [[], []]
        dataPEmpty = numpy.array(data).T
        dataFEmpty = numpy.array(data)

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


    #################
    # isIdentical() #
    #################

    def test_isIdentical_False(self):
        """ Test isIdentical() against some non-equal input """
        toTest = self.constructor([[4, 5]])
        assert not toTest.isIdentical(self.constructor([[1, 1], [2, 2]]))
        assert not toTest.isIdentical(self.constructor([[1, 2, 3]]))
        assert not toTest.isIdentical(self.constructor([[1, 2]]))

    def test_isIdentical_FalseBozoTypes(self):
        """ Test isIdentical() against some non-equal input of crazy types """
        toTest = self.constructor([[4, 5]])
        assert not toTest.isIdentical(numpy.matrix([[1, 1], [2, 2]]))
        assert not toTest.isIdentical('self.constructor([[1,2,3]])')
        assert not toTest.isIdentical(toTest.isIdentical)

    def test_isIdentical_True(self):
        """ Test isIdentical() against some actually equal input """
        toTest1 = self.constructor([[4, 5]])
        toTest2 = self.constructor(deepcopy([[4, 5]]))
        assert toTest1.isIdentical(toTest2)
        assert toTest2.isIdentical(toTest1)

    def test_isIdentical_FalseWithNaN(self):
        """ Test isIdentical() against some non-equal input with nan"""
        toTest1 = self.constructor([[1, numpy.nan, 5]])
        toTest2 = self.constructor(deepcopy([[1, numpy.nan, 3]]))
        assert not toTest1.isIdentical(toTest2)
        assert not toTest2.isIdentical(toTest1)

    def test_isIdentical_TrueWithNaN(self):
        """ Test isIdentical() against some actually equal input with nan """
        toTest1 = self.constructor([[1, numpy.nan, 5]])
        toTest2 = self.constructor(deepcopy([[1, numpy.nan, 5]]))
        assert toTest1.isIdentical(toTest2)
        assert toTest2.isIdentical(toTest1)


    ############
    # writeFile #
    ############

    def test_writeFile_CSVhandmade(self):
        """ Test writeFile() for csv extension with both data and featureNames """
        tmpFile = tempfile.NamedTemporaryFile(suffix=".csv")

        # instantiate object
        data = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        pointNames = ['1', 'one', '2', '0']
        featureNames = ['one', 'two', 'three']
        toWrite = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        orig = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        # call writeFile
        toWrite.writeFile(tmpFile.name, format='csv', includeNames=True)

        # read it back into a different object, then test equality
        readObj = self.constructor(data=tmpFile.name)

        assert readObj.isIdentical(toWrite)
        assert toWrite.isIdentical(readObj)

        assert toWrite == orig

    def test_writeFile_CSVauto(self):
        """ Test writeFile() will (if needed) autoconvert to Matrix to use its CSV output """
        tmpFile = tempfile.NamedTemporaryFile(suffix=".csv")

        # instantiate object
        data = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        pointNames = ['1', 'one', '2', '0']
        featureNames = ['one', 'two', 'three']
        toWrite = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        # cripple all but cannonical implementation
        if self.returnType != 'Matrix':
            toWrite._writeFile_implementation = None

        # call writeFile
        toWrite.writeFile(tmpFile.name, format='csv', includeNames=True)
        # read it back into a different object, then test equality
        readObj = self.constructor(data=tmpFile.name)

        assert readObj.isIdentical(toWrite)
        assert toWrite.isIdentical(readObj)


    def test_writeFile_CSV_excludeDefaultNames(self):
        tmpFile = tempfile.NamedTemporaryFile(suffix=".csv")

        def getDefNameIndex(name):
            return int(name[len(DEFAULT_PREFIX):])

        data = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        pointNames = ['1', 'one', '2', '0']
        featureNames = ['one', 'two', 'three']

        def excludeAxis(axis):
            if axis == 'point':
                exclude = self.constructor(data, featureNames=featureNames)
                if isinstance(exclude, UML.data.BaseView):
                    setter = exclude._source.points.setNames
                else:
                    setter = exclude.points.setNames
                count = len(exclude.points)
            else:
                exclude = self.constructor(data, pointNames=pointNames)
                if isinstance(exclude, UML.data.BaseView):
                    setter = exclude._source.features.setNames
                else:
                    setter = exclude.features.setNames
                count = len(exclude.features)

            # increase the index of the default point name so that it will be
            # recognizable when we read in from the file.
            axisExclude = getattr(exclude, axis + 's')
            while (getDefNameIndex(getattr(axisExclude, 'getName')(0)) <= 100):
                setter(None)

            # call writeFile
            exclude.writeFile(tmpFile.name, format='csv', includeNames=True)

            # read it back into a different object, then test equality
            if axis == 'point':
                readObj = self.constructor(data=tmpFile.name, featureNames=True)
            else:
                readObj = self.constructor(data=tmpFile.name, pointNames=True)
            axisRead = getattr(readObj, axis + 's')
            # isIdentical will ignore default names, but we still want to
            # ensure everything else is a match
            assert readObj.isIdentical(exclude)
            assert exclude.isIdentical(readObj)

            for i in range(count):
                origName = getattr(axisExclude, 'getName')(i)
                readName = getattr(axisRead, 'getName')(i)
                assert getDefNameIndex(origName) > 100
                assert getDefNameIndex(readName) < 10

        excludeAxis('point')
        excludeAxis('feature')


    def test_writeFile_MTXhandmade(self):
        """ Test writeFile() for mtx extension with both data and featureNames """
        tmpFile = tempfile.NamedTemporaryFile(suffix=".mtx")

        # instantiate object
        data = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        featureNames = ['one', 'two', 'three']
        pointNames = ['1', 'one', '2', '0']
        toWrite = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        # call writeFile
        toWrite.writeFile(tmpFile.name, format='mtx', includeNames=True)

        # read it back into a different object, then test equality
        readObj = self.constructor(data=tmpFile.name)

        assert readObj.isIdentical(toWrite)
        assert toWrite.isIdentical(readObj)

    def test_writeFile_MTXauto(self):
        """ Test writeFile() will (if needed) autoconvert to Matrix to use its MTX output """
        tmpFile = tempfile.NamedTemporaryFile(suffix=".mtx")

        # instantiate object
        data = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        featureNames = ['one', 'two', 'three']
        pointNames = ['1', 'one', '2', '0']
        toWrite = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        # cripple all but cannonical implementation
        if self.returnType != 'Sparse':
            toWrite._writeFile_implementation = None

        # call writeFile
        toWrite.writeFile(tmpFile.name, format='mtx', includeNames=True)

        # read it back into a different object, then test equality
        readObj = self.constructor(data=tmpFile.name)

        assert readObj.isIdentical(toWrite)
        assert toWrite.isIdentical(readObj)

    #################
    # save/LoadData #
    #################

    def test_save(self):
        tmpFile = tempfile.NamedTemporaryFile(suffix=".umld")

        data = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        featureNames = ['one', 'two', 'three']
        pointNames = ['1', 'one', '2', '0']
        toSave = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        toSave.save(tmpFile.name)

        LoadObj = loadData(tmpFile.name)
        assert toSave.isIdentical(LoadObj)
        assert LoadObj.isIdentical(toSave)

    def test_save_extensionHandling(self):
        tmpFile = tempfile.NamedTemporaryFile()
        data = [[1, 2, 3], [1, 2, 3], [2, 4, 6], [0, 0, 0]]
        featureNames = ['one', 'two', 'three']
        pointNames = ['1', 'one', '2', '0']
        toSave = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        toSave.save(tmpFile.name)
        LoadObj = loadData(tmpFile.name + '.umld')
        assert isinstance(LoadObj, UML.data.Base)

        try:
            LoadObj = loadData(tmpFile.name)
        except ArgumentException as ae:
            assert True
        else:
            assert False


    ##############
    # __getitem__#
    ##############

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
        assert toTest[1:4, :] == tmp2
        assert toTest[[1,2,3], :] == tmp2
        assert toTest[['4', '7', '0'], :] == tmp2

        tmp3 = self.constructor([['f'], ['m'], ['f'], ['m']], featureNames=['gender'], pointNames=pnames)
        assert toTest[:, 4] == tmp3
        assert toTest[:, 'gender'] == tmp3

        tmp4 = self.constructor([['f', 0], ['m', 0], ['f', 0], ['m', 0]], featureNames=['gender', 'zero'], pointNames=pnames)
        assert toTest[:, 4:2:-1] == tmp4
        assert toTest[:, [4,3]] == tmp4
        assert toTest[:, ['gender', 'zero']] == tmp4

        tmp5 = self.constructor([['f', 0], ['m', 0]], featureNames=['gender', 'zero'], pointNames=pnames[:2])
        assert toTest[:2, 4:2:-1] == tmp5
        assert toTest[['1', '4'], [4,3]] == tmp5
        assert toTest[[0,1], ['gender', 'zero']] == tmp5


    def test_getitem_simpleExampeWithZeroes(self):
        """ Test __getitem__ returns the correct output for a number of simple queries """
        featureNames = ["one", "two", "three", "zero"]
        pnames = ['1', '4', '7', '0']
        data = [[1, 2, 3, 0], [4, 5, 0, 0], [7, 0, 9, 0], [0, 0, 0, 0]]

        toTest = self.constructor(data, pointNames=pnames, featureNames=featureNames)

        assert toTest[0, 0] == 1
        assert toTest[1, 3] == 0
        assert toTest['7', 2] == 9
        assert toTest[3, 'zero'] == 0

        assert toTest[1, 'one'] == 4

    @raises(ArgumentException)
    def test_getitem_nonIntConvertableFloatSingleKey(self):
        data = [[0, 1, 2, 3]]
        toTest = self.constructor(data)

        assert toTest[0.1] == 0

    @raises(ArgumentException)
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


    ################
    # pointView #
    ################

    def test_pointView_FEmpty(self):
        """ Test pointView() when accessing a feature empty object """
        data = [[], []]
        data = numpy.array(data)
        toTest = self.constructor(data)

        v = toTest.pointView(0)

        assert len(v.features) == 0


    def test_pointView_isinstance(self):
        pointNames = ['1', '4', '7']
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        pView = toTest.pointView(0)

        assert isinstance(pView, BaseView)
        assert pView.name != toTest.name
        assert len(pView.points) == 1
        assert len(pView.features) == 3
        assert len(pView) == len(toTest.features)
        assert pView[0] == 1
        assert pView['two'] == 2
        assert pView['three'] == 3

    ##################
    # featureView #
    ##################

    def test_featureView_FEmpty(self):
        """ Test featureView() when accessing a point empty object """
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)

        v = toTest.featureView(0)

        assert len(v.points) == 0

    def test_featureView_isinstance(self):
        """ Test featureView() returns an instance of the BaseView """
        pointNames = ['1', '4', '7']
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        fView = toTest.featureView('one')

        assert isinstance(fView, BaseView)
        assert fView.name != toTest.name
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

        try:
            toTest.view(pointStart=1.5)
            assert False  # pointStart is non-ID didn't raise exception
        except ArgumentException as ae:
            if textCheck:
                print(ae)

        try:
            toTest.view(pointEnd=5)
            assert False  # pointEnd > pointCount didn't raise exception
        except ArgumentException as ae:
            if textCheck:
                print(ae)

        try:
            toTest.view(pointEnd=1.4)
            assert False  # pointEnd is non-ID didn't raise exception
        except ArgumentException as ae:
            if textCheck:
                print(ae)

        try:
            toTest.view(pointStart='7', pointEnd='4')
            assert False  # pointStart > pointEnd didn't raise exception
        except ArgumentException as ae:
            if textCheck:
                print(ae)

    def test_view_featureStart_featureEnd_validation(self):
        pointNames = ['1', '4', '7']
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        textCheck = False

        try:
            toTest.view(featureStart=1.5)
            assert False  # featureStart is non-ID didn't raise exception
        except ArgumentException as ae:
            if textCheck:
                print(ae)

        try:
            toTest.view(featureEnd=4)
            assert False  # featureEnd > featureCount didn't raise exception
        except ArgumentException as ae:
            if textCheck:
                print(ae)

        try:
            toTest.view(featureEnd=1.4)
            assert False  # featureEnd is non-ID didn't raise exception
        except ArgumentException as ae:
            if textCheck:
                print(ae)

        try:
            toTest.view(featureStart='three', featureEnd='two')
            assert False  # featureStart > featureEnd didn't raise exception
        except ArgumentException as ae:
            if textCheck:
                print(ae)


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

    @attr('slow')
    def test_toString_nameAndValRecreation_randomized(self):
        """ Regression test with random data and limits. Recreates expected results """
        for pNum in [3, 9]:
            for fNum in [2, 5, 8, 15]:
                randGen = UML.createRandomData("List", pNum, fNum, 0, elementType='int')
                raw = randGen.data

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
                            checkToStringRet(ret, data, inc)

    def test_toString_emptyObjects(self):
        # no checks, but this at least confirms that it is runnable
        names = ['n1', 'n2', 'n3']

        rawPEmpty = numpy.zeros((0, 3))
        objPEmpty = self.constructor(rawPEmpty, featureNames=names)

        assert objPEmpty.toString() == ""

        rawFEmpty = numpy.zeros((3, 0))
        objFEmpty = self.constructor(rawFEmpty, pointNames=names)

        assert objFEmpty.toString() == ""


    # makeConsistentFNamesAndData(fnames, dataTable, dataWidths,colHold):
    def test_makeConsistentFNamesAndData_completeData(self):
        colHold = '--'
        chLen = len(colHold)

        fnames = ['one', colHold, 'four']
        data = [['333', '4444', '22', '1']]
        dataWidths = [3, 4, 2, 1]

        makeConsistentFNamesAndData(fnames, data, dataWidths, colHold)

        expNames = ['one', colHold, 'four']
        expData = [['333', colHold, '1']]
        expDataWidhts = [3, chLen, 1]

        assert fnames == expNames
        assert data == expData
        assert dataWidths == expDataWidhts

    def test_makeConsistentFNamesAndData_completeNames(self):
        colHold = '--'
        chLen = len(colHold)

        fnames = ['one', 'two', 'three', 'four']
        data = [['333', colHold, '1']]
        dataWidths = [3, chLen, 1]

        makeConsistentFNamesAndData(fnames, data, dataWidths, colHold)

        expNames = ['one', colHold, 'four']
        expData = [['333', colHold, '1']]
        expDataWidhts = [3, chLen, 1]

        assert fnames == expNames
        assert data == expData
        assert dataWidths == expDataWidhts

    def test_makeConsistentFNamesAndData_allComplete(self):
        colHold = '--'
        chLen = len(colHold)

        fnames = ['one', 'two', 'three', 'four']
        data = [['333', '22', '666666', '1']]
        dataWidths = [3, 2, 6, 1]

        makeConsistentFNamesAndData(fnames, data, dataWidths, colHold)

        expNames = ['one', 'two', 'three', 'four']
        expData = [['333', '22', '666666', '1']]
        expDataWidhts = [3, 2, 6, 1]

        assert fnames == expNames
        assert data == expData
        assert dataWidths == expDataWidhts

    def test_makeConsistentFNamesAndData_bothIncomplete(self):
        colHold = '--'
        chLen = len(colHold)

        fnames = ['one', 'two', colHold, 'five']
        data = [['333', '22', colHold, '4444', '1']]
        dataWidths = [3, 2, chLen, 4, 1]

        makeConsistentFNamesAndData(fnames, data, dataWidths, colHold)

        expNames = ['one', 'two', colHold, 'five']
        expData = [['333', '22', colHold, '1']]
        expDataWidhts = [3, 2, chLen, 1]

        assert fnames == expNames
        assert data == expData
        assert dataWidths == expDataWidhts

    def test_makeConsistentFNamesAndData_incompleteIsSameLength(self):
        colHold = '--'
        chLen = len(colHold)

        fnames = ['one', 'two', colHold, 'five']
        data = [['333', '22', '4444', '1']]
        dataWidths = [3, 2, 4, 1]

        makeConsistentFNamesAndData(fnames, data, dataWidths, colHold)

        expNames = ['one', 'two', colHold, 'five']
        expData = [['333', '22', colHold, '1']]
        expDataWidhts = [3, 2, chLen, 1]

        assert fnames == expNames
        assert data == expData
        assert dataWidths == expDataWidhts


    def test_makeConsistentFNamesAndData_largeLengthDifference(self):
        colHold = '--'
        chLen = len(colHold)

        fnames = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        data = [['333', '22', colHold, '1']]
        dataWidths = [3, 2, chLen, 1]

        makeConsistentFNamesAndData(fnames, data, dataWidths, colHold)

        expNames = ['1', '2', colHold, '9']
        expData = [['333', '22', colHold, '1']]
        expDataWidhts = [3, 2, chLen, 1]

        assert fnames == expNames
        assert data == expData
        assert dataWidths == expDataWidhts


    # _arrangeFeatureNames(self, maxWidth, nameLength, colSep, colHold, nameHold):
    def test_arrangeFeatureNames_correctSplit(self):
        colSep = ' '
        colHold = '--'
        nameHold = '...'

        raw = [[300, 310, 320, 330], [301, 311, 321, 331], [302, 312, 322, 332]]
        initnames = ['zero', 'one', 'two', 'three']
        obj = self.constructor(raw, featureNames=initnames)

        fnames = obj._arrangeFeatureNames(9, 11, colSep, colHold, nameHold)
        assert fnames == ['zero', '--']

    def test_arrangeFeatureNames_correctTruncation(self):
        colSep = ' '
        colHold = '--'
        nameHold = '...'

        raw = [[300, 310, 320, 330], [301, 311, 321, 331], [302, 312, 322, 332]]
        initnames = ['zerooo', 'one', 'two', 'threee']
        obj = self.constructor(raw, featureNames=initnames)

        fnames = obj._arrangeFeatureNames(80, 3, colSep, colHold, nameHold)
        assert fnames == ['...', 'one', 'two', '...']

    def test_arrangeFeatureNames_omitDefault(self):
        colSep = ' '
        colHold = '--'
        nameHold = '...'

        raw = [[300, 310, 320, 330], [301, 311, 321, 331], [302, 312, 322, 332]]
        initnames = [None, 'one', None, 'three']
        obj = self.constructor(raw, featureNames=initnames)

        fnames = obj._arrangeFeatureNames(80, 11, colSep, colHold, nameHold)
        assert fnames == ['', 'one', '', 'three']


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

    @raises(ArgumentException)
    def test_arrangeDataWithLimits_exception_maxH(self):
        randGen = UML.createRandomData("List", 5, 5, 0, elementType='int')
        randGen._arrangeDataWithLimits(maxHeight=1, maxWidth=120)

    @attr('slow')
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

            return UML.createData(rType, raw)

        def runTrial(pNum, fNum, valLen, maxW, maxH, colSep):
            if pNum == 0 and fNum == 0:
                return
            elif pNum == 0:
                data = makeUniformLength("List", 1, fNum, valLen)
                data.points.extract(0)
            elif fNum == 0:
                data = makeUniformLength("List", pNum, 1, valLen)
                data.features.extract(0)
            else:
                if valLen is None:
                    data = UML.createRandomData("List", pNum, fNum, .25, elementType='int')
                else:
                    data = makeUniformLength("List", pNum, fNum, valLen)
                #			raw = data.data
            ret, widths = data._arrangeDataWithLimits(maxW, maxH, colSep=colSep)

            assert len(ret) <= maxH
            for pRep in ret:
                assert len(pRep) == len(widths)
                lenSum = 0
                for val in pRep:
                    lenSum += len(val)
                assert lenSum <= (maxW - ((len(pRep) - 1) * len(colSep)))

            if len(ret) > 0:
                for fIndex in range(len(ret[0])):
                    widthBound = 0
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
                            for colSep in ['', ' ', ' ']:
                                runTrial(pNum, fNum, valLen, maxW, maxH, colSep)


    ##################### #######################
    # pointSimilarities # # featureSimilarities #
    ##################### #######################

    @raises(ArgumentException)
    def test_pointSimilaritesInvalidParamType(self):
        """ Test pointSimilarities raise exception for unexpected param type """
        self.backend_Sim_InvalidParamType(True)

    @raises(ArgumentException)
    def test_featureSimilaritesInvalidParamType(self):
        """ Test featureSimilarities raise exception for unexpected param type """
        self.backend_Sim_InvalidParamType(False)

    def backend_Sim_InvalidParamType(self, axis):
        data = [[1, 2], [3, 4]]
        obj = self.constructor(data)

        if axis:
            obj.pointSimilarities({"hello": 5})
        else:
            obj.featureSimilarities({"hello": 5})

    @raises(ArgumentException)
    def test_pointSimilaritesUnexpectedString(self):
        """ Test pointSimilarities raise exception for unexpected string value """
        self.backend_Sim_UnexpectedString(True)

    @raises(ArgumentException)
    def test_featureSimilaritesUnexpectedString(self):
        """ Test featureSimilarities raise exception for unexpected string value """
        self.backend_Sim_UnexpectedString(False)

    def backend_Sim_UnexpectedString(self, axis):
        data = [[1, 2], [3, 4]]
        obj = self.constructor(data)

        if axis:
            obj.pointSimilarities("foo")
        else:
            obj.featureSimilarities("foo")


    # test results covariance
    def test_pointSimilaritesSampleCovarianceResult(self):
        """ Test pointSimilarities returns correct sample covariance results """
        self.backend_Sim_SampleCovarianceResult(True)

    def test_featureSimilaritesSampleCovarianceResult(self):
        """ Test featureSimilarities returns correct sample covariance results """
        self.backend_Sim_SampleCovarianceResult(False)

    def backend_Sim_SampleCovarianceResult(self, axis):
        data = [[3, 0, 3], [0, 0, 3], [3, 0, 0]]
        dataT = numpy.array(data).T.tolist()
        orig = self.constructor(data)
        trans = self.constructor(dataT)
        sameAsOrig = self.constructor(data)
        sameAsOrigT = self.constructor(dataT)

        if axis:
            ret = orig.pointSimilarities("covariance ")
        else:
            ret = trans.featureSimilarities("sample\tcovariance")
            ret.transpose()

        # hand computed results
        expRow0 = [3, 1.5, 1.5]
        expRow1 = [1.5, 3, -1.5]
        expRow2 = [1.5, -1.5, 3]
        expData = [expRow0, expRow1, expRow2]
        expObj = self.constructor(expData)

        # numpy computted result -- bias=0 -> divisor of n-1
        npExpRaw = numpy.cov(data, bias=0)
        npExpObj = self.constructor(npExpRaw)
        assert ret.isApproximatelyEqual(npExpObj)

        assert expObj.isApproximatelyEqual(ret)
        assert sameAsOrig == orig
        assert sameAsOrigT == trans

    def test_pointSimilaritesPopulationCovarianceResult(self):
        """ Test pointSimilarities returns correct population covariance results """
        self.backend_Sim_populationCovarianceResult(True)

    def test_featureSimilaritesPopulationCovarianceResult(self):
        """ Test featureSimilarities returns correct population covariance results """
        self.backend_Sim_populationCovarianceResult(False)

    def backend_Sim_populationCovarianceResult(self, axis):
        data = [[3, 0, 3], [0, 0, 3], [3, 0, 0]]
        dataT = numpy.array(data).T.tolist()
        orig = self.constructor(data)
        trans = self.constructor(dataT)
        sameAsOrig = self.constructor(data)
        sameAsOrigT = self.constructor(dataT)

        if axis:
            ret = orig.pointSimilarities("population COvariance")
        else:
            ret = trans.featureSimilarities("populationcovariance")
            ret.transpose()

        # hand computed results
        expRow0 = [2, 1, 1]
        expRow1 = [1, 2, -1]
        expRow2 = [1, -1, 2]
        expData = [expRow0, expRow1, expRow2]
        expObj = self.constructor(expData)

        # numpy computted result -- bias=1 -> divisor of n
        npExpRaw = numpy.cov(data, bias=1)
        npExpObj = self.constructor(npExpRaw)
        assert ret.isApproximatelyEqual(npExpObj)

        assert expObj.isApproximatelyEqual(ret)
        assert sameAsOrig == orig
        assert sameAsOrigT == trans

    def test_pointSimilaritesSTDandVarianceIdentity(self):
        """ Test identity between population covariance and population std of points """
        self.backend_Sim_STDandVarianceIdentity(True)

    def test_featureSimilaritesSTDandVarianceIdentity(self):
        """ Test identity between population covariance and population std of features """
        self.backend_Sim_STDandVarianceIdentity(False)

    def backend_Sim_STDandVarianceIdentity(self, axis):
        data = [[3, 0, 3], [0, 0, 3], [3, 0, 0]]
        dataT = numpy.array(data).T.tolist()
        orig = self.constructor(data)
        trans = self.constructor(dataT)

        if axis:
            ret = orig.pointSimilarities(" populationcovariance")
            stdVector = orig.pointStatistics("population std")
        else:
            ret = trans.featureSimilarities("populationcovariance")
            stdVector = trans.featureStatistics("\npopulationstd")
            ret.transpose()

        numpy.testing.assert_approx_equal(ret[0, 0], stdVector[0] * stdVector[0])
        numpy.testing.assert_approx_equal(ret[1, 1], stdVector[1] * stdVector[1])
        numpy.testing.assert_approx_equal(ret[2, 2], stdVector[2] * stdVector[2])


    # test results correlation
    def test_pointSimilaritesCorrelationResult(self):
        """ Test pointSimilarities returns correct correlation results """
        self.backend_Sim_CorrelationResult(True)

    def test_featureSimilaritesCorrelationResult(self):
        """ Test featureSimilarities returns correct correlation results """
        self.backend_Sim_CorrelationResult(False)

    def backend_Sim_CorrelationResult(self, axis):
        data = [[3, 0, 3], [0, 0, 3], [3, 0, 0]]
        dataT = numpy.array(data).T.tolist()
        orig = self.constructor(data)
        trans = self.constructor(dataT)
        sameAsOrig = self.constructor(data)
        sameAsOrigT = self.constructor(dataT)

        if axis:
            ret = orig.pointSimilarities("correlation")
        else:
            ret = trans.featureSimilarities("corre lation")
            ret.transpose()

        expRow0 = [1, (1. / 2), (1. / 2)]
        expRow1 = [(1. / 2), 1, (-1. / 2)]
        expRow2 = [(1. / 2), (-1. / 2), 1]
        expData = [expRow0, expRow1, expRow2]
        expObj = self.constructor(expData)

        npExpRaw = numpy.corrcoef(data)
        npExpObj = self.constructor(npExpRaw)

        assert ret.isApproximatelyEqual(npExpObj)
        assert expObj.isApproximatelyEqual(ret)
        assert sameAsOrig == orig
        assert sameAsOrigT == trans

    def test_pointSimilaritesCorrelationHelpersEquiv(self):
        """ Compare pointSimilarities correlation using the various possible helpers """
        self.backend_Sim_CorrelationHelpersEquiv(True)

    def test_featureSimilaritesCorrelationHelpersEquiv(self):
        """ Compare featureSimilarities correlation using the various possible helpers """
        self.backend_Sim_CorrelationHelpersEquiv(False)

    def backend_Sim_CorrelationHelpersEquiv(self, axis):
        data = [[3, 0, 3], [0, 0, 3], [3, 0, 0]]
        dataT = numpy.array(data).T.tolist()
        orig = self.constructor(data)
        trans = self.constructor(dataT)
        sameAsOrig = self.constructor(data)

        def explicitCorr(X, sample=True):
            sampleStdVector = X.pointStatistics('samplestd')
            popStdVector = X.pointStatistics('populationstd')
            stdVector = sampleStdVector if sample else popStdVector

            stdVector_T = stdVector.copy()
            stdVector_T.transpose()

            if sample:
                cov = X.pointSimilarities('sample covariance')
            else:
                cov = X.pointSimilarities('population Covariance')

            stdMatrix = stdVector * stdVector_T
            ret = cov / stdMatrix

            return ret

        if axis:
            ret = orig.pointSimilarities("correlation")
            sampRet = explicitCorr(orig, True)
            popRet = explicitCorr(orig, False)
        else:
            ret = trans.featureSimilarities("correlation")
            # helper only calls pointStatistics, so have to make sure
            # that in this case, we are calling with the transpose of
            # the object used to test featureSimilarities
            sampRet = explicitCorr(orig, True)
            popRet = explicitCorr(orig, False)
            ret.transpose()

        npExpRawB0 = numpy.corrcoef(data, bias=0)
        npExpRawB1 = numpy.corrcoef(data, bias=1)
        npExpB0 = self.constructor(npExpRawB0)
        npExpB1 = self.constructor(npExpRawB1)

        assert ret.isApproximatelyEqual(sampRet)
        assert ret.isApproximatelyEqual(popRet)
        assert sampRet.isApproximatelyEqual(popRet)
        assert ret.isApproximatelyEqual(npExpB0)
        assert ret.isApproximatelyEqual(npExpB1)


    # test results dot product
    def test_pointSimilaritesDotProductResult(self):
        """ Test pointSimilarities returns correct dot product results """
        self.backend_Sim_DotProductResult(True)

    def test_featureSimilaritesDotProductResult(self):
        """ Test featureSimilarities returns correct dot product results """
        self.backend_Sim_DotProductResult(False)

    def backend_Sim_DotProductResult(self, axis):
        data = [[1, 1, 1], [0, 1, 1], [1, 0, 0]]
        dataT = numpy.array(data).T.tolist()
        orig = self.constructor(data)
        trans = self.constructor(dataT)
        sameAsOrig = self.constructor(data)
        sameAsOrigT = self.constructor(dataT)

        if axis:
            ret = orig.pointSimilarities("Dot Product")
        else:
            ret = trans.featureSimilarities("dotproduct\n")
            ret.transpose()

        expData = [[3, 2, 1], [2, 2, 0], [1, 0, 1]]
        expObj = self.constructor(expData)

        assert expObj == ret
        assert sameAsOrig == orig
        assert sameAsOrigT == trans

    # test input function validation
    @raises(ArgumentException)
    def todotest_pointSimilaritesFuncValidation(self):
        """ Test pointSimilarities raises exception for invalid funcitions """
        self.backend_Sim_FuncValidation(True)

    @raises(ArgumentException)
    def todotest_featureSimilaritesFuncValidation(self):
        """ Test featureSimilarities raises exception for invalid funcitions """
        self.backend_Sim_FuncValidation(False)

    def backend_Sim_FuncValidation(self, axis):
        assert False
        data = [[1, 2], [3, 4]]
        obj = self.constructor(data)

        def singleArg(one):
            return one

        if axis:
            obj.pointSimilarities(singleArg)
        else:
            obj.featureSimilarities(singleArg)

    # test results passed function
    def todotest_pointSimilariteGivenFuncResults(self):
        """ Test pointSimilarities returns correct results for given function """
        self.backend_Sim_GivenFuncResults(True)

    def todotest_featureSimilaritesGivenFuncResults(self):
        """ Test featureSimilarities returns correct results for given function """
        self.backend_Sim_GivenFuncResults(False)

    def backend_Sim_GivenFuncResults(self, axis):
        assert False
        data = [[1, 2], [3, 4]]
        obj = self.constructor(data)

        def euclideanDistance(left, right):
            assert False

        if axis:
            obj.pointSimilarities(euclideanDistance)
        else:
            obj.featureSimilarities(euclideanDistance)

    def backend_Sim_NamePath_Preservation(self, axis):
        data = [[3, 0, 3], [0, 0, 3], [3, 0, 0]]
        dataT = numpy.array(data).T.tolist()
        orig = self.constructor(data, name=preserveName, path=preservePair)
        trans = self.constructor(dataT, name=preserveName, path=preservePair)

        possible = [
            'correlation', 'covariance', 'dotproduct', 'samplecovariance',
            'populationcovariance'
        ]

        for curr in possible:
            if axis:
                ret = orig.pointSimilarities(curr)
            else:
                ret = trans.featureSimilarities(curr)

            assert orig.name == preserveName
            assert orig.absolutePath == preserveAPath
            assert orig.relativePath == preserveRPath

            assert ret.nameIsDefault()
            assert ret.absolutePath == preserveAPath
            assert ret.relativePath == preserveRPath

    def test_pointSimilaritesDot_NamePath_preservation(self):
        self.backend_Sim_NamePath_Preservation(True)

    def test_featureSimilarites_NamePath_preservation(self):
        self.backend_Sim_NamePath_Preservation(False)


    ################### ####################
    # pointStatistics # #featureStatistics #
    ################### ####################

    def test_pointStatistics_max(self):
        """ Test pointStatistics returns correct max results """
        self.backend_Stat_max(True)

    def test_featureStatistics_max(self):
        """ Test featureStatistics returns correct max results """
        self.backend_Stat_max(False)

    def backend_Stat_max(self, axis):
        data = [[1, 2, 1], [-10, -1, -21], [-1, 0, 0]]
        dataT = numpy.array(data).T.tolist()
        fnames = _fnames(3)
        pnames = _pnames(3)
        orig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        trans = self.constructor(dataT, featureNames=pnames, pointNames=fnames)
        sameAsOrig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        sameAsOrigT = self.constructor(dataT, featureNames=pnames, pointNames=fnames)

        if axis:
            ret = orig.pointStatistics("MAx")

        else:
            ret = trans.featureStatistics("max ")
            ret.transpose()

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

    def test_featureStatistics_groupbyfeature(self):
        orig = self.constructor([[1,2,3,'f'], [4,5,6,'m'], [7,8,9,'f'], [10,11,12,'m']], featureNames=['a','b', 'c', 'gender'])
        if isinstance(orig, UML.data.BaseView):
            return
        #don't test view.
        res = orig.featureStatistics('mean', groupByFeature='gender')
        expObjf = self.constructor([4,5,6], featureNames=['a','b', 'c'], pointNames=['mean'])
        expObjm = self.constructor([7,8,9], featureNames=['a','b', 'c'], pointNames=['mean'])
        assert expObjf == res['f']
        assert expObjm == res['m']

    def backend_Stat_mean(self, axis):
        data = [[1, 1, 1], [0, 1, 1], [1, 0, 0]]
        dataT = numpy.array(data).T.tolist()
        fnames = _fnames(3)
        pnames = _pnames(3)
        orig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        trans = self.constructor(dataT, featureNames=pnames, pointNames=fnames)
        sameAsOrig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        sameAsOrigT = self.constructor(dataT, featureNames=pnames, pointNames=fnames)

        if axis:
            ret = orig.pointStatistics("Mean")
        else:
            ret = trans.featureStatistics(" MEAN")
            ret.transpose()

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

    def backend_Stat_median(self, axis):
        data = [[1, 1, 1], [0, 1, 1], [1, 0, 0]]
        dataT = numpy.array(data).T.tolist()
        fnames = _fnames(3)
        pnames = _pnames(3)
        orig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        trans = self.constructor(dataT, featureNames=pnames, pointNames=fnames)
        sameAsOrig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        sameAsOrigT = self.constructor(dataT, featureNames=pnames, pointNames=fnames)

        if axis:
            ret = orig.pointStatistics("MeDian")
        else:
            ret = trans.featureStatistics("median")
            ret.transpose()

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

    def backend_Stat_min(self, axis):
        data = [[1, 2, 1], [-10, -1, -21], [-1, 0, 0]]
        dataT = numpy.array(data).T.tolist()
        fnames = _fnames(3)
        pnames = _pnames(3)
        orig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        trans = self.constructor(dataT, featureNames=pnames, pointNames=fnames)
        sameAsOrig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        sameAsOrigT = self.constructor(dataT, featureNames=pnames, pointNames=fnames)

        if axis:
            ret = orig.pointStatistics("mIN")
        else:
            ret = trans.featureStatistics("min")
            ret.transpose()

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

    def backend_Stat_uniqueCount(self, axis):
        data = [[1, 1, 1], [0, 1, 1], [1, 0, -1]]
        dataT = numpy.array(data).T.tolist()
        fnames = _fnames(3)
        pnames = _pnames(3)
        orig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        trans = self.constructor(dataT, featureNames=pnames, pointNames=fnames)
        sameAsOrig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        sameAsOrigT = self.constructor(dataT, featureNames=pnames, pointNames=fnames)

        if axis:
            ret = orig.pointStatistics("unique Count")
        else:
            ret = trans.featureStatistics("UniqueCount")
            ret.transpose()

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

    def backend_Stat_proportionMissing(self, axis):
        data = [[1, None, 1], [0, 1, float('nan')], [1, float('nan'), None]]
        dataT = numpy.array(data).T.tolist()
        fnames = _fnames(3)
        pnames = _pnames(3)
        orig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        trans = self.constructor(dataT, featureNames=pnames, pointNames=fnames)
        sameAsOrig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        sameAsOrigT = self.constructor(dataT, featureNames=pnames, pointNames=fnames)

        if axis:
            ret = orig.pointStatistics("Proportion Missing ")
        else:
            ret = trans.featureStatistics("proportionmissing")
            ret.transpose()

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

    def backend_Stat_proportionZero(self, axis):
        data = [[1, 1, 1], [0, 1, 1], [1, 0, 0]]
        dataT = numpy.array(data).T.tolist()
        fnames = _fnames(3)
        pnames = _pnames(3)
        orig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        trans = self.constructor(dataT, featureNames=pnames, pointNames=fnames)
        sameAsOrig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        sameAsOrigT = self.constructor(dataT, featureNames=pnames, pointNames=fnames)

        if axis:
            ret = orig.pointStatistics("proportionZero")
        else:
            ret = trans.featureStatistics("proportion Zero")
            assert len(ret.points) == 1
            assert len(ret.features) == 3
            ret.transpose()

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

    def backend_Stat_sampleStandardDeviation(self, axis):
        data = [[1, 1, 1], [0, 1, 1], [1, 0, 0]]
        dataT = numpy.array(data).T.tolist()
        fnames = _fnames(3)
        pnames = _pnames(3)
        orig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        trans = self.constructor(dataT, featureNames=pnames, pointNames=fnames)
        sameAsOrig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        sameAsOrigT = self.constructor(dataT, featureNames=pnames, pointNames=fnames)

        if axis:
            ret = orig.pointStatistics("samplestd  ")
        else:
            ret = trans.featureStatistics("standard deviation")
            ret.transpose()

        assert len(ret.points) == 3
        assert len(ret.features) == 1

        npExpRaw = numpy.std(data, axis=1, ddof=1, keepdims=True)
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

    def backend_Stat_populationStandardDeviation(self, axis):
        data = [[1, 1, 1], [0, 1, 1], [1, 0, 0]]
        dataT = numpy.array(data).T.tolist()
        fnames = _fnames(3)
        pnames = _pnames(3)
        orig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        trans = self.constructor(dataT, featureNames=pnames, pointNames=fnames)
        sameAsOrig = self.constructor(data, featureNames=fnames, pointNames=pnames)
        sameAsOrigT = self.constructor(dataT, featureNames=pnames, pointNames=fnames)

        if axis:
            ret = orig.pointStatistics("popu  lationstd")
        else:
            ret = trans.featureStatistics("population standarddeviation")
            ret.transpose()

        assert len(ret.points) == 3
        assert len(ret.features) == 1

        npExpRaw = numpy.std(data, axis=1, ddof=0, keepdims=True)
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

    @raises(ArgumentException)
    def test_pointStatistics_unexpectedString(self):
        """ Test pointStatistics returns correct std results """
        self.backend_Stat_unexpectedString(True)

    @raises(ArgumentException)
    def test_featureStatistics_unexpectedString(self):
        """ Test featureStatistics returns correct std results """
        self.backend_Stat_unexpectedString(False)

    def backend_Stat_unexpectedString(self, axis):
        data = [[1, 1, 1], [0, 1, 1], [1, 0, 0]]
        orig = self.constructor(data)
        sameAsOrig = self.constructor(data)

        if axis:
            ret = orig.pointStatistics("hello")
        else:
            ret = orig.featureStatistics("meanie")

    def backend_Stat_NamePath_preservation(self, axis):
        data = [[1, 2, 1], [-10, -1, -21], [-1, 0, 0]]
        orig = self.constructor(data, name=preserveName, path=preservePair)

        accepted = [
            'max', 'mean', 'median', 'min', 'uniquecount', 'proportionmissing',
            'proportionzero', 'standarddeviation', 'std', 'populationstd',
            'populationstandarddeviation', 'samplestd',
            'samplestandarddeviation'
        ]

        for curr in accepted:
            if axis:
                ret = orig.pointStatistics(curr)
            else:
                ret = orig.featureStatistics(curr)

            assert orig.name == preserveName
            assert orig.absolutePath == preserveAPath
            assert orig.relativePath == preserveRPath

            assert ret.nameIsDefault()
            assert ret.absolutePath == preserveAPath
            assert ret.relativePath == preserveRPath

    def test_pointStatistics_NamePath_preservations(self):
        self.backend_Stat_NamePath_preservation(True)

    def test_featureStatistics_NamePath_preservations(self):
        self.backend_Stat_NamePath_preservation(False)


    ########
    # plot #
    ########

    @attr('slow')
    def test_plot_fileOutput(self):
        with tempfile.NamedTemporaryFile(suffix='png') as outFile:
            path = outFile.name
            startSize = os.path.getsize(path)
            assert startSize == 0

            randGenerated = UML.createRandomData("List", 10, 10, 0)
            raw = randGenerated.copyAs('pythonlist')
            obj = self.constructor(raw)
            #we call the leading underscore version, because it
            # returns the process
            p = obj._plot(outPath=path)
            p.join()

            endSize = os.path.getsize(path)
            assert startSize < endSize

    ###########################
    # plotFeatureDistribution #
    ###########################

    @attr('slow')
    def test_plotFeatureDistribution_fileOutput(self):
        with tempfile.NamedTemporaryFile(suffix='png') as outFile:
            path = outFile.name
            startSize = os.path.getsize(path)
            assert startSize == 0

            randGenerated = UML.createRandomData("List", 10, 10, 0)
            raw = randGenerated.copyAs('pythonlist')
            obj = self.constructor(raw)
            #we call the leading underscore version, because it
            # returns the process
            p = obj._plotFeatureDistribution(feature=0, outPath=path)
            p.join()

            endSize = os.path.getsize(path)
            assert startSize < endSize


    #############################
    # plotFeatureAgainstFeature #
    #############################

    @attr('slow')
    def test_plotFeatureAgainstFeature_fileOutput(self):
        with tempfile.NamedTemporaryFile(suffix='png') as outFile:
            path = outFile.name
            startSize = os.path.getsize(path)
            assert startSize == 0

            randGenerated = UML.createRandomData("List", 10, 10, 0)
            raw = randGenerated.copyAs('pythonlist')
            obj = self.constructor(raw)
            #we call the leading underscore version, because it
            # returns the process
            p = obj._plotFeatureAgainstFeature(x=0, y=1, outPath=path)
            p.join()

            endSize = os.path.getsize(path)
            assert startSize < endSize

    ###################
    # points.__iter__ #
    ###################

    def test_points_iter_FemptyCorrectness(self):
        data = [[], []]
        data = numpy.array(data)
        toTest = self.constructor(data)
        pIter = toTest.points

        pView = next(pIter)
        assert len(pView) == 0
        pView = next(pIter)
        assert len(pView) == 0

        try:
            next(pIter)
            assert False  # expected StopIteration from prev statement
        except StopIteration:
            pass

    def test_points_iter_noNextPempty(self):
        """ test .points() has no next value when object is point empty """
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)
        viewIter = toTest.points
        try:
            next(viewIter)
        except StopIteration:
            return
        assert False

    def test_points_iter_exactValueViaFor(self):
        """ Test .points() gives views that contain exactly the correct data """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)

        viewIter = toTest.points

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

        viewIter = toTest.points
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

    def test_features_iter_PemptyCorrectness(self):
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)
        fIter = toTest.features

        fView = next(fIter)
        assert len(fView) == 0
        fView = next(fIter)
        assert len(fView) == 0

        try:
            next(fIter)
            assert False  # expected StopIteration from prev statement
        except StopIteration:
            pass

    def test_features_iter_noNextFempty(self):
        """ test .features() has no next value when object is feature empty """
        data = [[], []]
        data = numpy.array(data)
        toTest = self.constructor(data)
        viewIter = toTest.features
        try:
            next(viewIter)
        except StopIteration:
            return
        assert False


    def test_features_iter_exactValueViaFor(self):
        """ Test .features() gives views that contain exactly the correct data """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)

        viewIter = toTest.features

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
        """ Test .features() works when there are all zero points """
        data = [[0, 1, 0, 2, 0, 3, 0, 0], [0, 4, 0, 5, 0, 6, 0, 0], [0, 7, 0, 8, 0, 9, 0, 0]]
        toTest = self.constructor(data)

        viewIter = toTest.features
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

    #####################################################
    # points.nonZeroIterator / features.nonZeroIterator #
    #####################################################

    def test_points_nonZeroIterator_handmade(self):
        data = [[0, 1, 2], [0, 4, 0], [0, 0, 5], [0, 0, 0]]
        obj = self.constructor(data)

        ret = []
        for val in obj.points.nonZeroIterator():
            ret.append(val)

        assert ret == [1, 2, 4, 5]

    def test_points_nonZeroIterator_empty(self):
        data = []
        obj = self.constructor(data)

        ret = []
        for val in obj.points.nonZeroIterator():
            ret.append(val)

        assert ret == []

    def test_features_nonZeroIterator_handmade(self):
        data = [[0, 1, 2], [0, 4, 0], [0, 0, 5], [0, 0, 0]]
        obj = self.constructor(data)

        ret = []
        for val in obj.features.nonZeroIterator():
            ret.append(val)

        assert ret == [1, 4, 2, 5]

    def test_features_nonZeroIterator_empty(self):
        data = []
        obj = self.constructor(data)

        ret = []
        for val in obj.features.nonZeroIterator():
            ret.append(val)

        assert ret == []

###########
# Helpers #
###########

def checkToStringRet(ret, data, includeNames):
    cHold = '--'
    rHold = '|'
    pnameSep = '   '
    colSep = ' '
    sigDigits = 3
    rows = ret.split('\n')
    rows = rows[:(len(rows) - 1)]

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
                assert fromIndexFname == fnames[cDataIndex]
