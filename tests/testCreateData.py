import tempfile
import numpy
import os
import sys
import copy
import itertools
try:
    from unittest import mock #python >=3.3
except:
    import mock

from nose.tools import *
from nose.plugins.attrib import attr

import nimble
from nimble.exceptions import InvalidArgumentValue, InvalidArgumentType
from nimble.exceptions import FileFormatException
from nimble.data.dataHelpers import DEFAULT_PREFIX
from nimble.helpers import _intFloatOrString
# from .. import logger
from .assertionHelpers import oneLogEntryExpected

scipy = nimble.importModule('scipy.sparse')
pd = nimble.importModule('pandas')

returnTypes = copy.copy(nimble.data.available)
returnTypes.append(None)

class NoIter(object):
    def __init__(self, vals):
        self.vals = vals

    def __len__(self):
        return len(self.vals)

class IterNext(object):
    def __init__(self, vals):
        self.vals = vals
        self.pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos < len(self.vals):
            self.pos += 1
            return self.vals[self.pos - 1]
        else:
            raise StopIteration()

class GetItemOnly(object):
    def __init__(self, vals):
        self.vals = vals

    def __getitem__(self, key):
        return self.vals[key]

###########################
# Data values correctness #
###########################

def test_createData_raw_stringConversion():
    """
    """
    for t in returnTypes:
        values = []
        toTest = nimble.createData(t, [['1','2','3'], ['4','5','6'], ['7','8','9']])
        for i in range(len(toTest.points)):
            for j in range(len(toTest.features)):
                values.append(toTest[i,j])
        assert all(isinstance(val, float) for val in values)

def test_createData_raw_noStringConversion():
    """
    """
    for t in returnTypes:
        values = []
        toTest = nimble.createData(t, [['1','2','3'], ['4','5','6'], ['7','8','9']], elementType=object)
        for i in range(len(toTest.points)):
            for j in range(len(toTest.features)):
                values.append(toTest[i,j])
        assert all(isinstance(val, str) for val in values)

def test_createData_raw_invalidPointOrFeatureNames():
    for t in returnTypes:
        try:
            pNames = NoIter(['1', '4'])
            toTest = nimble.createData(t, [[1,2,3], [4,5,6]], pointNames=pNames)
            assert False # expected InvalidArgumentType
        except InvalidArgumentType:
            pass

        try:
            fNames = NoIter(['a', 'b', 'c'])
            toTest = nimble.createData(t, [[1,2,3], [4,5,6]], featureNames=fNames)
            assert False # expected InvalidArgumentType
        except InvalidArgumentType:
            pass

def test_createData_raw_pointAndFeatureIterators():
    for t in returnTypes:
        pNames = IterNext(['1', '4'])
        fNames = IterNext(['a', 'b', 'c'])
        toTest1 = nimble.createData(t, [[1,2,3], [4,5,6]], pointNames=pNames,
                                    featureNames=fNames)
        assert toTest1.points.getNames() == ['1', '4']
        assert toTest1.features.getNames() == ['a', 'b', 'c']

        pNames = GetItemOnly(['1', '4'])
        fNames = GetItemOnly(['a', 'b', 'c'])
        toTest2 = nimble.createData(t, [[1,2,3], [4,5,6]], pointNames=pNames,
                                    featureNames=fNames)
        assert toTest2.points.getNames() == ['1', '4']
        assert toTest2.features.getNames() == ['a', 'b', 'c']


def test_createData_CSV_data():
    """ Test of createData() loading a csv file, default params """
    for t in returnTypes:
        fromList = nimble.createData(returnType=t, data=[[1, 2, 3]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.flush()
            objName = 'fromCSV'
            fromCSV = nimble.createData(returnType=t, data=tmpCSV.name, name=objName)

            assert fromList == fromCSV


def test_createData_CSV_dataRandomExtension():
    """ Test of createData() loading a csv file without csv extension """
    for t in returnTypes:
        fromList = nimble.createData(returnType=t, data=[[1, 2, 3]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".foo", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.flush()
            objName = 'fromCSV'
            fromCSV = nimble.createData(returnType=t, data=tmpCSV.name, name=objName)

            assert fromList == fromCSV


def test_createData_CSV_data_noComment():
    for t in returnTypes:
        fromList = nimble.createData(returnType=t, data=[[1, 2], [1, 2]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,#3\n")
            tmpCSV.write("1,2,3\n")
            tmpCSV.flush()
            objName = 'fromCSV'
            fromCSV = nimble.createData(returnType=t, data=tmpCSV.name, name=objName, ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV


def test_createData_CSV_data_ListOnly():
    fromList = nimble.createData(returnType="List", data=[[1, 2, 'three'], [4, 5, 'six']])

    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,three\n")
        tmpCSV.write("4,5,six\n")
        tmpCSV.flush()
        objName = 'fromCSV'
        fromCSV = nimble.createData(returnType="List", data=tmpCSV.name, name=objName)

        assert fromList == fromCSV


def test_createData_CSV_data_ListOnly_noComment():
    fromList = nimble.createData(returnType="List", data=[[1, 2, 'three'], [4, 5, '#six']])

    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,three\n")
        tmpCSV.write("4,5,#six\n")
        tmpCSV.flush()
        objName = 'fromCSV'
        fromCSV = nimble.createData(returnType="List", data=tmpCSV.name, name=objName)

        assert fromList == fromCSV


def test_createData_MTXArr_data():
    """ Test of createData() loading a mtx (arr format) file, default params """
    for t in returnTypes:
        fromList = nimble.createData(returnType=t, data=[[1, 2, 3]])

        # instantiate from mtx array file
        with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXArr:
            tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
            tmpMTXArr.write("1 3\n")
            tmpMTXArr.write("1\n")
            tmpMTXArr.write("2\n")
            tmpMTXArr.write("3\n")
            tmpMTXArr.flush()
            objName = 'fromMTXArr'
            fromMTXArr = nimble.createData(returnType=t, data=tmpMTXArr.name, name=objName)

            if t is None and fromList.getTypeString() != fromMTXArr.getTypeString():
                assert fromList.isApproximatelyEqual(fromMTXArr)
            else:
                assert fromList == fromMTXArr

def test_createData_MTXArr_dataRandomExtension():
    """ Test of createData() loading a mtx (arr format) file without mtx extension """
    for t in returnTypes:
        fromList = nimble.createData(returnType=t, data=[[1, 2, 3]])

        # instantiate from mtx array file
        with tempfile.NamedTemporaryFile(suffix=".foo", mode='w') as tmpMTXArr:
            tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
            tmpMTXArr.write("1 3\n")
            tmpMTXArr.write("1\n")
            tmpMTXArr.write("2\n")
            tmpMTXArr.write("3\n")
            tmpMTXArr.flush()
            objName = 'fromMTXArr'
            fromMTXArr = nimble.createData(returnType=t, data=tmpMTXArr.name, name=objName)

            if t is None and fromList.getTypeString() != fromMTXArr.getTypeString():
                assert fromList.isApproximatelyEqual(fromMTXArr)
            else:
                assert fromList == fromMTXArr


def test_createData_MTXCoo_data():
    """ Test of createData() loading a mtx (coo format) file, default params """
    for t in returnTypes:
        fromList = nimble.createData(returnType=t, data=[[1, 2, 3]])

        # instantiate from mtx coordinate file
        with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXCoo:
            tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.write("1 1 1\n")
            tmpMTXCoo.write("1 2 2\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.flush()
            objName = 'fromMTXCoo'
            fromMTXCoo = nimble.createData(returnType=t, data=tmpMTXCoo.name, name=objName)

            if t is None and fromList.getTypeString() != fromMTXCoo.getTypeString():
                assert fromList.isApproximatelyEqual(fromMTXCoo)
            else:
                assert fromList == fromMTXCoo

def test_createData_MTXCoo_dataRandomExtension():
    """ Test of createData() loading a mtx (coo format) file without mtx extension """
    for t in returnTypes:
        fromList = nimble.createData(returnType=t, data=[[1, 2, 3]])

        # instantiate from mtx coordinate file
        with tempfile.NamedTemporaryFile(suffix=".foo", mode='w') as tmpMTXCoo:
            tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.write("1 1 1\n")
            tmpMTXCoo.write("1 2 2\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.flush()
            objName = 'fromMTXCoo'
            fromMTXCoo = nimble.createData(returnType=t, data=tmpMTXCoo.name, name=objName)

            if t is None and fromList.getTypeString() != fromMTXCoo.getTypeString():
                assert fromList.isApproximatelyEqual(fromMTXCoo)
            else:
                assert fromList == fromMTXCoo


@raises(FileFormatException)
def test_createData_CSV_unequalRowLength_short():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write('1,2,3,4\n')
        tmpCSV.write('4,5,6\n')
        tmpCSV.flush()

        nimble.createData(returnType="List", data=tmpCSV.name)

@raises(FileFormatException)
def test_createData_CSV_unequalRowLength_long():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("4,5,6,7\n")
        tmpCSV.flush()

        nimble.createData(returnType="List", data=tmpCSV.name)


@raises(FileFormatException)
def test_createData_CSV_unequalRowLength_definedByNames():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("one,two,three\n")
        tmpCSV.write("11,22,33,44\n")
        tmpCSV.write("4,5,6,7\n")
        tmpCSV.flush()

        nimble.createData(returnType="List", data=tmpCSV.name, featureNames=True)


def test_createData_CSV_unequalRowLength_position():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("#ignore\n")
        tmpCSV.write("1,2,3,4,0,0,0,0\n")
        tmpCSV.write("\n")
        tmpCSV.write("11,22,33,44,0,0,0,0\n")
        tmpCSV.write("4,5,6,0,0,0\n")
        tmpCSV.flush()

        try:
            nimble.createData(returnType="List", data=tmpCSV.name, featureNames=True)
            assert False  # the previous call should have raised an exception
        except FileFormatException as ffe:
            print(ffe.value)
            # We expect a message of the format:
            #
            assert '1' in ffe.value  # defining line
            assert '4' in ffe.value  # offending line
            # offending line number comes before defining line number
            assert ffe.value.index('4') < ffe.value.index('1')

            assert '8' in ffe.value  # expected length
            assert '6' in ffe.value  # offending length
            # offending length comes before expected length
            assert ffe.value.index('6') < ffe.value.index('8')


############################
# Name and path attributes #
############################


def test_createData_objName_and_path_CSV():
    for t in returnTypes:
        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.flush()

            objName = 'fromCSV'
            ret = nimble.createData(returnType=t, data=tmpCSV.name, name=objName)
            assert ret.name == objName
            assert ret.path == tmpCSV.name
            assert ret.absolutePath == tmpCSV.name

            relExp = os.path.relpath(ret.absolutePath)
            assert ret.relativePath == relExp

            retDefName = nimble.createData(returnType=t, data=tmpCSV.name)
            tokens = tmpCSV.name.rsplit(os.path.sep)
            assert retDefName.name == tokens[len(tokens) - 1]


def test_createData_objName_and_path_MTXArr():
    for t in returnTypes:
        # instantiate from mtx array file
        with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXArr:
            tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
            tmpMTXArr.write("1 3\n")
            tmpMTXArr.write("1\n")
            tmpMTXArr.write("2\n")
            tmpMTXArr.write("3\n")
            tmpMTXArr.flush()

            objName = 'fromMTXArr'
            ret = nimble.createData(returnType=t, data=tmpMTXArr.name, name=objName)
            assert ret.name == objName
            assert ret.path == tmpMTXArr.name
            assert ret.absolutePath == tmpMTXArr.name

            relExp = os.path.relpath(ret.absolutePath)
            assert ret.relativePath == relExp

            retDefName = nimble.createData(returnType=t, data=tmpMTXArr.name)
            tokens = tmpMTXArr.name.rsplit(os.path.sep)
            assert retDefName.name == tokens[len(tokens) - 1]


def test_createData_objName_and_path_MTXCoo():
    for t in returnTypes:
        # instantiate from mtx coordinate file
        with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXCoo:
            tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.write("1 1 1\n")
            tmpMTXCoo.write("1 2 2\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.flush()

            objName = 'fromMTXCoo'
            ret = nimble.createData(returnType=t, data=tmpMTXCoo.name, name=objName)
            assert ret.name == objName
            assert ret.path == tmpMTXCoo.name
            assert ret.absolutePath == tmpMTXCoo.name

            relExp = os.path.relpath(ret.absolutePath)
            assert ret.relativePath == relExp

            retDefName = nimble.createData(returnType=t, data=tmpMTXCoo.name)
            tokens = tmpMTXCoo.name.rsplit(os.path.sep)
            assert retDefName.name == tokens[len(tokens) - 1]


###################################
# Point / Feature names from File #
###################################

def test_extractNames_CSV():
    """ Test of createData() loading a csv file and extracting names """
    pNames = ['pn1']
    fNames = ['one', 'two', 'three']
    for t in returnTypes:
        fromList = nimble.createData(
            returnType=t, data=[[1, 2, 3]], pointNames=pNames, featureNames=fNames)

        # instantiate from csv file
        tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv", mode='w')
        tmpCSV.write('ignore,one,two,three\n')
        tmpCSV.write("pn1,1,2,3\n")
        tmpCSV.flush()

        fromCSV = nimble.createData(
            returnType=t, data=tmpCSV.name, pointNames=True, featureNames=True)
        tmpCSV.close()
        assert fromList == fromCSV


def test_names_AutoDetectedBlankLines_CSV():
    pNames = ['pn1']
    fNames = ['one', 'two', 'three']
    for t in returnTypes:
        fromList = nimble.createData(
            returnType=t, data=[[1, 2, 3]], pointNames=pNames, featureNames=fNames)

        # instantiate from csv file
        tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv", mode='w')
        tmpCSV.write("\n")
        tmpCSV.write("\n")
        tmpCSV.write("pointNames,one,two,three\n")
        tmpCSV.write("pn1,1,2,3\n")
        tmpCSV.flush()

        fromCSV = nimble.createData(returnType=t, data=tmpCSV.name)
        tmpCSV.close()
        assert fromList == fromCSV


def test_featNamesOnly_AutoDetectedBlankLines_CSV():
    fNames = ['one', 'two', 'three']
    for t in returnTypes:
        fromList = nimble.createData(returnType=t, data=[[1, 2, 3]], featureNames=fNames)

        # instantiate from csv file
        tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv", mode='w')
        tmpCSV.write("\n")
        tmpCSV.write("\n")
        tmpCSV.write("one,two,three\n")
        tmpCSV.write("1,2,3\n")
        tmpCSV.flush()

        fromCSV = nimble.createData(returnType=t, data=tmpCSV.name)
        tmpCSV.close()
        assert fromList == fromCSV

def test_pointNames_AutoDetected_from_specified_featNames_CSV():
    fNames = ['one', 'two', 'three']
    pNames = ['pn1']
    for t in returnTypes:
        fromList = nimble.createData(
            returnType=t, data=[[1, 2, 3]], pointNames=pNames, featureNames=fNames)

        # instantiate from csv file
        tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv", mode='w')
        tmpCSV.write("\n")
        tmpCSV.write("\n")
        tmpCSV.write("pointNames,one,two,three\n")
        tmpCSV.write("pn1,1,2,3\n")
        tmpCSV.flush()
        fromCSV = nimble.createData(returnType=t, data=tmpCSV.name, featureNames=True)
        tmpCSV.close()
        assert fromList == fromCSV


def test_specifiedIgnore_overides_autoDetectBlankLine_CSV():
    for t in returnTypes:
        data = [[0, 1, 2, 3], [10, 11, 12, 13]]
        fromList = nimble.createData(returnType=t, data=data)

        # instantiate from csv file
        tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv", mode='w')
        tmpCSV.write("\n")
        tmpCSV.write("\n")
        tmpCSV.write('0,1,2,3\n')
        tmpCSV.write("10,11,12,13\n")
        tmpCSV.flush()
        fromCSV = nimble.createData(
            returnType=t, data=tmpCSV.name, pointNames=False, featureNames=False)
        tmpCSV.close()
        assert fromList == fromCSV


def helper_auto(rawStr, rawType, returnType, pointNames, featureNames):
    """
    Writes a CSV and reads it in using nimble.createData, fixing in place the given arguments,
    returning the resultant object

    """
    if rawType == 'csv':
        tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv", mode='w')
        tmpCSV.write(rawStr)
        tmpCSV.flush()
        ret = nimble.createData(returnType=returnType, data=tmpCSV.name,
                                pointNames=pointNames, featureNames=featureNames)
        tmpCSV.close()
    else:
        fnameRow = list(map(_intFloatOrString, rawStr.split('\n')[0].split(',')))
        dataRow = list(map(_intFloatOrString, rawStr.split('\n')[1].split(',')))
        lolFromRaw = [fnameRow, dataRow]
        baseObj = nimble.createData("List", lolFromRaw, pointNames=False, featureNames=False)
        if rawType == 'scipycoo':
            npRaw = numpy.array(lolFromRaw, dtype=object)
            finalRaw = scipy.sparse.coo_matrix(npRaw)
        else:
            finalRaw = baseObj.copy(to=rawType)

        ret = nimble.createData(returnType=returnType, data=finalRaw,
                                pointNames=pointNames, featureNames=featureNames)

    return ret

def test_automaticByType_fnames_rawAndCSV():
    availableRaw = ['csv', 'pythonlist', 'numpyarray', 'numpymatrix', 'scipycoo']
    for (rawT, retT) in itertools.product(availableRaw, returnTypes):
#        rawT = 'scipycoo'
#        retT = 'List'
#        print rawT + " " + str(retT)
#        import pdb
#        pdb.set_trace()

        # example which triggers automatic removal
        correctRaw = "fname0,fname1,fname2\n1,2,3\n"
        correct = helper_auto(correctRaw, rawT, retT, pointNames='automatic', featureNames='automatic')
        assert correct.features.getNames() == ['fname0','fname1','fname2']

        # example where first line contains a non-string interpretable value
        nonStringFail1Raw = "fname0,1.0,fname2\n1,2,3"
        fail1 = helper_auto(nonStringFail1Raw, rawT, retT, pointNames='automatic', featureNames='automatic')
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), fail1.features.getNames()))

        # example where the first line contains all strings, but second line also contains strings
        sameTypeFail2Raw = "fname0,fname1,fname2\n1,data2,3"
        fail2 = helper_auto(sameTypeFail2Raw, rawT, retT, pointNames='automatic', featureNames='automatic')
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), fail2.features.getNames()))


def test_userOverrideOfAutomaticByType_fnames_rawAndCSV():
    availableRaw = ['csv', 'pythonlist', 'numpyarray', 'numpymatrix', 'scipycoo']
    for (rawT, retT) in itertools.product(availableRaw, returnTypes):
        # example where user provided False overides automatic detection
        correctRaw = "fname0,fname1,fname2\n1,2,3\n"
        overide1a = helper_auto(correctRaw, rawT, retT, pointNames='automatic', featureNames=False)
        overide1b = helper_auto(correctRaw, rawT, retT, pointNames='automatic', featureNames=None)
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), overide1a.features.getNames()))
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), overide1b.features.getNames()))

        # example where user provided True extracts non-detectable first line
        nonStringFail1Raw = "fname0,1.0,fname2\n1,2,3"
        overide2 = helper_auto(nonStringFail1Raw, rawT, retT, pointNames='automatic', featureNames=True)
        assert overide2.features.getNames() == ['fname0', '1.0', 'fname2']

        # example where user provided True extracts non-detectable first line
        sameTypeFail2Raw = "fname0,fname1,fname2\ndata1,data2,data3"
        overide3 = helper_auto(sameTypeFail2Raw, rawT, retT, pointNames='automatic', featureNames=True)
        assert overide3.features.getNames() == ['fname0', 'fname1', 'fname2']


def test_automaticByType_pname_interaction_with_fname():
    availableRaw = ['csv', 'pythonlist', 'numpyarray', 'numpymatrix', 'scipycoo']
    for (rawT, retT) in itertools.product(availableRaw, returnTypes):
#        rawT = 'scipycoo'
#        retT = None
#        print rawT + " " + str(retT)
#        import pdb
#        pdb.set_trace()

        # pnames auto triggered with auto fnames
        raw = "pointNames,fname0,fname1,fname2\npname0,1,2,3\n"
        testObj = helper_auto(raw, rawT, retT, pointNames='automatic', featureNames='automatic')
        assert testObj.features.getNames() == ['fname0','fname1','fname2']
        assert testObj.points.getNames() == ['pname0']

        # pnames auto triggereed with explicit fnames
        raw = "pointNames,fname0,fname1,fname2\npname0,1,2,3\n"
        testObj = helper_auto(raw, rawT, retT, pointNames='automatic', featureNames=True)
        assert testObj.features.getNames() == ['fname0','fname1','fname2']
        assert testObj.points.getNames() == ['pname0']

        #pnames not triggered given 'pointNames' at [0,0] when fnames auto trigger fails CASE1
        raw = "pointNames,fname0,1.0,fname2\npname0,1,2,3\n"
        testObj = helper_auto(raw, rawT, retT, pointNames='automatic', featureNames='automatic')
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), testObj.features.getNames()))
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), testObj.points.getNames()))

        #pnames not triggered given 'pointNames' at [0,0] when fnames auto trigger fails CASE2
        raw = "pointNames,fname0,fname1,fname2\npname0,data1,data2,data3\n"
        testObj = helper_auto(raw, rawT, retT, pointNames='automatic', featureNames='automatic')
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), testObj.features.getNames()))
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), testObj.points.getNames()))

        #pnames not triggered given 'pointNames' at [0,0] when fnames explicit False
        raw = "pointNames,fname0,fname1,fname2\npname0,1,2,3\n"
        testObj = helper_auto(raw, rawT, retT, pointNames='automatic', featureNames=False)
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), testObj.features.getNames()))
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), testObj.points.getNames()))

        #pnames explicit False given 'pointNames' at [0,0] and fname auto extraction
        raw = "pointNames,fname0,fname1,fname2\npname0,1,2,3\n"
        testObj = helper_auto(raw, rawT, retT, pointNames=False, featureNames=True)
        assert testObj.features.getNames() == ['pointNames', 'fname0', 'fname1', 'fname2']
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), testObj.points.getNames()))


def test_names_AutomaticVsTrueVsFalseVsNone():
    """ Test createData() accepted inputs for pointNames and featureNames """
    for t in returnTypes:
        # pNames and fNames triggered for automatic
        raw1 = [['pointNames', 'fname0','fname1','fname2'],
                ['pname0', 0, 1, 2]]
        testAuto = nimble.createData(t, raw1, pointNames='automatic', featureNames='automatic')
        testTrue = nimble.createData(t, raw1, pointNames=True, featureNames=True)
        testFalse = nimble.createData(t, raw1, pointNames=False, featureNames=False)
        testNone = nimble.createData(t, raw1, pointNames=None, featureNames=None)

        assert testAuto == testTrue
        assert testAuto != testFalse
        assert testFalse == testNone

        # pNames not triggered, fNames triggered for automatic
        raw2 = [['either', 'fname0','fname1','fname2'],
                [99, 0, 1, 2]]
        testAuto = nimble.createData(t, raw2, pointNames='automatic', featureNames='automatic')
        testTrue = nimble.createData(t, raw2, pointNames=True, featureNames=True)
        testFalse = nimble.createData(t, raw2, pointNames=False, featureNames=False)
        testNone = nimble.createData(t, raw2, pointNames=None, featureNames=None)

        assert testAuto != testTrue
        assert testAuto != testFalse
        assert testTrue != testFalse
        assert testFalse == testNone

        # no names triggered for automatic
        raw3 = [[-1, 9, 8, 7],
                [99, 0, 1, 2]]
        testAuto = nimble.createData(t, raw3, pointNames='automatic', featureNames='automatic')
        testTrue = nimble.createData(t, raw3, pointNames=True, featureNames=True)
        testFalse = nimble.createData(t, raw3, pointNames=False, featureNames=False)
        testNone = nimble.createData(t, raw3, pointNames=None, featureNames=None)

        assert testAuto != testTrue
        assert testAuto == testFalse
        assert testFalse == testNone


def test_namesInComment_MTXArr():
    """ Test of createData() loading a mtx (arr format) file and comment Names """
    pNames = ['pn1']
    fNames = ['one', 'two', 'three']
    for t in returnTypes:
        fromList = nimble.createData(returnType=t, data=[[1, 2, 3]], pointNames=pNames, featureNames=fNames)

        # instantiate from mtx array file
        tmpMTXArr = tempfile.NamedTemporaryFile(suffix=".mtx", mode='w')
        tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
        tmpMTXArr.write("%#pn1\n")
        tmpMTXArr.write("%#one,two,three\n")
        tmpMTXArr.write("1 3\n")
        tmpMTXArr.write("1\n")
        tmpMTXArr.write("2\n")
        tmpMTXArr.write("3\n")
        tmpMTXArr.flush()
        fromMTXArr = nimble.createData(returnType=t, data=tmpMTXArr.name)
        tmpMTXArr.close()
        if t is None and fromList.getTypeString() != fromMTXArr.getTypeString():
            assert fromList.isApproximatelyEqual(fromMTXArr)
        else:
            assert fromList == fromMTXArr


def test_namesInComment_MTXCoo():
    """ Test of createData() loading a mtx (coo format) file and comment Names """
    pNames = ['pn1']
    fNames = ['one', 'two', 'three']
    for t in returnTypes:
        fromList = nimble.createData(
            returnType=t, data=[[1, 2, 3]], pointNames=pNames, featureNames=fNames)

        # instantiate from mtx coordinate file
        tmpMTXCoo = tempfile.NamedTemporaryFile(suffix=".mtx", mode='w')
        tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
        tmpMTXCoo.write("%#pn1\n")
        tmpMTXCoo.write("%#one,two,three\n")
        tmpMTXCoo.write("1 3 3\n")
        tmpMTXCoo.write("1 1 1\n")
        tmpMTXCoo.write("1 2 2\n")
        tmpMTXCoo.write("1 3 3\n")
        tmpMTXCoo.flush()
        fromMTXCoo = nimble.createData(returnType=t, data=tmpMTXCoo.name)
        tmpMTXCoo.close()
        if t is None and fromList.getTypeString() != fromMTXCoo.getTypeString():
            assert fromList.isApproximatelyEqual(fromMTXCoo)
        else:
            assert fromList == fromMTXCoo


def test_extractNames_MTXArr():
    """ Test of createData() loading a mtx (arr format) file and extracting names """
    pNames = ['11']
    fNames = ['1', '2', '3']
    for t in returnTypes:
        fromList = nimble.createData(
            returnType=t, data=[[21, 22, 23]], pointNames=pNames, featureNames=fNames)

        # instantiate from mtx array file
        tmpMTXArr = tempfile.NamedTemporaryFile(suffix=".mtx", mode='w')
        tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
        tmpMTXArr.write("2 4\n")
        tmpMTXArr.write("-4\n")
        tmpMTXArr.write("11\n")
        tmpMTXArr.write("1\n")
        tmpMTXArr.write("21\n")
        tmpMTXArr.write("2\n")
        tmpMTXArr.write("22\n")
        tmpMTXArr.write("3\n")
        tmpMTXArr.write("23\n")
        tmpMTXArr.flush()

        fromMTXArr = nimble.createData(
            returnType=t, data=tmpMTXArr.name, pointNames=True, featureNames=True)
        tmpMTXArr.close()
        if t is None and fromList.getTypeString() != fromMTXArr.getTypeString():
            assert fromList.isApproximatelyEqual(fromMTXArr)
        else:
            assert fromList == fromMTXArr


def test_extractNames_MTXCoo():
    """ Test of createData() loading a mtx (coo format) file and extracting names """
    pNames = ['21']
    fNames = ['1', '2', '3']
    for t in returnTypes:
        fromList = nimble.createData(
            returnType=t, data=[[22, -5, 23]], pointNames=pNames, featureNames=fNames)

        # instantiate from mtx coordinate file
        tmpMTXCoo = tempfile.NamedTemporaryFile(suffix=".mtx", mode='w')
        tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
        tmpMTXCoo.write("2 4 8\n")
        tmpMTXCoo.write("1 1 11\n")
        tmpMTXCoo.write("1 2 1\n")
        tmpMTXCoo.write("1 3 2\n")
        tmpMTXCoo.write("1 4 3\n")
        tmpMTXCoo.write("2 1 21\n")
        tmpMTXCoo.write("2 2 22\n")
        tmpMTXCoo.write("2 3 -5\n")
        tmpMTXCoo.write("2 4 23\n")
        tmpMTXCoo.flush()
        fromMTXCoo = nimble.createData(
            returnType=t, data=tmpMTXCoo.name, pointNames=True, featureNames=True)
        tmpMTXCoo.close()
        if t is None and fromList.getTypeString() != fromMTXCoo.getTypeString():
            assert fromList.isApproximatelyEqual(fromMTXCoo)
        else:
            assert fromList == fromMTXCoo


@raises(InvalidArgumentValue)
def test_csv_extractNames_duplicatePointName():
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write('ignore,one,two,three\n')
        tmpCSV.write("pn1,1,2,3\n")
        tmpCSV.write("pn1,11,22,33\n")
        tmpCSV.flush()

        nimble.createData(returnType="List", data=tmpCSV.name, pointNames=True)


@raises(InvalidArgumentValue)
def test_csv_extractNames_duplicateFeatureName():
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write('one,two,one\n')
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.flush()

        nimble.createData(returnType="List", data=tmpCSV.name, featureNames=True)


def test_csv_roundtrip_autonames():
    for retType in returnTypes:
        data = [[1, 0, 5, 12], [0, 1, 3, 17], [0, 0, 8, 22]]
        pnames = ['p0','p1','p2']
        fnames = ['f0','f1','f2', 'f3']

        withFnames = nimble.createData(retType, data, featureNames=fnames)
        withBoth = nimble.createData(retType, data, featureNames=fnames, pointNames=pnames)

        with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSVFnames:
            withFnames.writeFile(tmpCSVFnames.name, 'csv', includeNames=True)
            fromFileFnames = nimble.createData(returnType=retType, data=tmpCSVFnames.name)
            assert fromFileFnames == withFnames

        with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSVBoth:
            withBoth.writeFile(tmpCSVBoth.name, 'csv', includeNames=True)
            fromFileBoth = nimble.createData(returnType=retType, data=tmpCSVBoth.name)
            assert fromFileBoth == withBoth


##################################
# Point / Feature names from Raw #
##################################


def test_extractNames_pythonList():
    """ Test of createData() given python list, extracting names """
    pNames = ['pn1']
    fNames = ['one', '2', 'three']

    for t in returnTypes:
        inDataRaw = [['foo', 'one', 2, 'three'], ['pn1', 1, -1, -3]]
        specRaw = [[1, -1, -3]]

        inData = nimble.createData(
            returnType=t, data=inDataRaw, pointNames=True, featureNames=True)
        specified = nimble.createData(
            returnType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
        assert inData == specified


def test_extractNames_NPArray():
    """ Test of createData() given numpy array, extracting names """
    pNames = ['11']
    fNames = ['21', '22', '23']

    for t in returnTypes:
        inDataRaw = numpy.array([[-111, 21, 22, 23], [11, 1, -1, -3]])
        specRaw = numpy.array([[1, -1, -3]])
        inData = nimble.createData(
            returnType=t, data=inDataRaw, pointNames=True, featureNames=True)
        specified = nimble.createData(
            returnType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
        assert inData == specified


def test_extractNames_NPMatrix():
    """ Test of createData() given numpy matrix, extracting names """
    pNames = ['11']
    fNames = ['21', '22', '23']

    for t in returnTypes:
        inDataRaw = numpy.array([[-111, 21, 22, 23], [11, 1, -1, -3]])
        specRaw = numpy.matrix([[1, -1, -3]])
        inData = nimble.createData(
            returnType=t, data=inDataRaw, pointNames=True, featureNames=True)
        specified = nimble.createData(
            returnType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
        assert inData == specified


def test_extractNames_CooSparse():
    """ Test of createData() given scipy Coo matrix, extracting names """
    if not scipy:
        return

    pNames = ['11']
    fNames = ['21', '22', '23']

    for t in returnTypes:
        inDataRaw = numpy.array([[-111, 21, 22, 23], [11, 1, -1, -3]])
        inDataRaw = scipy.sparse.coo_matrix(inDataRaw)
        specRaw = numpy.array([[1, -1, -3]])
        specRaw = scipy.sparse.coo_matrix(specRaw)

        inData = nimble.createData(
            returnType=t, data=inDataRaw, pointNames=True, featureNames=True)
        specified = nimble.createData(
            returnType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
        assert inData == specified


def test_extractNames_CscSparse():
    """ Test of createData() given scipy Coo matrix, extracting names """
    if not scipy:
        return

    pNames = ['11']
    fNames = ['21', '22', '23']

    for t in returnTypes:
        inDataRaw = numpy.array([[-111, 21, 22, 23], [11, 1, -1, -3]])
        inDataRaw = scipy.sparse.csc_matrix(inDataRaw)
        specRaw = numpy.array([[1, -1, -3]])
        specRaw = scipy.sparse.csc_matrix(specRaw)

        inData = nimble.createData(
            returnType=t, data=inDataRaw, pointNames=True, featureNames=True)
        specified = nimble.createData(
            returnType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
        assert inData == specified


def test_extractNames_pandasDataFrame():
    pNames = ['11']
    fNames = ['21', '22', '23']

    for t in returnTypes:
        inDataRaw = pd.DataFrame([[1, -1, -3]], index=[11], columns=[21, 22, 23])
        specRaw = pd.DataFrame([[1, -1, -3]])

        inData = nimble.createData(
            returnType=t, data=inDataRaw, pointNames=True, featureNames=True)
        specified = nimble.createData(
            returnType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
        assert inData == specified


def test_names_dataUnmodified():
    """ Test original data unmodifed when names set to 'automatic' or True """
    autoData = [['pointNames', 'fname0', 'fname1', 'fname2'], ['pt', 1, -1, -3]]
    autoArray = numpy.array(autoData, dtype=numpy.object_)
    trueData = [[-111, 21, 22, 23], [11, 1, -1, -3]]

    def assertUnmodified(rawData, names):
        if isinstance(rawData, list):
            rawDataCopy = [lst.copy() for lst in rawData]
        else:
            rawDataCopy = rawData.copy()
        inData = nimble.createData(
            returnType=t, data=rawData, pointNames=names, featureNames=names)

        if isinstance(rawData, list):
            rawData == rawDataCopy
        elif scipy.sparse.isspmatrix(rawData):
            numpy.testing.assert_array_equal(rawData.todense(), rawDataCopy.todense())
        else:
            numpy.testing.assert_array_equal(rawData, rawDataCopy)


    for t in returnTypes:
        assertUnmodified(autoData, 'automatic')
        assertUnmodified(trueData, True)
        assertUnmodified(autoArray, 'automatic')
        assertUnmodified(numpy.array(trueData), True)
        assertUnmodified(numpy.matrix(autoArray), 'automatic')
        assertUnmodified(numpy.matrix(trueData), True)
        assertUnmodified(scipy.sparse.coo_matrix(autoArray), 'automatic')
        assertUnmodified(scipy.sparse.coo_matrix(trueData), True)
        assertUnmodified(pd.DataFrame([[1, -1, -3]], index=['pt'],
                                      columns=['fname0', 'fname1', 'fname2']),
                         'automatic')
        assertUnmodified(pd.DataFrame([[1, -1, -3]], index=[11], columns=[21, 22, 23]),
                         True)


###############################
# Open file as source of data #
###############################

class NamelessFile(object):
    def __init__(self, toWrap):
        self.inner = toWrap

    def __getattr__(self, name):
        if name != 'name':
            return getattr(self.inner, name)
        else:
            raise AttributeError

    def __iter__(self):
        return self.inner.__iter__()


def test_createData_CSV_passedOpen():
    for t in returnTypes:
        fromList = nimble.createData(returnType=t, data=[[1, 2, 3]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.flush()
            objName = 'fromCSV'
            openFile = open(tmpCSV.name, 'rU')
            fromCSV = nimble.createData(returnType=t, data=openFile, name=objName)
            openFile.close()

            assert fromList == fromCSV

            assert fromCSV.path == openFile.name
            assert fromCSV.absolutePath == openFile.name
            assert fromCSV.relativePath == os.path.relpath(openFile.name)

            openFile = open(openFile.name, 'rU')
            namelessOpenFile = NamelessFile(openFile)
            fromCSV = nimble.createData(
                returnType=t, data=namelessOpenFile)
            assert fromCSV.name.startswith(nimble.data.dataHelpers.DEFAULT_NAME_PREFIX)
            assert fromCSV.path is None
            assert fromCSV.absolutePath is None
            assert fromCSV.relativePath is None


def test_createData_MTXArr_passedOpen():
    for t in returnTypes:
        fromList = nimble.createData(returnType=t, data=[[1, 2, 3]])

        # instantiate from mtx array file
        with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXArr:
            tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
            tmpMTXArr.write("1 3\n")
            tmpMTXArr.write("1\n")
            tmpMTXArr.write("2\n")
            tmpMTXArr.write("3\n")
            tmpMTXArr.flush()
            objName = 'fromMTXArr'
            openFile = open(tmpMTXArr.name, 'rU')
            fromMTXArr = nimble.createData(returnType=t, data=openFile, name=objName)
            openFile.close()

            if t is None and fromList.getTypeString() != fromMTXArr.getTypeString():
                assert fromList.isApproximatelyEqual(fromMTXArr)
            else:
                assert fromList == fromMTXArr

            assert fromMTXArr.path == openFile.name
            assert fromMTXArr.absolutePath == openFile.name
            assert fromMTXArr.relativePath == os.path.relpath(openFile.name)

            openFile = open(tmpMTXArr.name, 'rU')
            namelessOpenFile = NamelessFile(openFile)
            fromMTXArr = nimble.createData(
                returnType=t, data=namelessOpenFile)
            assert fromMTXArr.name.startswith(
                nimble.data.dataHelpers.DEFAULT_NAME_PREFIX)
            assert fromMTXArr.path is None
            assert fromMTXArr.absolutePath is None
            assert fromMTXArr.relativePath is None


def test_createData_MTXCoo_passedOpen():
    for t in returnTypes:
        fromList = nimble.createData(returnType=t, data=[[1, 2, 3]])

        # instantiate from mtx coordinate file
        with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXCoo:
            tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.write("1 1 1\n")
            tmpMTXCoo.write("1 2 2\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.flush()
            objName = 'fromMTXCoo'
            openFile = open(tmpMTXCoo.name, 'rU')
            fromMTXCoo = nimble.createData(returnType=t, data=openFile, name=objName)
            openFile.close()

            if t is None and fromList.getTypeString() != fromMTXCoo.getTypeString():
                assert fromList.isApproximatelyEqual(fromMTXCoo)
            else:
                assert fromList == fromMTXCoo

            assert fromMTXCoo.path == openFile.name
            assert fromMTXCoo.absolutePath == openFile.name
            assert fromMTXCoo.relativePath == os.path.relpath(openFile.name)

            openFile = open(tmpMTXCoo.name, 'rU')
            namelessOpenFile = NamelessFile(openFile)
            fromMTXCoo = nimble.createData(
                returnType=t, data=namelessOpenFile)
            assert fromMTXCoo.name.startswith(
                nimble.data.dataHelpers.DEFAULT_NAME_PREFIX)
            assert fromMTXCoo.path is None
            assert fromMTXCoo.absolutePath is None
            assert fromMTXCoo.relativePath is None

###########################
# url as a source of data #
###########################

def mocked_requests_get(*args, **kwargs):
    class MockResponse:
        """mock of Response object returned by a call to requests.get"""
        def __init__(self, content, status_code, ok=True, reason=None, encoding='utf-8'):
            # In Response object, .content returns bytes and .text returns unicode
            # python2 uses .content and python3 uses .text in the code, so setting
            # self.content and self.text to content replicates the desired behavior
            self.content = content
            self.text = content
            self.status_code = status_code
            self.ok = ok
            self.reason = reason
            self.apparent_encoding = encoding

    if args[0] == 'http://mockrequests.nimble/CSVNoExtension':
        return MockResponse('1,2,3\n4,5,6', 200)
    elif args[0] == 'http://mockrequests.nimble/CSVAmbiguousExtension.data':
        return MockResponse('1,2,3\n4,5,6', 200)
    elif args[0] == 'http://mockrequests.nimble/CSV.csv':
        return MockResponse('1,2,3\n4,5,6', 200)
    elif args[0] == 'http://mockrequests.nimble/CSVcarriagereturn.csv':
        return MockResponse('1,2,3\r4,5,6', 200)
    elif args[0] == 'http://mockrequests.nimble/CSVunicodetest.csv':
        return MockResponse('1,2,\xc2\xa1\n4,5,6', 200)
    elif args[0] == 'http://mockrequests.nimble/CSVquotednewline.csv':
        # csv allows for newline characters in field values within double quotes
        return MockResponse('1,2,"a/nb"\n4,5,6', 200)
    elif (args[0] == 'http://mockrequests.nimble/MTXNoExtension' or
          args[0] == 'http://mockrequests.nimble/MTXAmbiguousExtension.data' or
          args[0] == 'http://mockrequests.nimble/MTX.mtx'):
        mtx = '%%MatrixMarket matrix coordinate real general\n2 3 6\n1 1 1\n1 2 2\n1 3 3\n2 1 4\n2 2 5\n2 3 6'
        return MockResponse(mtx, 200)

    return MockResponse(None, 404, False, 'Not Found')

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_createData_http_CSVNoExtension(mock_get):
    for t in returnTypes:
        exp = nimble.createData(returnType=t, data=[[1,2,3],[4,5,6]])
        url = 'http://mockrequests.nimble/CSVNoExtension'
        fromWeb = nimble.createData(returnType=t, data=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_createData_http_CSVAmbiguousExtension(mock_get):
    for t in returnTypes:
        exp = nimble.createData(returnType=t, data=[[1,2,3],[4,5,6]])
        url = 'http://mockrequests.nimble/CSVAmbiguousExtension.data'
        fromWeb = nimble.createData(returnType=t, data=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_createData_http_CSVFileOK(mock_get):
    for t in returnTypes:
        exp = nimble.createData(returnType=t, data=[[1,2,3],[4,5,6]])
        url = 'http://mockrequests.nimble/CSV.csv'
        fromWeb = nimble.createData(returnType=t, data=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_createData_http_CSVCarriageReturn(mock_get):
    for t in returnTypes:
        exp = nimble.createData(returnType=t, data=[[1,2,3],[4,5,6]])
        url = 'http://mockrequests.nimble/CSVcarriagereturn.csv'
        fromWeb = nimble.createData(returnType=t, data=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_createData_http_CSVNonUnicodeValues(mock_get):
    for t in returnTypes:
        exp = nimble.createData(returnType=t, data=[[1,2,'\xc2\xa1'],[4,5,6]])
        url = 'http://mockrequests.nimble/CSVunicodetest.csv'
        fromWeb = nimble.createData(returnType=t, data=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_createData_http_CSVQuotedNewLine(mock_get):
    for t in returnTypes:
        exp = nimble.createData(returnType=t, data=[[1,2,"a/nb"],[4,5,6]])
        url = 'http://mockrequests.nimble/CSVquotednewline.csv'
        fromWeb = nimble.createData(returnType=t, data=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_createData_http_CSVPathsEqualUrl(mock_get):
    for t in returnTypes:
        exp = nimble.createData(returnType=t, data=[[1,2,3],[4,5,6]])
        url = 'http://mockrequests.nimble/CSVNoExtension'
        fromWeb = nimble.createData(returnType=t, data=url)
        assert fromWeb.absolutePath == url
        assert fromWeb.relativePath == None

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_createData_http_MTXNoExtension(mock_get):
    for t in returnTypes:
        # None returnType for url will default to Sparse so use coo_matrix for data
        data = scipy.sparse.coo_matrix([[1,2,3],[4,5,6]])
        exp = nimble.createData(returnType=t, data=data)
        url = 'http://mockrequests.nimble/MTXNoExtension'
        fromWeb = nimble.createData(returnType=t, data=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_createData_http_MTXAmbiguousExtension(mock_get):
    for t in returnTypes:
        # None returnType for url will default to Sparse so use coo_matrix for data
        data = scipy.sparse.coo_matrix([[1,2,3],[4,5,6]])
        exp = nimble.createData(returnType=t, data=data)
        url = 'http://mockrequests.nimble/MTXAmbiguousExtension.data'
        fromWeb = nimble.createData(returnType=t, data=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_createData_http_MTXFileOK(mock_get):
    for t in returnTypes:
        # None returnType for url will default to Sparse so use coo_matrix for data
        data = scipy.sparse.coo_matrix([[1,2,3],[4,5,6]])
        exp = nimble.createData(returnType=t, data=data)
        url = 'http://mockrequests.nimble/MTX.mtx'
        fromWeb = nimble.createData(returnType=t, data=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_createData_http_MTXPathsEqualUrl(mock_get):
    for t in returnTypes:
        # None returnType for url will default to Sparse so use coo_matrix for data
        data = scipy.sparse.coo_matrix([[1,2,3],[4,5,6]])
        exp = nimble.createData(returnType=t, data=data)
        url = 'http://mockrequests.nimble/MTXNoExtension'
        fromWeb = nimble.createData(returnType=t, data=url)
        assert fromWeb.absolutePath == url
        assert fromWeb.relativePath == None

@raises(InvalidArgumentValue)
@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_createData_http_linkError(mock_get):
    for t in returnTypes:
        url = 'http://mockrequests.nimble/linknotfound.csv'
        fromWeb = nimble.createData(returnType=t, data=url)

###################################
# ignoreNonNumericalFeatures flag #
###################################

def test_createData_ignoreNonNumericalFeaturesCSV():
    for t in returnTypes:
        fromList = nimble.createData(returnType=t, data=[[1, 3], [5, 7]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,two,3.0,four\n")
            tmpCSV.write("5,six,7,8\n")
            tmpCSV.flush()

            fromCSV = nimble.createData(
                returnType=t, data=tmpCSV.name, ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV

            # sanity check
            fromCSV = nimble.createData(returnType=t, data=tmpCSV.name)
            assert len(fromCSV.features) == 4


def test_createData_CSV_ignoreNonNumerical_removalCleanup_hard():
    for t in returnTypes:
        fromList = nimble.createData(returnType=t, data=[[1, 3], [5, 7], [11, 12], [13, 14]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3.0,4.0,1\n")
            tmpCSV.write("5,six,7,8,1\n")
            tmpCSV.write("11,6,12,eight,1.0\n")
            tmpCSV.write("13,one,14,9,who?\n")
            tmpCSV.flush()

            fromCSV = nimble.createData(
                returnType=t, data=tmpCSV.name, ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV

            # sanity check
            fromCSV = nimble.createData(returnType=t, data=tmpCSV.name)
            assert len(fromCSV.features) == 5


def test_createData_CSV_ignoreNonNumerical_removalCleanup_easy():
    for t in returnTypes:
        fromList = nimble.createData(returnType=t, data=[[1, 3], [5, 7], [11, 12], [13, 14]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,two,3.0,four,one\n")
            tmpCSV.write("5,6,7,8,1\n")
            tmpCSV.write("11,6,12,8,1.0\n")
            tmpCSV.write("13,1,14,9,2\n")
            tmpCSV.flush()

            fromCSV = nimble.createData(
                returnType=t, data=tmpCSV.name, ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV

            # sanity check
            fromCSV = nimble.createData(returnType=t, data=tmpCSV.name)
            assert len(fromCSV.features) == 5


def test_createData_ignoreNonNumericalFeaturesCSV_noEffect():
    for t in returnTypes:
        fromList = nimble.createData(returnType=t, data=[[1, 2, 3, 4], [5, 6, 7, 8]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3,4\n")
            tmpCSV.write("5,6,7,8\n")
            tmpCSV.flush()

            fromCSV = nimble.createData(
                returnType=t, data=tmpCSV.name, ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV

            fromCSV = nimble.createData(returnType=t, data=tmpCSV.name)
            assert len(fromCSV.features) == 4


def test_CSV_ignoreNonNumericalFeatures_featureNamesDontTrigger():
    for t in returnTypes:
        fnames = ['1', '2', '3', 'four']
        fromList = nimble.createData(returnType=t, featureNames=fnames, data=[[5, 6, 7, 8]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3,four\n")
            tmpCSV.write("5,6,7,8\n")
            tmpCSV.flush()

            fromCSV = nimble.createData(
                returnType=t, data=tmpCSV.name, featureNames=True,
                ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV


def test_CSV_ignoreNonNumericalFeatures_featureNamesAdjusted():
    for t in returnTypes:
        fNames = ["1", "2", "3"]
        data = [[1, 2, 3], [5, 6, 7]]
        fromList = nimble.createData(returnType=t, featureNames=fNames, data=data)

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3,4\n")
            tmpCSV.write("1,2,3,four\n")
            tmpCSV.write("5,6,7,H8\n")
            tmpCSV.flush()

            fromCSV = nimble.createData(
                returnType=t, data=tmpCSV.name, featureNames=True,
                ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV


def test_CSV_createData_ignoreNonNumericalFeatures_allRemoved():
    for t in returnTypes:
        pNames = ['single', 'dubs', 'trips']
        fromList = nimble.createData(returnType=t, pointNames=pNames, data=[[], [], []])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write(",ones,twos,threes\n")
            tmpCSV.write("single,1A,2A,3A\n")
            tmpCSV.write("dubs,11,22A,33\n")
            tmpCSV.write("trips,111,222,333\n")
            tmpCSV.flush()

            fromCSV = nimble.createData(
                returnType=t, data=tmpCSV.name, pointNames=True,
                featureNames=True, ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV


####################################################
# Difficult CSV Formatting: whitespace and quoting #
####################################################

def test_CSVformatting_simpleQuotedValues():
    for t in returnTypes:
        fromList = nimble.createData(returnType=t, data=[[1, 2, 3, 4], [5, 6, 7, 8]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,\"2\",\"3\",4\n")
            tmpCSV.write("5,\"6\",\"7\",8\n")
            tmpCSV.flush()

            fromCSV = nimble.createData(returnType=t, data=tmpCSV.name)

            assert fromList == fromCSV


def test_CSVformatting_specialCharsInQuotes():
    for t in returnTypes:
        fNames = ["1,ONE", "2;TWO", "3\t'EE'"]
        data = [[1, 2, 3], [5, 6, 7]]
        dataAll = [[1, 2, 3, 4], [5, 6, 7, 8]]
        fromList = nimble.createData(returnType=t, featureNames=fNames[:3], data=data)

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("\"1,ONE\",\"2;TWO\",\"3\t'EE'\",\"4f\"\n")
            tmpCSV.write("1,2,3,four\n")
            tmpCSV.write("5,6,7,H8\n")
            tmpCSV.flush()

            fromCSV = nimble.createData(
                returnType=t, data=tmpCSV.name, featureNames=True,
                ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV


def test_CSVformatting_emptyAndCommentLines():
    for t in returnTypes:
        data = [[1, 2, 3, 4], ['#11', 22, 33, 44], [5, 6, 7, 8]]

        fromList = nimble.createData(returnType=t, data=data)

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("#stuff\n")
            tmpCSV.write("\n")
            tmpCSV.write("\n")
            tmpCSV.write("#1,2,3,4\n")
            tmpCSV.write("\n")
            tmpCSV.write("1,2,3,4\n")
            tmpCSV.write("#11,22,33, 44\n")
            tmpCSV.write("\n")
            tmpCSV.write("5,6,7,8\n")
            tmpCSV.write("\n")
            tmpCSV.flush()

            fromCSV = nimble.createData(
                returnType=t, data=tmpCSV.name, featureNames=False)

            assert fromList == fromCSV


def test_CSVformatting_scientificNotation():
    for t in returnTypes:
        data = [[1., 2., 3.], [11., 22., 33.], [111., 222., 333.]]
        fromRaw = nimble.createData(returnType=t, data=data)

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1.000000000e+00,2.000000000e+00,3.000000000e+00\n")
            tmpCSV.write("1.100000000e+01,2.200000000e+01,3.300000000e+01\n")
            tmpCSV.write("1.110000000e+02,2.220000000e+02,3.330000000e+02\n")
            tmpCSV.flush()

            fromCSV = nimble.createData(returnType=t, data=tmpCSV.name)

            assert fromRaw == fromCSV


################################
# keepPoints, keepFeatures #
################################

def test_createData_keepPF_AllPossibleNatOrder():
    filesForms = ['csv', 'mtx']
    for (t, f) in itertools.product(returnTypes, filesForms):
        data = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
        orig = nimble.createData(returnType=t, data=data)
        with tempfile.NamedTemporaryFile(suffix="." + f) as tmpF:
            orig.writeFile(tmpF.name, fileFormat=f, includeNames=False)
            tmpF.flush()

            poss = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2], 'all']
            for (pSel, fSel) in itertools.product(poss, poss):
                ret = nimble.createData(
                    t, tmpF.name, keepPoints=pSel, keepFeatures=fSel)
                fromOrig = nimble.createData(
                    t, orig.data, keepPoints=pSel, keepFeatures=fSel)

                assert ret == fromOrig


def test_createData_keepPF_AllPossibleReverseOrder():
    filesForms = ['csv', 'mtx']
    for (t, f) in itertools.product(returnTypes, filesForms):
        data = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
        orig = nimble.createData(returnType=t, data=data)
        with tempfile.NamedTemporaryFile(suffix="." + f) as tmpF:
            orig.writeFile(tmpF.name, fileFormat=f, includeNames=False)
            tmpF.flush()

            poss = [[0, 1], [0, 2], [1, 2], [0, 1, 2]]
            for (pSel, fSel) in itertools.product(poss, poss):
                ret = nimble.createData(
                    t, tmpF.name, keepPoints=pSel, keepFeatures=fSel)
                fromOrig = nimble.createData(
                    t, orig.data, keepPoints=pSel, keepFeatures=fSel)

                assert ret == fromOrig

                pSelR = copy.copy(pSel)
                pSelR.reverse()
                fSelR = copy.copy(fSel)
                fSelR.reverse()

                retT = nimble.createData(
                    t, tmpF.name, keepPoints=pSelR, keepFeatures=fSelR)
                fromOrigT = nimble.createData(
                    t, orig.data, keepPoints=pSelR, keepFeatures=fSelR)

                assert retT != ret
                assert retT == fromOrigT
                assert fromOrigT != fromOrig


def test_createData_keepPF_AllPossibleWithNames_extracted():
    data = [[1., 2., 3.], [11., 22., 33.], [111., 222., 333.]]
    orig = nimble.createData(returnType="List", data=data)
    filesForms = ['csv', 'mtx']
    for (t, f) in itertools.product(returnTypes, filesForms):
        with tempfile.NamedTemporaryFile(suffix="." + f) as tmpF:
            orig.writeFile(tmpF.name, fileFormat=f, includeNames=False)
            tmpF.flush()

            poss = [[0], [1], [0, 1], [1, 0], 'all']
            for (pSel, fSel) in itertools.product(poss, poss):
                toUse = orig.copy(to="pythonlist")
                fromOrig = nimble.createData(
                    t, toUse, keepPoints=pSel, keepFeatures=fSel,
                    pointNames=True, featureNames=True)

                ret = nimble.createData(
                    t, tmpF.name, keepPoints=pSel, keepFeatures=fSel,
                    pointNames=True, featureNames=True)

                pSelUse = copy.copy(pSel)
                fSelUse = copy.copy(fSel)
                if pSel != 'all':
                    for i in range(len(pSel)):
                        pSelUse[i] = ret.points.getName(i)

                if fSel != 'all':
                    for i in range(len(fSel)):
                        fSelUse[i] = ret.features.getName(i)

                retN = nimble.createData(
                    t, tmpF.name, keepPoints=pSelUse,
                    keepFeatures=fSelUse, pointNames=True,
                    featureNames=True)

                assert ret == fromOrig
                assert retN == fromOrig


def test_createData_keepPF_AllPossibleWithNames_listProvided():
    pnames = ["11.", "111."]
    fnames = ["2.", "3."]
    data = [[22., 33.], [222., 333.]]
    orig = nimble.createData(
        returnType="List", data=data, pointNames=pnames, featureNames=fnames)

    filesForms = ['csv', 'mtx']
    for (t, f) in itertools.product(returnTypes, filesForms):
        with tempfile.NamedTemporaryFile(suffix="." + f) as tmpF:
            orig.writeFile(tmpF.name, fileFormat=f, includeNames=False)
            tmpF.flush()

            poss = [[0], [1], 'all']
            for (pSel, fSel) in itertools.product(poss, poss):
                toUseData = orig.copy(to="pythonlist")

                fromOrig = nimble.createData(
                    t, toUseData, keepPoints=pSel, keepFeatures=fSel,
                    pointNames=pnames, featureNames=fnames)

                ret = nimble.createData(
                    t, tmpF.name, keepPoints=pSel, keepFeatures=fSel,
                    pointNames=pnames, featureNames=fnames)

                pSelUse = copy.copy(pSel)
                fSelUse = copy.copy(fSel)
                if pSel != 'all':
                    for i in range(len(pSel)):
                        pSelUse[i] = ret.points.getName(i)

                if fSel != 'all':
                    for i in range(len(fSel)):
                        fSelUse[i] = ret.features.getName(i)

                retN = nimble.createData(
                    t, tmpF.name, keepPoints=pSelUse,
                    keepFeatures=fSelUse, pointNames=pnames,
                    featureNames=fnames)

                assert ret == fromOrig
                assert retN == fromOrig


def test_createData_keepPF_AllPossibleWithNames_dictProvided():
    pnames = {"11.": 0, "111.": 1}
    fnames = {"2.": 0, "3.": 1}
    data = [[22., 33.], [222., 333.]]
    orig = nimble.createData(
        returnType="List", data=data, pointNames=pnames, featureNames=fnames)

    filesForms = ['csv', 'mtx']
    for (t, f) in itertools.product(returnTypes, filesForms):
        with tempfile.NamedTemporaryFile(suffix="." + f) as tmpF:
            orig.writeFile(tmpF.name, fileFormat=f, includeNames=False)
            tmpF.flush()

            poss = [[0], [1], 'all']
            for (pSel, fSel) in itertools.product(poss, poss):
                toUseData = orig.copy(to="pythonlist")

                fromOrig = nimble.createData(
                    t, toUseData, keepPoints=pSel, keepFeatures=fSel,
                    pointNames=pnames, featureNames=fnames)

                ret = nimble.createData(
                    t, tmpF.name, keepPoints=pSel, keepFeatures=fSel,
                    pointNames=pnames, featureNames=fnames)

                pSelUse = copy.copy(pSel)
                fSelUse = copy.copy(fSel)
                if pSel != 'all':
                    for i in range(len(pSel)):
                        pSelUse[i] = ret.points.getName(i)

                if fSel != 'all':
                    for i in range(len(fSel)):
                        fSelUse[i] = ret.features.getName(i)

                retN = nimble.createData(
                    t, tmpF.name, keepPoints=pSelUse,
                    keepFeatures=fSelUse, pointNames=pnames,
                    featureNames=fnames)

                assert ret == fromOrig
                assert retN == fromOrig


def test_createData_csv_keepPoints_IndexingGivenFeatureNames():
    data = [[111, 222, 333]]
    fnames = ['1', '2', '3']
    wanted = nimble.createData("Matrix", data=data, featureNames=fnames)
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        fromCSV = nimble.createData(
            "Matrix", data=tmpCSV.name, keepPoints=[1], featureNames=True)

        raw = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
        fromRaw = nimble.createData(
            "Matrix", data=raw, keepPoints=[1], featureNames=True)

        assert fromRaw == wanted
        assert fromCSV == wanted


# since the createData helper for raw data is chained onto the
# helper for file input, we need tests to show that we don't
# just read all of the data into an object and then remove the
# stuff we don't want in the raw data helper. If these pass,
# unwanted data could still be stored in memory, but it limits
# that mistake to the file input helpers only.

def test_createData_keepPF_csv_noUncessaryStorage():
    wanted = nimble.createData("List", data=[[22], [222]])
    backup = nimble.helpers.initDataObject

    try:
        def fakeinitDataObject(
                returnType, rawData, pointNames, featureNames, name, path,
                keepPoints, keepFeatures, treatAsMissing, replaceMissingWith,
                reuseData=False):
            assert len(rawData) == 2
            assert len(rawData[0]) == 1
            return nimble.data.List(rawData)

        nimble.helpers.initDataObject = fakeinitDataObject

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.write("11,22,33\n")
            tmpCSV.write("111,222,333\n")
            tmpCSV.flush()

            fromCSV = nimble.createData(
                "List", data=tmpCSV.name, keepPoints=[1, 2], keepFeatures=[1])
            assert fromCSV == wanted
    finally:
        nimble.helpers.initDataObject = backup

#def TODOtest_createData_keepPF_mtxArr_noUncessaryStorage():
#	fromList = nimble.createData(returnType='Matrix', data=[[2]])
#	backup = nimble.helpers.initDataObject
#
#	try:
#		def fakeinitDataObject(
#				returnType, rawData, pointNames, featureNames, name, path,
#				keepPoints, keepFeatures):
#			assert len(rawData) == 1
#			assert len(rawData[0]) == 1
#			return nimble.data.List(rawData)
#
#		nimble.helpers.initDataObject = fakeinitDataObject
#
#		# instantiate from mtx array file
#		with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXArr:
#			tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
#			tmpMTXArr.write("1 3\n")
#			tmpMTXArr.write("1\n")
#			tmpMTXArr.write("2\n")
#			tmpMTXArr.write("3\n")
#			tmpMTXArr.flush()
#			fromMTXArr = nimble.createData(
#				returnType='Matrix', data=tmpMTXArr.name, keepPoints=[0],
#				keepFeatures=[1])
#
#			assert fromList == fromMTXArr
#	finally:
#		nimble.helpers.initDataObject = backup


#def TODOtest_createData_keepPF_mtxCoo_noUncessaryStorage():
#	fromList = nimble.createData(returnType='Matrix', data=[[2]])
#	backup = nimble.helpers.initDataObject
#
#	try:
#		def fakeinitDataObject(
#				returnType, rawData, pointNames, featureNames, name, path,
#				keepPoints, keepFeatures):
#			assert rawData.shape == (1,1)
#			return nimble.data.List(rawData)
#
#		nimble.helpers.initDataObject = fakeinitDataObject
#
#		# instantiate from mtx coordinate file
#		with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXCoo:
#			tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
#			tmpMTXCoo.write("1 3 3\n")
#			tmpMTXCoo.write("1 1 1\n")
#			tmpMTXCoo.write("1 2 2\n")
#			tmpMTXCoo.write("1 3 3\n")
#			tmpMTXCoo.flush()
#			fromMTXCoo = nimble.createData(
#				returnType='Matrix', data=tmpMTXCoo.name, keepPoints=[0],
#				keepFeatures=[1])
#
#			assert fromList == fromMTXCoo
#
#	finally:
#		nimble.helpers.initDataObject = backup


def test_createData_keepPF_csv_simple():
    wanted = nimble.createData("Matrix", data=[[222], [22]])
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        fromCSV = nimble.createData(
            "Matrix", data=tmpCSV.name, keepPoints=[2, 1], keepFeatures=[1])
        assert fromCSV == wanted


def test_createData_keepPF_mtxArr_simple():
    fromList = nimble.createData(returnType='Matrix', data=[[2]])

    # instantiate from mtx array file
    with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXArr:
        tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
        tmpMTXArr.write("1 3\n")
        tmpMTXArr.write("1\n")
        tmpMTXArr.write("2\n")
        tmpMTXArr.write("3\n")
        tmpMTXArr.flush()
        fromMTXArr = nimble.createData(
            returnType='Matrix', data=tmpMTXArr.name, keepPoints=[0],
            keepFeatures=[1])

        assert fromList == fromMTXArr


def test_createData_keepPF_mtxCoo_simple():
    fromList = nimble.createData(returnType='Matrix', data=[[2]])

    # instantiate from mtx coordinate file
    with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXCoo:
        tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
        tmpMTXCoo.write("1 3 3\n")
        tmpMTXCoo.write("1 1 1\n")
        tmpMTXCoo.write("1 2 2\n")
        tmpMTXCoo.write("1 3 3\n")
        tmpMTXCoo.flush()
        fromMTXCoo = nimble.createData(
            returnType='Matrix', data=tmpMTXCoo.name, keepPoints=[0],
            keepFeatures=[1])

        assert fromList == fromMTXCoo


def test_createData_keepPF_pythonList_simple():
    wanted = nimble.createData("Matrix", data=[[22, 33], [222, 333]])
    raw = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]

    fromList = nimble.createData(
        "Matrix", data=raw, keepPoints=[1, 2], keepFeatures=[1, 2])
    assert fromList == wanted

    wanted = nimble.createData("Matrix", data=[[333, 222], [33, 22]])
    fromList = nimble.createData(
        "Matrix", data=raw, keepPoints=[2, 1], keepFeatures=[2, 1])
    assert fromList == wanted


def test_createData_keepPF_npArray_simple():
    wanted = nimble.createData("Matrix", data=[[22, 33], [222, 333]])
    rawList = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
    raw = numpy.array(rawList)

    fromNPArr = nimble.createData(
        "Matrix", data=raw, keepPoints=[1, 2], keepFeatures=[1, 2])
    assert fromNPArr == wanted

    wanted = nimble.createData("Matrix", data=[[333, 222], [33, 22]])
    fromNPArr = nimble.createData(
        "Matrix", data=raw, keepPoints=[2, 1], keepFeatures=[2, 1])
    assert fromNPArr == wanted


def test_createData_keepPF_npMatrix_simple():
    wanted = nimble.createData("Matrix", data=[[22, 33], [222, 333]])
    rawList = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
    raw = numpy.matrix(rawList)

    fromList = nimble.createData(
        "Matrix", data=raw, keepPoints=[1, 2], keepFeatures=[1, 2])
    assert fromList == wanted

    wanted = nimble.createData("Matrix", data=[[333, 222], [33, 22]])
    fromList = nimble.createData(
        "Matrix", data=raw, keepPoints=[2, 1], keepFeatures=[2, 1])
    assert fromList == wanted


def test_createData_keepPF_spCoo_simple():
    if not scipy:
        return
    wanted = nimble.createData("Matrix", data=[[22, 33], [222, 333]])
    rawList = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
    rawMat = numpy.matrix(rawList)
    raw = scipy.sparse.coo_matrix(rawMat)

    fromCOO = nimble.createData(
        "Matrix", data=raw, keepPoints=[1, 2], keepFeatures=[1, 2])
    assert fromCOO == wanted

    wanted = nimble.createData("Matrix", data=[[333, 222], [33, 22]])
    fromCOO = nimble.createData(
        "Matrix", data=raw, keepPoints=[2, 1], keepFeatures=[2, 1])
    assert fromCOO == wanted


def test_createData_keepPF_spCsc_simple():
    if not scipy:
        return
    wanted = nimble.createData("Matrix", data=[[22, 33], [222, 333]])
    rawList = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
    rawMat = numpy.matrix(rawList)
    raw = scipy.sparse.csc_matrix(rawMat)

    fromCSC = nimble.createData(
        "Matrix", data=raw, keepPoints=[1, 2], keepFeatures=[1, 2])
    assert fromCSC == wanted

    wanted = nimble.createData("Matrix", data=[[333, 222], [33, 22]])
    fromCSC = nimble.createData(
        "Matrix", data=raw, keepPoints=[2, 1], keepFeatures=[2, 1])
    assert fromCSC == wanted


@raises(InvalidArgumentValue)
def test_keepPF_csv_ExceptionUnknownFeatureName_Extracted():
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        nimble.createData(
            returnType='List', data=tmpCSV.name, pointNames=True,
            featureNames=True, keepFeatures=[0, "fours"])


@raises(InvalidArgumentValue)
def test_keepPF_csv_ExceptionUnknownFeatureName_Provided():
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        nimble.createData(
            returnType='List', data=tmpCSV.name,
            featureNames=['ones', 'twos', 'threes'], keepFeatures=[0, "fours"])


@raises(InvalidArgumentValue)
def test_csv_keepFeatures_indexNotInFile():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        nimble.createData(
            returnType='List', data=tmpCSV.name, pointNames=False,
            featureNames=False, keepPoints=[1, 2], keepFeatures=[1, 42])


@raises(InvalidArgumentValue)
def test_csv_keepPoints_indexNotInFile():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        nimble.createData(
            returnType='List', data=tmpCSV.name, pointNames=False,
            featureNames=False, keepPoints=[12, 2, 3], keepFeatures=[1, 2])


@raises(InvalidArgumentValue)
def test_keepPF_csv_ExceptionUnknownPointName_extracted():
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        nimble.createData(
            returnType='List', data=tmpCSV.name, pointNames=True,
            featureNames=True, keepPoints=[1, "quads"])


@raises(InvalidArgumentValue)
def test_keepPF_csv_ExceptionUnknownPointName_provided():
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        nimble.createData(
            returnType='List', data=tmpCSV.name,
            pointNames=['single', 'dubs', 'trips'], keepPoints=[1, "quads"])


@raises(InvalidArgumentValue)
def test_csv_keepPoints_noNamesButNameSpecified():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        nimble.createData(
            returnType='List', data=tmpCSV.name, pointNames=False,
            featureNames=False, keepPoints=['dubs', 1], keepFeatures=[2])


@raises(InvalidArgumentValue)
def test_csv_keepFeatures_noNamesButNameSpecified():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        nimble.createData(
            returnType='List', data=tmpCSV.name, pointNames=False,
            featureNames=False, keepFeatures=["threes"])


def test_csv_keepFeatures_duplicatesInList():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        try:
            nimble.createData(
                returnType='List', data=tmpCSV.name, pointNames=True,
                featureNames=True, keepFeatures=[1, 1])
            assert False
        except InvalidArgumentValue:
            pass
        try:
            nimble.createData(
                returnType='List', data=tmpCSV.name, pointNames=True,
                featureNames=True, keepFeatures=[1, 'twos'])
            assert False
        except InvalidArgumentValue:
            pass
        try:
            nimble.createData(
                returnType='List', data=tmpCSV.name, pointNames=True,
                featureNames=True, keepFeatures=['threes', 'threes'])
            assert False
        except InvalidArgumentValue:
            pass
        try:
            nimble.createData(
                returnType='List', data=tmpCSV.name, pointNames=True,
                featureNames=['ones', 'twos', 'threes'], keepFeatures=[1, 'twos'])
            assert False
        except InvalidArgumentValue:
            pass
        try:
            nimble.createData(
                returnType='List', data=tmpCSV.name, pointNames=True,
                featureNames=['ones', 'twos', 'threes'],
                keepFeatures=['threes', 'threes'])
            assert False
        except InvalidArgumentValue:
            pass


def test_csv_keepPoints_duplicatesInList():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        try:
            nimble.createData(
                returnType='List', data=tmpCSV.name, pointNames=True,
                featureNames=True, keepPoints=[1, 1])
            assert False
        except InvalidArgumentValue:
            pass
        try:
            nimble.createData(
                returnType='List', data=tmpCSV.name, pointNames=True,
                featureNames=True, keepPoints=[1, 'dubs'])
            assert False
        except InvalidArgumentValue:
            pass
        try:
            nimble.createData(
                returnType='List', data=tmpCSV.name, pointNames=True,
                featureNames=True, keepPoints=['trips', 'trips'])
            assert False
        except InvalidArgumentValue:
            pass
        try:
            nimble.createData(
                returnType='List', data=tmpCSV.name,
                pointNames=['single', 'dubs', 'trips'], featureNames=True,
                keepPoints=[1, 'dubs'])
            assert False
        except InvalidArgumentValue:
            pass
        try:
            nimble.createData(
                returnType='List', data=tmpCSV.name,
                pointNames=['single', 'dubs', 'trips'], featureNames=True,
                keepPoints=['trips', 'trips'])
            assert False
        except InvalidArgumentValue:
            pass


def test_createData_csv_keepPF_and_ignoreFlag():
    for t in returnTypes:
        fnames = ['threes']
        pnames = ['trips', 'dubs']
        data = [[333], [33]]
        fromList = nimble.createData(
            returnType=t, data=data, pointNames=pnames, featureNames=fnames)

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("pns,ones,twos,threes\n")
            tmpCSV.write("single,1,2,3A\n")
            tmpCSV.write("dubs,11,22A,33\n")
            tmpCSV.write("trips,111,222,333\n")
            tmpCSV.flush()

            fromCSV = nimble.createData(
                returnType=t, data=tmpCSV.name, pointNames=True,
                featureNames=True, keepPoints=[2, 'dubs'],
                keepFeatures=[1, 'threes'], ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV


def test_createData_keepPoints_csv_endAfterAllFound():
    wanted = nimble.createData("Matrix", data=[[11, 22, 33], [1, 2, 3]])
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        # This line has an extra value - if it was actually read by the
        # csv input helper, it would cause an exception to be raised.
        tmpCSV.write("111,222,333,444\n")
        tmpCSV.flush()

        fromCSV = nimble.createData("Matrix", data=tmpCSV.name, keepPoints=[1, 0])
        assert fromCSV == wanted


def test_createData_keepPF_csv_nameAlignment_allNames():
    for t in nimble.data.available:
        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.write("11,22,33\n")
            tmpCSV.write("111,222,333\n")
            tmpCSV.flush()

            # names includes all names for point/features in csv,
            # even though we are not keeping all of them
            pNamesL = ['first', 'second', 'third']
            fNamesL = ['one', 'two', 'three']
            pNamesD = {'first': 0, 'second': 1, 'third': 2}
            fNamesD = {'one': 0, 'two': 1, 'three': 2}

            fromCSVL = nimble.createData(t, data=tmpCSV.name, pointNames=pNamesL,
                                         featureNames=fNamesL, keepPoints=[2, 1],
                                         keepFeatures=[1, 0])
            fromCSVD = nimble.createData(t, data=tmpCSV.name, pointNames=pNamesD,
                                         featureNames=fNamesD, keepPoints=[2, 1],
                                         keepFeatures=[1, 0])

        keptPNames = ['third', 'second']
        keptFNames = ['two', 'one']
        keptData = [[222, 111], [22, 11]]
        expected = nimble.createData(t, keptData, keptPNames, keptFNames)

        assert fromCSVL == expected
        assert fromCSVD == expected


def test_createData_keepPF_csv_nameAlignment_keptNames():
    for t in nimble.data.available:
        # instantiate from csv file
        keptPNames = ['third', 'second']
        keptFNames = ['two', 'one']
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.write("11,22,33\n")
            tmpCSV.write("111,222,333\n")
            tmpCSV.flush()

            fromCSVL = nimble.createData(t, data=tmpCSV.name, pointNames=keptPNames,
                                         featureNames=keptFNames, keepPoints=[2, 1],
                                         keepFeatures=[1, 0])
            fromCSVD = nimble.createData(t, data=tmpCSV.name, pointNames=keptPNames,
                                         featureNames=keptFNames, keepPoints=[2, 1],
                                         keepFeatures=[1, 0])

        keptPNames = ['third', 'second']
        keptFNames = ['two', 'one']
        keptData = [[222, 111], [22, 11]]
        expected = nimble.createData(t, keptData, keptPNames, keptFNames)

        assert fromCSVL == expected
        assert fromCSVD == expected


def test_createData_csv_keepPoints_keepingAllPointNames_index():
    data = [[111, 222, 333], [11, 22, 33], [1, 2, 3]]
    pnames = ['1', '2', '3']
    wanted = nimble.createData("Matrix", data=data, pointNames=pnames)
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        # cannot assume that pnames contains all pointNames for data
        fromCSV = nimble.createData(
            "Matrix", data=tmpCSV.name, pointNames=pnames, keepPoints=[2, 1, 0])
        assert fromCSV == wanted


@raises(InvalidArgumentValue)
def test_createData_csv_keepPoints_keepingAllPointNames_names():
    data = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
    pnames = ['1', '2', '3']
    wanted = nimble.createData("Matrix", data=data, pointNames=pnames)
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        # cannot assume that pnames contains all pointNames for data
        fromCSV = nimble.createData(
            "Matrix", data=tmpCSV.name, pointNames=pnames, keepPoints=['3', '2', '1'])


def test_createData_csv_keepFeatures_keepingAllFeatureNames_index():
    data = [[2, 3, 1], [22, 33, 11], [222, 333, 111]]
    fnames = ['2', '3', '1']
    wanted = nimble.createData("Matrix", data=data, featureNames=fnames)
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        # assume featureNames passed aligns with order of keepFeatures
        fromCSV = nimble.createData(
            "Matrix", data=tmpCSV.name, featureNames=fnames, keepFeatures=[1, 2, 0])
        assert fromCSV == wanted

@raises(InvalidArgumentValue)
def test_createData_csv_keepFeatures_keepingAllFeatureNames_names():
    data = [[2, 3, 1], [22, 33, 11], [222, 333, 111]]
    fnames = ['b', 'c', 'a']
    wanted = nimble.createData("Matrix", data=data, featureNames=fnames)
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        # assume featureNames passed aligns with order of keepFeatures
        fromCSV = nimble.createData(
            "Matrix", data=tmpCSV.name, featureNames=['a', 'b', 'c'], keepFeatures=['b', 'c' ,'a'])
        assert fromCSV == wanted


def test_createData_csv_keepFeatures_reordersFeatureNames_fnamesTrue():
    data = [[22, 33, 11], [222, 333, 111]]
    fnames = ['2', '3', '1']
    wanted = nimble.createData("Matrix", data=data, featureNames=fnames)
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        # reordered based on keepFeatures since featureNames extracted
        fromCSVNames = nimble.createData(
            "Matrix", data=tmpCSV.name, featureNames=True, keepFeatures=fnames)
        assert fromCSVNames == wanted

        # reordered based on keepFeatures since featureNames extracted
        fromCSVIndex = nimble.createData(
            "Matrix", data=tmpCSV.name, featureNames=True, keepFeatures=[1, 2, 0])
        assert fromCSVIndex == wanted

######################
### inputSeparator ###
######################

def test_createData_csv_inputSeparatorAutomatic():
    wanted = nimble.createData("Matrix", data=[[1,2,3], [4,5,6]])
    # instantiate from csv file
    for delimiter in [',', '\t', ' ', ':', ';', '|']:
        with tempfile.NamedTemporaryFile(mode='w') as tmpCSV:
            tmpCSV.write("1{0}2{0}3\n".format(delimiter))
            tmpCSV.write("4{0}5{0}6\n".format(delimiter))
            tmpCSV.flush()

            fromCSV = nimble.createData("Matrix", data=tmpCSV.name)
            assert fromCSV == wanted

def test_createData_csv_inputSeparatorSpecified():
    wanted = nimble.createData("Matrix", data=[[1,2,3], [4,5,6]])
    # instantiate from csv file
    for delimiter in [',', '\t', ' ', ':', ';', '|']:
        with tempfile.NamedTemporaryFile(mode='w') as tmpCSV:
            tmpCSV.write("1{0}2{0}3\n".format(delimiter))
            tmpCSV.write("4{0}5{0}6\n".format(delimiter))
            tmpCSV.flush()

            fromCSV = nimble.createData("Matrix", data=tmpCSV.name, inputSeparator=delimiter)
            assert fromCSV == wanted

@raises(FileFormatException)
def test_createData_csv_inputSeparatorConfusion():
    with tempfile.NamedTemporaryFile(mode='w') as tmpCSV:
        tmpCSV.write("1,2;3\n")
        tmpCSV.write("4,5,6\n")
        tmpCSV.flush()

        fromCSV = nimble.createData("Matrix", data=tmpCSV.name)

@raises(InvalidArgumentValue)
def test_createData_csv_inputSeparatorNot1Character():
    with tempfile.NamedTemporaryFile(mode='w') as tmpCSV:
        tmpCSV.write("1,,2,,3\n")
        tmpCSV.write("4,,5,,6\n")
        tmpCSV.flush()

        fromCSV = nimble.createData("Matrix", data=tmpCSV.name, inputSeparator=',,')


#########################################
# treatAsMissing and replaceMissingWith #
#########################################

def test_missingDefaults():
    for t in returnTypes:
        nan = numpy.nan
        data = [[1, 2, float('nan')], [numpy.nan, 5, 6], [7, None, 9], ["", "nan", "None"]]
        toTest = nimble.createData(t, data)
        expData = [[1, 2, nan], [nan, 5, 6], [7, nan, 9], [nan, nan, nan]]
        expRet = nimble.createData(t, expData)
        assert toTest == expRet

def test_handmadeReplaceMissingWith():
    for t in returnTypes:
        data = [[1, 2, float('nan')], [numpy.nan, 5, 6], [7, None, 9], ["", "nan", "None"]]
        toTest = nimble.createData(t, data, replaceMissingWith=0)
        expData = [[1, 2, 0], [0, 5, 6], [7, 0, 9], [0, 0, 0]]
        expRet = nimble.createData(t, expData)
        assert toTest == expRet

def test_numericalReplaceMissingWithNonNumeric():
    for t in returnTypes:
        data = [[1, 2, None], [None, 5, 6], [7, None, 9], [None, None, None]]
        toTest = nimble.createData(t, data, replaceMissingWith="Missing")
        expData = [[1, 2, "Missing"], ["Missing", 5, 6], [7, "Missing", 9], ["Missing", "Missing", "Missing"]]
        expRet = nimble.createData(t, expData)
        assert toTest == expRet

def test_handmadeTreatAsMissing():
    for t in returnTypes:
        nan = numpy.nan
        data = [[1, 2, ""], [nan, 5, 6], [7, "", 9], [nan, "nan", "None"]]
        missingList = [nan, "", 5]
        assert numpy.array(missingList).dtype != numpy.object_
        toTest = nimble.createData(t, data, treatAsMissing=missingList)
        expData = [[1, 2, nan], [nan, nan, 6], [7, nan, 9], [nan, "nan", "None"]]
        expRet = nimble.createData(t, expData, treatAsMissing=None)
        assert toTest == expRet

def test_handmadeConsiderAndReplaceMissingWith():
    for t in returnTypes:
        data = [[1, 2, "NA"], ["NA", 5, 6], [7, "NA", 9], ["NA", "NA", "NA"]]
        toTest = nimble.createData(t, data, treatAsMissing=["NA"], replaceMissingWith=0)
        expData = [[1, 2, 0], [0, 5, 6], [7, 0, 9], [0, 0, 0]]
        expRet = nimble.createData(t, expData)
        assert toTest == expRet

def test_replaceDataTypeMismatch():
    for t in returnTypes:
        data = [[1, 2, 99], [99, 5, 6], [7, 99, 9], [99, 99, 99]]
        toTest = nimble.createData(t, data, treatAsMissing=[99], replaceMissingWith="")
        expData = [[1, 2, ""], ["", 5, 6], [7, "", 9], ["", "", ""]]
        expRet = nimble.createData(t, expData, treatAsMissing=None)
        assert toTest == expRet

def test_keepNanAndReplaceAlternateMissing():
    for t in returnTypes:
        nan = numpy.nan
        data = [[1, 2, "NA"], [numpy.nan, 5, 6], [7, "NA", 9], ["NA", numpy.nan, "NA"]]
        toTest = nimble.createData(t, data, treatAsMissing=["NA"], replaceMissingWith=-1)
        expData = [[1, 2, -1], [nan, 5, 6], [7, -1, 9], [-1, nan, -1]]
        expRet = nimble.createData(t, expData, treatAsMissing=None)
        assert toTest == expRet

def test_treatAsMissingIsNone():
    for t in returnTypes:
        nan = numpy.nan
        data = [[1, 2, None], [None, 5, 6], [7, None, 9], ["", numpy.nan, ""]]
        toTest = nimble.createData(t, data, treatAsMissing=None)
        notExpData = [[1,2, nan], [nan, 5, 6], [7, nan, 9], [nan, nan, nan]]
        notExpRet = nimble.createData(t, notExpData, treatAsMissing=None, elementType=object)
        assert toTest != notExpRet

def test_DataOutputWithMissingDataTypes1D():
    for t in returnTypes:
        nan = numpy.nan
        expListOutput = [[1.0, 2.0, nan]]
        expMatrixOutput = numpy.matrix(expListOutput)
        expDataFrameOutput = pd.DataFrame(expListOutput)
        expSparseOutput = scipy.sparse.coo_matrix(expListOutput)

        orig1 = nimble.createData(t, [1,2,"None"])
        orig2 = nimble.createData(t, (1,2,"None"))
        orig3 = nimble.createData(t, {'a':1, 'b':2, 'c':"None"})
        orig3.features.sort(sortBy=orig3.points.getName(0))
        orig4 = nimble.createData(t, [{'a':1, 'b':2, 'c':"None"}])
        orig4.features.sort(sortBy=orig4.points.getName(0))
        orig5 = nimble.createData(t, numpy.array([1,2,"None"]))
        orig6 = nimble.createData(t, numpy.matrix([1,2,"None"]))
        if pd:
            orig7 = nimble.createData(t, pd.DataFrame([[1,2,"None"]]))
            orig8 = nimble.createData(t, pd.Series([1,2,"None"]))
            orig9 = nimble.createData(t, pd.SparseDataFrame([[1,2,"None"]]))
        if scipy:
            orig10 = nimble.createData(t, scipy.sparse.coo_matrix(numpy.array([1,2,"None"], dtype=object)))
            orig11 = nimble.createData(t, scipy.sparse.csc_matrix(numpy.array([1,2,float('nan')])))
            orig12 = nimble.createData(t, scipy.sparse.csr_matrix(numpy.array([1,2,float('nan')])))

        originals = [orig1, orig2, orig3, orig4, orig5, orig6, orig7, orig8, orig9, orig10, orig11, orig12]

        for orig in originals:
            if orig.getTypeString() == "List":
                assert orig.data[0][0] == expListOutput[0][0]
                assert orig.data[0][1] == expListOutput[0][1]
                assert numpy.isnan(orig.data[0][2])
            elif orig.getTypeString() == "Matrix":
                assert numpy.array_equal(orig.data[0, :2], expMatrixOutput[0, :2])
                assert numpy.isnan(orig.data[0, 2])
            elif orig.getTypeString() == "DataFrame":
                assert numpy.array_equal(orig.data.values[0, :2], expDataFrameOutput.values[0, :2])
                assert numpy.isnan(orig.data.values[0, 2])
            else:
                orig._sortInternal('point')
                assert numpy.array_equal(orig.data.data[:2], expSparseOutput.data[:2])
                assert numpy.isnan(orig.data.data[2])

def test_DataOutputWithMissingDataTypes2D():
    for t in returnTypes:
        nan = numpy.nan
        expListOutput = [[1, 2, nan], [3,4,'b']]
        expMatrixOutput = numpy.matrix(expListOutput, dtype=object)
        expDataFrameOutput = pd.DataFrame(expMatrixOutput)
        expSparseOutput = scipy.sparse.coo_matrix(expMatrixOutput)

        orig1 = nimble.createData(t, [[1,2,'None'], [3,4,'b']])
        orig2 = nimble.createData(t, ((1,2,'None'), (3,4,'b')))
        orig3 = nimble.createData(t, {'a':[1,3], 'b':[2,4], 'c':['None', 'b']}, elementType=object)
        orig3.features.sort(sortBy=orig3.points.getName(0))
        orig4 = nimble.createData(t, [{'a':1, 'b':2, 'c':'None'}, {'a':3, 'b':4, 'c':'b'}], elementType=object)
        orig4.features.sort(sortBy=orig4.points.getName(0))
        orig5 = nimble.createData(t, numpy.array([[1,2,'None'], [3,4,'b']], dtype=object))
        orig6 = nimble.createData(t, numpy.matrix([[1,2,'None'], [3,4,'b']], dtype=object))
        if pd:
            orig7 = nimble.createData(t, pd.DataFrame([[1,2,'None'], [3,4,'b']]))
            orig8 = nimble.createData(t, pd.SparseDataFrame([[1,2,'None'], [3,4,'b']]))
        if scipy:
            orig9 = nimble.createData(t, scipy.sparse.coo_matrix(numpy.array([[1,2,'None'], [3,4,'b']], dtype=object)))

        originals = [orig1, orig2, orig3, orig4, orig5, orig6, orig7, orig8, orig9]
        for orig in originals:
            if orig.getTypeString() == "List":
                assert orig.data[0][0] == expListOutput[0][0]
                assert orig.data[0][1] == expListOutput[0][1]
                assert numpy.isnan(orig.data[0][2])
                assert orig.data[1] == expListOutput[1]
            elif orig.getTypeString() == "Matrix":
                assert numpy.array_equal(orig.data[0, :2], expMatrixOutput[0, :2])
                assert numpy.isnan(orig.data[0, 2])
                assert numpy.array_equal(orig.data[1,:], expMatrixOutput[1,:])
            elif orig.getTypeString() == "DataFrame":
                assert numpy.array_equal(orig.data.values[0, :2], expDataFrameOutput.values[0, :2])
                assert numpy.isnan(orig.data.values[0, 2])
                assert numpy.array_equal(orig.data.values[1,:], expDataFrameOutput.values[1,:])
            else:
                orig._sortInternal('point')
                assert numpy.array_equal(orig.data.data[:2], expSparseOutput.data[:2])
                assert numpy.isnan(orig.data.data[2])
                assert numpy.array_equal(orig.data.data[3:], expSparseOutput.data[3:])


###################
### Other tests ###
###################

def test_createData_csv_nonremoval_efficiency():
    # setup the test fail trigger by replacing the function we don't want to
    # called

    def failFunction(arg1, arg2, arg3, arg4):
        assert False  # the function we didn't want to be called was called

    for t in returnTypes:
        fromList = nimble.createData(returnType=t, data=[[1, 2, 3]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.flush()
            objName = 'fromCSV'

            try:
                backup = nimble.helpers._removalCleanupAndSelectionOrdering
                nimble.helpers._removalCleanupAndSelectionOrdering = failFunction
                fromCSV = nimble.createData(returnType=t, data=tmpCSV.name, name=objName)
                assert fromList == fromCSV
            finally:
                nimble.helpers._removalCleanupAndSelectionOrdering = backup


#################
# Logging count #
#################
def test_createData_logCount():
    """Test createData adds one entry to the log for each return type"""

    @oneLogEntryExpected
    def byType(rType):
        toTest = nimble.createData(rType, [[1,2,3], [4,5,6], [7,8,9]])

    for t in returnTypes:
        byType(t)


# tests for combination of one name set being specified and one set being
# in data.


# test that if both in comment and specified names are present, the
# specified names win out.


# unit tests demonstrating our file loaders can handle arbitrarly placed blank lines


# comment lines
