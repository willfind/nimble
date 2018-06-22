from nose.tools import *
from nose.plugins.attrib import attr

import tempfile
import numpy
import os
import copy
import itertools

import UML
from UML.exceptions import ArgumentException
from UML.exceptions import FileFormatException
from UML.data.dataHelpers import DEFAULT_PREFIX
from UML.helpers import _intFloatOrString
scipy = UML.importModule('scipy.sparse')

#returnTypes = ['Matrix', 'Sparse', None]  # None for auto
returnTypes = copy.copy(UML.data.available)
returnTypes.append(None)

###########################
# Data values correctness #
###########################

def test_createData_CSV_data():
    """ Test of createData() loading a csv file, default params """
    for t in returnTypes:
        fromList = UML.createData(returnType=t, data=[[1, 2, 3]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.flush()
            objName = 'fromCSV'
            fromCSV = UML.createData(returnType=t, data=tmpCSV.name, name=objName)

            assert fromList == fromCSV


def test_createData_CSV_dataRandomExtension():
    """ Test of createData() loading a csv file without csv extension """
    for t in returnTypes:
        fromList = UML.createData(returnType=t, data=[[1, 2, 3]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".foo", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.flush()
            objName = 'fromCSV'
            fromCSV = UML.createData(returnType=t, data=tmpCSV.name, name=objName)

            assert fromList == fromCSV


def test_createData_CSV_data_noComment():
    for t in returnTypes:
        fromList = UML.createData(returnType=t, data=[[1, 2], [1, 2]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,#3\n")
            tmpCSV.write("1,2,3\n")
            tmpCSV.flush()
            objName = 'fromCSV'
            fromCSV = UML.createData(returnType=t, data=tmpCSV.name, name=objName, ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV


def test_createData_CSV_data_ListOnly():
    fromList = UML.createData(returnType="List", data=[[1, 2, 'three'], [4, 5, 'six']])

    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,three\n")
        tmpCSV.write("4,5,six\n")
        tmpCSV.flush()
        objName = 'fromCSV'
        fromCSV = UML.createData(returnType="List", data=tmpCSV.name, name=objName)

        assert fromList == fromCSV


def test_createData_CSV_data_ListOnly_noComment():
    fromList = UML.createData(returnType="List", data=[[1, 2, 'three'], [4, 5, '#six']])

    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,three\n")
        tmpCSV.write("4,5,#six\n")
        tmpCSV.flush()
        objName = 'fromCSV'
        fromCSV = UML.createData(returnType="List", data=tmpCSV.name, name=objName)

        assert fromList == fromCSV


def test_createData_MTXArr_data():
    """ Test of createData() loading a mtx (arr format) file, default params """
    for t in returnTypes:
        fromList = UML.createData(returnType=t, data=[[1, 2, 3]])

        # instantiate from mtx array file
        with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXArr:
            tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
            tmpMTXArr.write("1 3\n")
            tmpMTXArr.write("1\n")
            tmpMTXArr.write("2\n")
            tmpMTXArr.write("3\n")
            tmpMTXArr.flush()
            objName = 'fromMTXArr'
            fromMTXArr = UML.createData(returnType=t, data=tmpMTXArr.name, name=objName)

            if t is None and fromList.getTypeString() != fromMTXArr.getTypeString():
                assert fromList.isApproximatelyEqual(fromMTXArr)
            else:
                assert fromList == fromMTXArr

def test_createData_MTXArr_dataRandomExtension():
    """ Test of createData() loading a mtx (arr format) file without mtx extension """
    for t in returnTypes:
        fromList = UML.createData(returnType=t, data=[[1, 2, 3]])

        # instantiate from mtx array file
        with tempfile.NamedTemporaryFile(suffix=".foo", mode='w') as tmpMTXArr:
            tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
            tmpMTXArr.write("1 3\n")
            tmpMTXArr.write("1\n")
            tmpMTXArr.write("2\n")
            tmpMTXArr.write("3\n")
            tmpMTXArr.flush()
            objName = 'fromMTXArr'
            fromMTXArr = UML.createData(returnType=t, data=tmpMTXArr.name, name=objName)

            if t is None and fromList.getTypeString() != fromMTXArr.getTypeString():
                assert fromList.isApproximatelyEqual(fromMTXArr)
            else:
                assert fromList == fromMTXArr


def test_createData_MTXCoo_data():
    """ Test of createData() loading a mtx (coo format) file, default params """
    for t in returnTypes:
        fromList = UML.createData(returnType=t, data=[[1, 2, 3]])

        # instantiate from mtx coordinate file
        with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXCoo:
            tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.write("1 1 1\n")
            tmpMTXCoo.write("1 2 2\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.flush()
            objName = 'fromMTXCoo'
            fromMTXCoo = UML.createData(returnType=t, data=tmpMTXCoo.name, name=objName)

            if t is None and fromList.getTypeString() != fromMTXCoo.getTypeString():
                assert fromList.isApproximatelyEqual(fromMTXCoo)
            else:
                assert fromList == fromMTXCoo

def test_createData_MTXCoo_dataRandomExtension():
    """ Test of createData() loading a mtx (coo format) file without mtx extension """
    for t in returnTypes:
        fromList = UML.createData(returnType=t, data=[[1, 2, 3]])

        # instantiate from mtx coordinate file
        with tempfile.NamedTemporaryFile(suffix=".foo", mode='w') as tmpMTXCoo:
            tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.write("1 1 1\n")
            tmpMTXCoo.write("1 2 2\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.flush()
            objName = 'fromMTXCoo'
            fromMTXCoo = UML.createData(returnType=t, data=tmpMTXCoo.name, name=objName)

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

        UML.createData(returnType="List", data=tmpCSV.name)

@raises(FileFormatException)
def test_createData_CSV_unequalRowLength_long():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("4,5,6,7\n")
        tmpCSV.flush()

        UML.createData(returnType="List", data=tmpCSV.name)


@raises(FileFormatException)
def test_createData_CSV_unequalRowLength_definedByNames():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("one,two,three\n")
        tmpCSV.write("11,22,33,44\n")
        tmpCSV.write("4,5,6,7\n")
        tmpCSV.flush()

        UML.createData(returnType="List", data=tmpCSV.name, featureNames=True)


def test_createData_CSV_unequalRowLength_position():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("#ignore\n")
        tmpCSV.write("1,2,3,4,0,0,0,0\n")
        tmpCSV.write("\n")
        tmpCSV.write("11,22,33,44,0,0,0,0\n")
        tmpCSV.write("4,5,6,0,0,0\n")
        tmpCSV.flush()

        try:
            UML.createData(returnType="List", data=tmpCSV.name, featureNames=True)
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
            ret = UML.createData(returnType=t, data=tmpCSV.name, name=objName)
            assert ret.name == objName
            assert ret.path == tmpCSV.name
            assert ret.absolutePath == tmpCSV.name

            relExp = os.path.relpath(ret.absolutePath)
            assert ret.relativePath == relExp

            retDefName = UML.createData(returnType=t, data=tmpCSV.name)
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
            ret = UML.createData(returnType=t, data=tmpMTXArr.name, name=objName)
            assert ret.name == objName
            assert ret.path == tmpMTXArr.name
            assert ret.absolutePath == tmpMTXArr.name

            relExp = os.path.relpath(ret.absolutePath)
            assert ret.relativePath == relExp

            retDefName = UML.createData(returnType=t, data=tmpMTXArr.name)
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
            ret = UML.createData(returnType=t, data=tmpMTXCoo.name, name=objName)
            assert ret.name == objName
            assert ret.path == tmpMTXCoo.name
            assert ret.absolutePath == tmpMTXCoo.name

            relExp = os.path.relpath(ret.absolutePath)
            assert ret.relativePath == relExp

            retDefName = UML.createData(returnType=t, data=tmpMTXCoo.name)
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
        fromList = UML.createData(
            returnType=t, data=[[1, 2, 3]], pointNames=pNames, featureNames=fNames)

        # instantiate from csv file
        tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv", mode='w')
        tmpCSV.write('ignore,one,two,three\n')
        tmpCSV.write("pn1,1,2,3\n")
        tmpCSV.flush()

        fromCSV = UML.createData(
            returnType=t, data=tmpCSV.name, pointNames=True, featureNames=True)
        tmpCSV.close()
        assert fromList == fromCSV


def test_names_AutoDetectedBlankLines_CSV():
    pNames = ['pn1']
    fNames = ['one', 'two', 'three']
    for t in returnTypes:
        fromList = UML.createData(
            returnType=t, data=[[1, 2, 3]], pointNames=pNames, featureNames=fNames)

        # instantiate from csv file
        tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv", mode='w')
        tmpCSV.write("\n")
        tmpCSV.write("\n")
        tmpCSV.write("point_names,one,two,three\n")
        tmpCSV.write("pn1,1,2,3\n")
        tmpCSV.flush()

        fromCSV = UML.createData(returnType=t, data=tmpCSV.name)
        tmpCSV.close()
        assert fromList == fromCSV


def test_featNamesOnly_AutoDetectedBlankLines_CSV():
    fNames = ['one', 'two', 'three']
    for t in returnTypes:
        fromList = UML.createData(returnType=t, data=[[1, 2, 3]], featureNames=fNames)

        # instantiate from csv file
        tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv", mode='w')
        tmpCSV.write("\n")
        tmpCSV.write("\n")
        tmpCSV.write("one,two,three\n")
        tmpCSV.write("1,2,3\n")
        tmpCSV.flush()

        fromCSV = UML.createData(returnType=t, data=tmpCSV.name)
        tmpCSV.close()
        assert fromList == fromCSV

def test_pointNames_AutoDetected_from_specified_featNames_CSV():
    fNames = ['one', 'two', 'three']
    pNames = ['pn1']
    for t in returnTypes:
        fromList = UML.createData(
            returnType=t, data=[[1, 2, 3]], pointNames=pNames, featureNames=fNames)

        # instantiate from csv file
        tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv", mode='w')
        tmpCSV.write("\n")
        tmpCSV.write("\n")
        tmpCSV.write("point_names,one,two,three\n")
        tmpCSV.write("pn1,1,2,3\n")
        tmpCSV.flush()
        fromCSV = UML.createData(returnType=t, data=tmpCSV.name, featureNames=True)
        tmpCSV.close()
        assert fromList == fromCSV


def test_specifiedIgnore_overides_autoDetectBlankLine_CSV():
    for t in returnTypes:
        data = [[0, 1, 2, 3], [10, 11, 12, 13]]
        fromList = UML.createData(returnType=t, data=data)

        # instantiate from csv file
        tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv", mode='w')
        tmpCSV.write("\n")
        tmpCSV.write("\n")
        tmpCSV.write('0,1,2,3\n')
        tmpCSV.write("10,11,12,13\n")
        tmpCSV.flush()
        fromCSV = UML.createData(
            returnType=t, data=tmpCSV.name, pointNames=False, featureNames=False)
        tmpCSV.close()
        assert fromList == fromCSV


def helper_auto(rawStr, rawType, returnType, pointNames, featureNames):
    """
    Writes a CSV and reads it in using UML.createData, fixing in place the given arguments,
    returning the resultant object

    """
    if rawType == 'csv':
        tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv", mode='w')
        tmpCSV.write(rawStr)
        tmpCSV.flush()
        ret = UML.createData(returnType=returnType, data=tmpCSV.name,
                             pointNames=pointNames, featureNames=featureNames)
        tmpCSV.close()
    else:
        fnameRow = list(map(_intFloatOrString, rawStr.split('\n')[0].split(',')))
        dataRow = list(map(_intFloatOrString, rawStr.split('\n')[1].split(',')))
        lolFromRaw = [fnameRow, dataRow]
        baseObj = UML.createData("List", lolFromRaw, pointNames=False, featureNames=False)
        if rawType == 'scipycoo':
            npRaw = numpy.array(lolFromRaw, dtype=object)
            finalRaw = scipy.sparse.coo_matrix(npRaw)
        else:
            finalRaw = baseObj.copyAs(rawType)

        ret = UML.createData(returnType=returnType, data=finalRaw,
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
        assert correct.getFeatureNames() == ['fname0','fname1','fname2']

        # example where first line contains a non-string interpretable value
        nonStringFail1Raw = "fname0,1.0,fname2\n1,2,3"
        fail1 = helper_auto(nonStringFail1Raw, rawT, retT, pointNames='automatic', featureNames='automatic')
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), fail1.getFeatureNames()))

        # example where the first line contains all strings, but second line also contains strings
        sameTypeFail2Raw = "fname0,fname1,fname2\n1,data2,3"
        fail2 = helper_auto(sameTypeFail2Raw, rawT, retT, pointNames='automatic', featureNames='automatic')
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), fail2.getFeatureNames()))


def test_userOverrideOfAutomaticByType_fnames_rawAndCSV():
    availableRaw = ['csv', 'pythonlist', 'numpyarray', 'numpymatrix', 'scipycoo']
    for (rawT, retT) in itertools.product(availableRaw, returnTypes):
        # example where user provided False overides automatic detection
        correctRaw = "fname0,fname1,fname2\n1,2,3\n"
        overide1 = helper_auto(correctRaw, rawT, retT, pointNames='automatic', featureNames=False)
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), overide1.getFeatureNames()))

        # example where user provided True extracts non-detectable first line
        nonStringFail1Raw = "fname0,1.0,fname2\n1,2,3"
        overide2 = helper_auto(nonStringFail1Raw, rawT, retT, pointNames='automatic', featureNames=True)
        assert overide2.getFeatureNames() == ['fname0', '1.0', 'fname2']

        # example where user provided True extracts non-detectable first line
        sameTypeFail2Raw = "fname0,fname1,fname2\ndata1,data2,data3"
        overide3 = helper_auto(sameTypeFail2Raw, rawT, retT, pointNames='automatic', featureNames=True)
        assert overide3.getFeatureNames() == ['fname0', 'fname1', 'fname2']


def test_automaticByType_pname_interaction_with_fname():
    availableRaw = ['csv', 'pythonlist', 'numpyarray', 'numpymatrix', 'scipycoo']
    for (rawT, retT) in itertools.product(availableRaw, returnTypes):
#        rawT = 'scipycoo'
#        retT = None
#        print rawT + " " + str(retT)
#        import pdb
#        pdb.set_trace()

        # pnames auto triggered with auto fnames
        raw = "point_names,fname0,fname1,fname2\npname0,1,2,3\n"
        testObj = helper_auto(raw, rawT, retT, pointNames='automatic', featureNames='automatic')
        assert testObj.getFeatureNames() == ['fname0','fname1','fname2']
        assert testObj.getPointNames() == ['pname0']

        # pnames auto triggereed with explicit fnames
        raw = "point_names,fname0,fname1,fname2\npname0,1,2,3\n"
        testObj = helper_auto(raw, rawT, retT, pointNames='automatic', featureNames=True)
        assert testObj.getFeatureNames() == ['fname0','fname1','fname2']
        assert testObj.getPointNames() == ['pname0']

        #pnames not triggered given 'point_names' at [0,0] when fnames auto trigger fails CASE1
        raw = "point_names,fname0,1.0,fname2\npname0,1,2,3\n"
        testObj = helper_auto(raw, rawT, retT, pointNames='automatic', featureNames='automatic')
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), testObj.getFeatureNames()))
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), testObj.getPointNames()))

        #pnames not triggered given 'point_names' at [0,0] when fnames auto trigger fails CASE2
        raw = "point_names,fname0,fname1,fname2\npname0,data1,data2,data3\n"
        testObj = helper_auto(raw, rawT, retT, pointNames='automatic', featureNames='automatic')
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), testObj.getFeatureNames()))
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), testObj.getPointNames()))

        #pnames not triggered given 'point_names' at [0,0] when fnames explicit False
        raw = "point_names,fname0,fname1,fname2\npname0,1,2,3\n"
        testObj = helper_auto(raw, rawT, retT, pointNames='automatic', featureNames=False)
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), testObj.getFeatureNames()))
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), testObj.getPointNames()))

        #pnames explicit False given 'point_names' at [0,0] and fname auto extraction
        raw = "point_names,fname0,fname1,fname2\npname0,1,2,3\n"
        testObj = helper_auto(raw, rawT, retT, pointNames=False, featureNames=True)
        assert testObj.getFeatureNames() == ['point_names', 'fname0', 'fname1', 'fname2']
        assert all(map(lambda x: x.startswith(DEFAULT_PREFIX), testObj.getPointNames()))


def test_namesInComment_MTXArr():
    """ Test of createData() loading a mtx (arr format) file and comment Names """
    pNames = ['pn1']
    fNames = ['one', 'two', 'three']
    for t in returnTypes:
        fromList = UML.createData(returnType=t, data=[[1, 2, 3]], pointNames=pNames, featureNames=fNames)

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
        fromMTXArr = UML.createData(returnType=t, data=tmpMTXArr.name)
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
        fromList = UML.createData(
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
        fromMTXCoo = UML.createData(returnType=t, data=tmpMTXCoo.name)
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
        fromList = UML.createData(
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

        fromMTXArr = UML.createData(
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
        fromList = UML.createData(
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
        fromMTXCoo = UML.createData(
            returnType=t, data=tmpMTXCoo.name, pointNames=True, featureNames=True)
        tmpMTXCoo.close()
        if t is None and fromList.getTypeString() != fromMTXCoo.getTypeString():
            assert fromList.isApproximatelyEqual(fromMTXCoo)
        else:
            assert fromList == fromMTXCoo


@raises(ArgumentException)
def test_csv_extractNames_duplicatePointName():
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write('ignore,one,two,three\n')
        tmpCSV.write("pn1,1,2,3\n")
        tmpCSV.write("pn1,11,22,33\n")
        tmpCSV.flush()

        UML.createData(returnType="List", data=tmpCSV.name, pointNames=True)


@raises(ArgumentException)
def test_csv_extractNames_duplicateFeatureName():
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write('one,two,one\n')
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.flush()

        UML.createData(returnType="List", data=tmpCSV.name, featureNames=True)


def test_csv_roundtrip_autonames():
    for retType in returnTypes:
        data = [[1, 0, 5, 12], [0, 1, 3, 17], [0, 0, 8, 22]]
        pnames = ['p0','p1','p2']
        fnames = ['f0','f1','f2', 'f3']

        withFnames = UML.createData(retType, data, featureNames=fnames)
        withBoth = UML.createData(retType, data, featureNames=fnames, pointNames=pnames)

        with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSVFnames:
            withFnames.writeFile(tmpCSVFnames.name, 'csv', includeNames=True)
            fromFileFnames = UML.createData(returnType=retType, data=tmpCSVFnames.name)
            assert fromFileFnames == withFnames

        with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSVBoth:
            withBoth.writeFile(tmpCSVBoth.name, 'csv', includeNames=True)
            fromFileBoth = UML.createData(returnType=retType, data=tmpCSVBoth.name)
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

        inData = UML.createData(
            returnType=t, data=inDataRaw, pointNames=True, featureNames=True)
        specified = UML.createData(
            returnType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
        assert inData == specified


def test_extractNames_NPArray():
    """ Test of createData() given numpy array, extracting names """
    pNames = ['11']
    fNames = ['21', '22', '23']

    for t in returnTypes:
        inDataRaw = numpy.array([[-111, 21, 22, 23], [11, 1, -1, -3]])
        specRaw = numpy.array([[1, -1, -3]])
        inData = UML.createData(
            returnType=t, data=inDataRaw, pointNames=True, featureNames=True)
        specified = UML.createData(
            returnType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
        assert inData == specified


def test_extractNames_NPMatrix():
    """ Test of createData() given numpy matrix, extracting names """
    pNames = ['11']
    fNames = ['21', '22', '23']

    for t in returnTypes:
        inDataRaw = numpy.array([[-111, 21, 22, 23], [11, 1, -1, -3]])
        specRaw = numpy.matrix([[1, -1, -3]])
        inData = UML.createData(
            returnType=t, data=inDataRaw, pointNames=True, featureNames=True)
        specified = UML.createData(
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

        inData = UML.createData(
            returnType=t, data=inDataRaw, pointNames=True, featureNames=True)
        specified = UML.createData(
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

        inData = UML.createData(
            returnType=t, data=inDataRaw, pointNames=True, featureNames=True)
        specified = UML.createData(
            returnType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
        assert inData == specified


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
        fromList = UML.createData(returnType=t, data=[[1, 2, 3]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.flush()
            objName = 'fromCSV'
            openFile = open(tmpCSV.name, 'rU')
            fromCSV = UML.createData(returnType=t, data=openFile, name=objName)
            openFile.close()

            assert fromList == fromCSV

            assert fromCSV.path == openFile.name
            assert fromCSV.absolutePath == openFile.name
            assert fromCSV.relativePath == os.path.relpath(openFile.name)

            openFile = open(openFile.name, 'rU')
            namelessOpenFile = NamelessFile(openFile)
            fromCSV = UML.createData(
                returnType=t, data=namelessOpenFile)
            assert fromCSV.name.startswith(UML.data.dataHelpers.DEFAULT_NAME_PREFIX)
            assert fromCSV.path is None
            assert fromCSV.absolutePath is None
            assert fromCSV.relativePath is None


def test_createData_MTXArr_passedOpen():
    for t in returnTypes:
        fromList = UML.createData(returnType=t, data=[[1, 2, 3]])

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
            fromMTXArr = UML.createData(returnType=t, data=openFile, name=objName)
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
            fromMTXArr = UML.createData(
                returnType=t, data=namelessOpenFile)
            assert fromMTXArr.name.startswith(
                UML.data.dataHelpers.DEFAULT_NAME_PREFIX)
            assert fromMTXArr.path is None
            assert fromMTXArr.absolutePath is None
            assert fromMTXArr.relativePath is None


def test_createData_MTXCoo_passedOpen():
    for t in returnTypes:
        fromList = UML.createData(returnType=t, data=[[1, 2, 3]])

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
            fromMTXCoo = UML.createData(returnType=t, data=openFile, name=objName)
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
            fromMTXCoo = UML.createData(
                returnType=t, data=namelessOpenFile)
            assert fromMTXCoo.name.startswith(
                UML.data.dataHelpers.DEFAULT_NAME_PREFIX)
            assert fromMTXCoo.path is None
            assert fromMTXCoo.absolutePath is None
            assert fromMTXCoo.relativePath is None


###################################
# ignoreNonNumericalFeatures flag #
###################################

def test_createData_ignoreNonNumericalFeaturesCSV():
    for t in returnTypes:
        fromList = UML.createData(returnType=t, data=[[1, 3], [5, 7]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,two,3.0,four\n")
            tmpCSV.write("5,six,7,8\n")
            tmpCSV.flush()

            fromCSV = UML.createData(
                returnType=t, data=tmpCSV.name, ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV

            # sanity check
            fromCSV = UML.createData(returnType=t, data=tmpCSV.name)
            assert fromCSV.features == 4


def test_createData_CSV_ignoreNonNumerical_removalCleanup_hard():
    for t in returnTypes:
        fromList = UML.createData(returnType=t, data=[[1, 3], [5, 7], [11, 12], [13, 14]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3.0,4.0,1\n")
            tmpCSV.write("5,six,7,8,1\n")
            tmpCSV.write("11,6,12,eight,1.0\n")
            tmpCSV.write("13,one,14,9,who?\n")
            tmpCSV.flush()

            fromCSV = UML.createData(
                returnType=t, data=tmpCSV.name, ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV

            # sanity check
            fromCSV = UML.createData(returnType=t, data=tmpCSV.name)
            assert fromCSV.features == 5


def test_createData_CSV_ignoreNonNumerical_removalCleanup_easy():
    for t in returnTypes:
        fromList = UML.createData(returnType=t, data=[[1, 3], [5, 7], [11, 12], [13, 14]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,two,3.0,four,one\n")
            tmpCSV.write("5,6,7,8,1\n")
            tmpCSV.write("11,6,12,8,1.0\n")
            tmpCSV.write("13,1,14,9,2\n")
            tmpCSV.flush()

            fromCSV = UML.createData(
                returnType=t, data=tmpCSV.name, ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV

            # sanity check
            fromCSV = UML.createData(returnType=t, data=tmpCSV.name)
            assert fromCSV.features == 5


def test_createData_ignoreNonNumericalFeaturesCSV_noEffect():
    for t in returnTypes:
        fromList = UML.createData(returnType=t, data=[[1, 2, 3, 4], [5, 6, 7, 8]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3,4\n")
            tmpCSV.write("5,6,7,8\n")
            tmpCSV.flush()

            fromCSV = UML.createData(
                returnType=t, data=tmpCSV.name, ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV

            fromCSV = UML.createData(returnType=t, data=tmpCSV.name)
            assert fromCSV.features == 4


def test_CSV_ignoreNonNumericalFeatures_featureNamesDontTrigger():
    for t in returnTypes:
        fnames = ['1', '2', '3', 'four']
        fromList = UML.createData(returnType=t, featureNames=fnames, data=[[5, 6, 7, 8]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3,four\n")
            tmpCSV.write("5,6,7,8\n")
            tmpCSV.flush()

            fromCSV = UML.createData(
                returnType=t, data=tmpCSV.name, featureNames=True,
                ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV


def test_CSV_ignoreNonNumericalFeatures_featureNamesAdjusted():
    for t in returnTypes:
        fNames = ["1", "2", "3"]
        data = [[1, 2, 3], [5, 6, 7]]
        fromList = UML.createData(returnType=t, featureNames=fNames, data=data)

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3,4\n")
            tmpCSV.write("1,2,3,four\n")
            tmpCSV.write("5,6,7,H8\n")
            tmpCSV.flush()

            fromCSV = UML.createData(
                returnType=t, data=tmpCSV.name, featureNames=True,
                ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV


def test_CSV_createData_ignoreNonNumericalFeatures_allRemoved():
    for t in returnTypes:
        pNames = ['single', 'dubs', 'trips']
        fromList = UML.createData(returnType=t, pointNames=pNames, data=[[], [], []])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write(",ones,twos,threes\n")
            tmpCSV.write("single,1A,2A,3A\n")
            tmpCSV.write("dubs,11,22A,33\n")
            tmpCSV.write("trips,111,222,333\n")
            tmpCSV.flush()

            fromCSV = UML.createData(
                returnType=t, data=tmpCSV.name, pointNames=True,
                featureNames=True, ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV


####################################################
# Difficult CSV Formatting: whitespace and quoting #
####################################################

def test_CSVformatting_simpleQuotedValues():
    for t in returnTypes:
        fromList = UML.createData(returnType=t, data=[[1, 2, 3, 4], [5, 6, 7, 8]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,\"2\",\"3\",4\n")
            tmpCSV.write("5,\"6\",\"7\",8\n")
            tmpCSV.flush()

            fromCSV = UML.createData(returnType=t, data=tmpCSV.name)

            assert fromList == fromCSV


def test_CSVformatting_specialCharsInQuotes():
    for t in returnTypes:
        fNames = ["1,ONE", "2;TWO", "3\t'EE'"]
        data = [[1, 2, 3], [5, 6, 7]]
        dataAll = [[1, 2, 3, 4], [5, 6, 7, 8]]
        fromList = UML.createData(returnType=t, featureNames=fNames[:3], data=data)

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("\"1,ONE\",\"2;TWO\",\"3\t'EE'\",\"4f\"\n")
            tmpCSV.write("1,2,3,four\n")
            tmpCSV.write("5,6,7,H8\n")
            tmpCSV.flush()

            fromCSV = UML.createData(
                returnType=t, data=tmpCSV.name, featureNames=True,
                ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV


def test_CSVformatting_emptyAndCommentLines():
    for t in returnTypes:
        data = [[1, 2, 3, 4], ['#11', 22, 33, 44], [5, 6, 7, 8]]

        fromList = UML.createData(returnType=t, data=data)

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

            fromCSV = UML.createData(
                returnType=t, data=tmpCSV.name, featureNames=False)

            assert fromList == fromCSV


def test_CSVformatting_scientificNotation():
    for t in returnTypes:
        data = [[1., 2., 3.], [11., 22., 33.], [111., 222., 333.]]
        fromRaw = UML.createData(returnType=t, data=data)

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1.000000000e+00,2.000000000e+00,3.000000000e+00\n")
            tmpCSV.write("1.100000000e+01,2.200000000e+01,3.300000000e+01\n")
            tmpCSV.write("1.110000000e+02,2.220000000e+02,3.330000000e+02\n")
            tmpCSV.flush()

            fromCSV = UML.createData(returnType=t, data=tmpCSV.name)

            assert fromRaw == fromCSV


################################
# keepPoints, keepFeatures #
################################

def test_createData_keepPF_AllPossibleNatOrder():
    filesForms = ['csv', 'mtx']
    for (t, f) in itertools.product(returnTypes, filesForms):
        data = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
        orig = UML.createData(returnType=t, data=data)
        with tempfile.NamedTemporaryFile(suffix="." + f) as tmpF:
            orig.writeFile(tmpF.name, format=f, includeNames=False)
            tmpF.flush()

            poss = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2], 'all']
            for (pSel, fSel) in itertools.product(poss, poss):
                ret = UML.createData(
                    t, tmpF.name, keepPoints=pSel, keepFeatures=fSel)
                fromOrig = UML.createData(
                    t, orig.data, keepPoints=pSel, keepFeatures=fSel)

                assert ret == fromOrig


def test_createData_keepPF_AllPossibleReverseOrder():
    filesForms = ['csv', 'mtx']
    for (t, f) in itertools.product(returnTypes, filesForms):
        data = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
        orig = UML.createData(returnType=t, data=data)
        with tempfile.NamedTemporaryFile(suffix="." + f) as tmpF:
            orig.writeFile(tmpF.name, format=f, includeNames=False)
            tmpF.flush()

            poss = [[0, 1], [0, 2], [1, 2], [0, 1, 2]]
            for (pSel, fSel) in itertools.product(poss, poss):
                ret = UML.createData(
                    t, tmpF.name, keepPoints=pSel, keepFeatures=fSel)
                fromOrig = UML.createData(
                    t, orig.data, keepPoints=pSel, keepFeatures=fSel)

                assert ret == fromOrig

                pSelR = copy.copy(pSel)
                pSelR.reverse()
                fSelR = copy.copy(fSel)
                fSelR.reverse()

                retT = UML.createData(
                    t, tmpF.name, keepPoints=pSelR, keepFeatures=fSelR)
                fromOrigT = UML.createData(
                    t, orig.data, keepPoints=pSelR, keepFeatures=fSelR)

                assert retT != ret
                assert retT == fromOrigT
                assert fromOrigT != fromOrig


def test_createData_keepPF_AllPossibleWithNames_extracted():
    data = [[1., 2., 3.], [11., 22., 33.], [111., 222., 333.]]
    orig = UML.createData(returnType="List", data=data)
    filesForms = ['csv', 'mtx']
    for (t, f) in itertools.product(returnTypes, filesForms):
        with tempfile.NamedTemporaryFile(suffix="." + f) as tmpF:
            orig.writeFile(
                tmpF.name, format=f, includeNames=False)
            tmpF.flush()

            poss = [[0], [1], [0, 1], [1, 0], 'all']
            for (pSel, fSel) in itertools.product(poss, poss):
                toUse = orig.copyAs("pythonlist")
                fromOrig = UML.createData(
                    t, toUse, keepPoints=pSel, keepFeatures=fSel,
                    pointNames=True, featureNames=True)

                ret = UML.createData(
                    t, tmpF.name, keepPoints=pSel, keepFeatures=fSel,
                    pointNames=True, featureNames=True)

                pSelUse = copy.copy(pSel)
                fSelUse = copy.copy(fSel)
                if pSel != 'all':
                    for i in range(len(pSel)):
                        pSelUse[i] = ret.getPointName(i)

                if fSel != 'all':
                    for i in range(len(fSel)):
                        fSelUse[i] = ret.getFeatureName(i)

                retN = UML.createData(
                    t, tmpF.name, keepPoints=pSelUse,
                    keepFeatures=fSelUse, pointNames=True,
                    featureNames=True)

                assert ret == fromOrig
                assert retN == fromOrig


def test_createData_keepPF_AllPossibleWithNames_listProvided():
    pnames = ["11.", "111."]
    fnames = ["2.", "3."]
    data = [[22., 33.], [222., 333.]]
    orig = UML.createData(
        returnType="List", data=data, pointNames=pnames, featureNames=fnames)

    filesForms = ['csv', 'mtx']
    for (t, f) in itertools.product(returnTypes, filesForms):
        with tempfile.NamedTemporaryFile(suffix="." + f) as tmpF:
            orig.writeFile(
                tmpF.name, format=f, includeNames=False)
            tmpF.flush()

            poss = [[0], [1], [0, 1], [1, 0], 'all']
            for (pSel, fSel) in itertools.product(poss, poss):
                toUseData = orig.copyAs("pythonlist")

                fromOrig = UML.createData(
                    t, toUseData, keepPoints=pSel, keepFeatures=fSel,
                    pointNames=pnames, featureNames=fnames)

                ret = UML.createData(
                    t, tmpF.name, keepPoints=pSel, keepFeatures=fSel,
                    pointNames=pnames, featureNames=fnames)

                pSelUse = copy.copy(pSel)
                fSelUse = copy.copy(fSel)
                if pSel != 'all':
                    for i in range(len(pSel)):
                        pSelUse[i] = ret.getPointName(i)

                if fSel != 'all':
                    for i in range(len(fSel)):
                        fSelUse[i] = ret.getFeatureName(i)

                retN = UML.createData(
                    t, tmpF.name, keepPoints=pSelUse,
                    keepFeatures=fSelUse, pointNames=pnames,
                    featureNames=fnames)

                assert ret == fromOrig
                assert retN == fromOrig


def test_createData_keepPF_AllPossibleWithNames_dictProvided():
    pnames = {"11.": 0, "111.": 1}
    fnames = {"2.": 0, "3.": 1}
    data = [[22., 33.], [222., 333.]]
    orig = UML.createData(
        returnType="List", data=data, pointNames=pnames, featureNames=fnames)

    filesForms = ['csv', 'mtx']
    for (t, f) in itertools.product(returnTypes, filesForms):
        with tempfile.NamedTemporaryFile(suffix="." + f) as tmpF:
            orig.writeFile(
                tmpF.name, format=f, includeNames=False)
            tmpF.flush()

            poss = [[0], [1], [0, 1], [1, 0], 'all']
            for (pSel, fSel) in itertools.product(poss, poss):
                toUseData = orig.copyAs("pythonlist")

                fromOrig = UML.createData(
                    t, toUseData, keepPoints=pSel, keepFeatures=fSel,
                    pointNames=pnames, featureNames=fnames)

                ret = UML.createData(
                    t, tmpF.name, keepPoints=pSel, keepFeatures=fSel,
                    pointNames=pnames, featureNames=fnames)

                pSelUse = copy.copy(pSel)
                fSelUse = copy.copy(fSel)
                if pSel != 'all':
                    for i in range(len(pSel)):
                        pSelUse[i] = ret.getPointName(i)

                if fSel != 'all':
                    for i in range(len(fSel)):
                        fSelUse[i] = ret.getFeatureName(i)

                retN = UML.createData(
                    t, tmpF.name, keepPoints=pSelUse,
                    keepFeatures=fSelUse, pointNames=pnames,
                    featureNames=fnames)

                assert ret == fromOrig
                assert retN == fromOrig


def test_createData_csv_keepPoints_IndexingGivenFeatureNames():
    data = [[111, 222, 333]]
    fnames = ['1', '2', '3']
    wanted = UML.createData("Matrix", data=data, featureNames=fnames)
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        fromCSV = UML.createData(
            "Matrix", data=tmpCSV.name, keepPoints=[1], featureNames=True)

        raw = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
        fromRaw = UML.createData(
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
    wanted = UML.createData("List", data=[[22], [222]])
    backup = UML.helpers.initDataObject

    try:
        def fakeinitDataObject(
                returnType, rawData, pointNames, featureNames, name, path,
                keepPoints, keepFeatures):
            assert len(rawData) == 2
            assert len(rawData[0]) == 1
            return UML.data.List(rawData)

        UML.helpers.initDataObject = fakeinitDataObject

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.write("11,22,33\n")
            tmpCSV.write("111,222,333\n")
            tmpCSV.flush()

            fromCSV = UML.createData(
                "List", data=tmpCSV.name, keepPoints=[1, 2], keepFeatures=[1])
            assert fromCSV == wanted
    finally:
        UML.helpers.initDataObject = backup

#def TODOtest_createData_keepPF_mtxArr_noUncessaryStorage():
#	fromList = UML.createData(returnType='Matrix', data=[[2]])
#	backup = UML.helpers.initDataObject
#
#	try:
#		def fakeinitDataObject(
#				returnType, rawData, pointNames, featureNames, name, path,
#				keepPoints, keepFeatures):
#			assert len(rawData) == 1
#			assert len(rawData[0]) == 1
#			return UML.data.List(rawData)
#
#		UML.helpers.initDataObject = fakeinitDataObject
#
#		# instantiate from mtx array file
#		with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXArr:
#			tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
#			tmpMTXArr.write("1 3\n")
#			tmpMTXArr.write("1\n")
#			tmpMTXArr.write("2\n")
#			tmpMTXArr.write("3\n")
#			tmpMTXArr.flush()
#			fromMTXArr = UML.createData(
#				returnType='Matrix', data=tmpMTXArr.name, keepPoints=[0],
#				keepFeatures=[1])
#
#			assert fromList == fromMTXArr
#	finally:
#		UML.helpers.initDataObject = backup


#def TODOtest_createData_keepPF_mtxCoo_noUncessaryStorage():
#	fromList = UML.createData(returnType='Matrix', data=[[2]])
#	backup = UML.helpers.initDataObject
#
#	try:
#		def fakeinitDataObject(
#				returnType, rawData, pointNames, featureNames, name, path,
#				keepPoints, keepFeatures):
#			assert rawData.shape == (1,1)
#			return UML.data.List(rawData)
#
#		UML.helpers.initDataObject = fakeinitDataObject
#
#		# instantiate from mtx coordinate file
#		with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXCoo:
#			tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
#			tmpMTXCoo.write("1 3 3\n")
#			tmpMTXCoo.write("1 1 1\n")
#			tmpMTXCoo.write("1 2 2\n")
#			tmpMTXCoo.write("1 3 3\n")
#			tmpMTXCoo.flush()
#			fromMTXCoo = UML.createData(
#				returnType='Matrix', data=tmpMTXCoo.name, keepPoints=[0],
#				keepFeatures=[1])
#
#			assert fromList == fromMTXCoo
#
#	finally:
#		UML.helpers.initDataObject = backup


def test_createData_keepPF_csv_simple():
    wanted = UML.createData("Matrix", data=[[222], [22]])
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        fromCSV = UML.createData(
            "Matrix", data=tmpCSV.name, keepPoints=[2, 1], keepFeatures=[1])
        assert fromCSV == wanted


def test_createData_keepPF_mtxArr_simple():
    fromList = UML.createData(returnType='Matrix', data=[[2]])

    # instantiate from mtx array file
    with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXArr:
        tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
        tmpMTXArr.write("1 3\n")
        tmpMTXArr.write("1\n")
        tmpMTXArr.write("2\n")
        tmpMTXArr.write("3\n")
        tmpMTXArr.flush()
        fromMTXArr = UML.createData(
            returnType='Matrix', data=tmpMTXArr.name, keepPoints=[0],
            keepFeatures=[1])

        assert fromList == fromMTXArr


def test_createData_keepPF_mtxCoo_simple():
    fromList = UML.createData(returnType='Matrix', data=[[2]])

    # instantiate from mtx coordinate file
    with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXCoo:
        tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
        tmpMTXCoo.write("1 3 3\n")
        tmpMTXCoo.write("1 1 1\n")
        tmpMTXCoo.write("1 2 2\n")
        tmpMTXCoo.write("1 3 3\n")
        tmpMTXCoo.flush()
        fromMTXCoo = UML.createData(
            returnType='Matrix', data=tmpMTXCoo.name, keepPoints=[0],
            keepFeatures=[1])

        assert fromList == fromMTXCoo


def test_createData_keepPF_pythonList_simple():
    wanted = UML.createData("Matrix", data=[[22, 33], [222, 333]])
    raw = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]

    fromList = UML.createData(
        "Matrix", data=raw, keepPoints=[1, 2], keepFeatures=[1, 2])
    assert fromList == wanted

    wanted = UML.createData("Matrix", data=[[333, 222], [33, 22]])
    fromList = UML.createData(
        "Matrix", data=raw, keepPoints=[2, 1], keepFeatures=[2, 1])
    assert fromList == wanted


def test_createData_keepPF_npArray_simple():
    wanted = UML.createData("Matrix", data=[[22, 33], [222, 333]])
    rawList = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
    raw = numpy.array(rawList)

    fromNPArr = UML.createData(
        "Matrix", data=raw, keepPoints=[1, 2], keepFeatures=[1, 2])
    assert fromNPArr == wanted

    wanted = UML.createData("Matrix", data=[[333, 222], [33, 22]])
    fromNPArr = UML.createData(
        "Matrix", data=raw, keepPoints=[2, 1], keepFeatures=[2, 1])
    assert fromNPArr == wanted


def test_createData_keepPF_npMatrix_simple():
    wanted = UML.createData("Matrix", data=[[22, 33], [222, 333]])
    rawList = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
    raw = numpy.matrix(rawList)

    fromList = UML.createData(
        "Matrix", data=raw, keepPoints=[1, 2], keepFeatures=[1, 2])
    assert fromList == wanted

    wanted = UML.createData("Matrix", data=[[333, 222], [33, 22]])
    fromList = UML.createData(
        "Matrix", data=raw, keepPoints=[2, 1], keepFeatures=[2, 1])
    assert fromList == wanted


def test_createData_keepPF_spCoo_simple():
    if not scipy:
        return
    wanted = UML.createData("Matrix", data=[[22, 33], [222, 333]])
    rawList = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
    rawMat = numpy.matrix(rawList)
    raw = scipy.sparse.coo_matrix(rawMat)

    fromCOO = UML.createData(
        "Matrix", data=raw, keepPoints=[1, 2], keepFeatures=[1, 2])
    assert fromCOO == wanted

    wanted = UML.createData("Matrix", data=[[333, 222], [33, 22]])
    fromCOO = UML.createData(
        "Matrix", data=raw, keepPoints=[2, 1], keepFeatures=[2, 1])
    assert fromCOO == wanted


def test_createData_keepPF_spCsc_simple():
    if not scipy:
        return
    wanted = UML.createData("Matrix", data=[[22, 33], [222, 333]])
    rawList = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
    rawMat = numpy.matrix(rawList)
    raw = scipy.sparse.csc_matrix(rawMat)

    fromCSC = UML.createData(
        "Matrix", data=raw, keepPoints=[1, 2], keepFeatures=[1, 2])
    assert fromCSC == wanted

    wanted = UML.createData("Matrix", data=[[333, 222], [33, 22]])
    fromCSC = UML.createData(
        "Matrix", data=raw, keepPoints=[2, 1], keepFeatures=[2, 1])
    assert fromCSC == wanted


@raises(ArgumentException)
def test_keepPF_csv_ExceptionUnknownFeatureName_Extracted():
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        UML.createData(
            returnType='List', data=tmpCSV.name, pointNames=True,
            featureNames=True, keepFeatures=[0, "fours"])


@raises(ArgumentException)
def test_keepPF_csv_ExceptionUnknownFeatureName_Provided():
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        UML.createData(
            returnType='List', data=tmpCSV.name,
            featureNames=['ones', 'twos', 'threes'], keepFeatures=[0, "fours"])


@raises(ArgumentException)
def test_csv_keepFeatures_indexNotInFile():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        UML.createData(
            returnType='List', data=tmpCSV.name, pointNames=False,
            featureNames=False, keepPoints=[1, 2], keepFeatures=[1, 42])


@raises(ArgumentException)
def test_csv_keepPoints_indexNotInFile():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        UML.createData(
            returnType='List', data=tmpCSV.name, pointNames=False,
            featureNames=False, keepPoints=[12, 2, 3], keepFeatures=[1, 2])


@raises(ArgumentException)
def test_keepPF_csv_ExceptionUnknownPointName_extracted():
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        UML.createData(
            returnType='List', data=tmpCSV.name, pointNames=True,
            featureNames=True, keepPoints=[1, "quads"])


@raises(ArgumentException)
def test_keepPF_csv_ExceptionUnknownPointName_provided():
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        UML.createData(
            returnType='List', data=tmpCSV.name,
            pointNames=['single', 'dubs', 'trips'], keepPoints=[1, "quads"])


@raises(ArgumentException)
def test_csv_keepPoints_noNamesButNameSpecified():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        UML.createData(
            returnType='List', data=tmpCSV.name, pointNames=False,
            featureNames=False, keepPoints=['dubs', 1], keepFeatures=[2])


@raises(ArgumentException)
def test_csv_keepFeatures_noNamesButNameSpecified():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        UML.createData(
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
            UML.createData(
                returnType='List', data=tmpCSV.name, pointNames=True,
                featureNames=True, keepFeatures=[1, 1])
            assert False
        except ArgumentException:
            pass
        try:
            UML.createData(
                returnType='List', data=tmpCSV.name, pointNames=True,
                featureNames=True, keepFeatures=[1, 'twos'])
            assert False
        except ArgumentException:
            pass
        try:
            UML.createData(
                returnType='List', data=tmpCSV.name, pointNames=True,
                featureNames=True, keepFeatures=['threes', 'threes'])
            assert False
        except ArgumentException:
            pass
        try:
            UML.createData(
                returnType='List', data=tmpCSV.name, pointNames=True,
                featureNames=['ones', 'twos', 'threes'], keepFeatures=[1, 'twos'])
            assert False
        except ArgumentException:
            pass
        try:
            UML.createData(
                returnType='List', data=tmpCSV.name, pointNames=True,
                featureNames=['ones', 'twos', 'threes'],
                keepFeatures=['threes', 'threes'])
            assert False
        except ArgumentException:
            pass


def test_csv_keepPoints_duplicatesInList():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        try:
            UML.createData(
                returnType='List', data=tmpCSV.name, pointNames=True,
                featureNames=True, keepPoints=[1, 1])
            assert False
        except ArgumentException:
            pass
        try:
            UML.createData(
                returnType='List', data=tmpCSV.name, pointNames=True,
                featureNames=True, keepPoints=[1, 'dubs'])
            assert False
        except ArgumentException:
            pass
        try:
            UML.createData(
                returnType='List', data=tmpCSV.name, pointNames=True,
                featureNames=True, keepPoints=['trips', 'trips'])
            assert False
        except ArgumentException:
            pass
        try:
            UML.createData(
                returnType='List', data=tmpCSV.name,
                pointNames=['single', 'dubs', 'trips'], featureNames=True,
                keepPoints=[1, 'dubs'])
            assert False
        except ArgumentException:
            pass
        try:
            UML.createData(
                returnType='List', data=tmpCSV.name,
                pointNames=['single', 'dubs', 'trips'], featureNames=True,
                keepPoints=['trips', 'trips'])
            assert False
        except ArgumentException:
            pass


def test_createData_csv_keepPF_and_ignoreFlag():
    for t in returnTypes:
        fnames = ['threes']
        pnames = ['dubs', 'trips']
        data = [[33], [333]]
        fromList = UML.createData(
            returnType=t, data=data, pointNames=pnames, featureNames=fnames)

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("pns,ones,twos,threes\n")
            tmpCSV.write("single,1,2,3A\n")
            tmpCSV.write("dubs,11,22A,33\n")
            tmpCSV.write("trips,111,222,333\n")
            tmpCSV.flush()

            fromCSV = UML.createData(
                returnType=t, data=tmpCSV.name, pointNames=True,
                featureNames=True, keepPoints=[2, 'dubs'],
                keepFeatures=[1, 'threes'], ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV


def test_createData_keepPoints_csv_endAfterAllFound():
    wanted = UML.createData("Matrix", data=[[11, 22, 33], [1, 2, 3]])
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        # This line has an extra value - if it was actually read by the
        # csv input helper, it would cause an exception to be raised.
        tmpCSV.write("111,222,333,444\n")
        tmpCSV.flush()

        fromCSV = UML.createData("Matrix", data=tmpCSV.name, keepPoints=[1, 0])
        assert fromCSV == wanted

######################
### inputSeparator ###
######################

def test_createData_csv_inputSeparatorAutomatic():
    wanted = UML.createData("Matrix", data=[[1,2,3], [4,5,6]])
    # instantiate from csv file
    for delimiter in [',', '\t', ' ', ':', ';', '|']:
        with tempfile.NamedTemporaryFile(mode='w') as tmpCSV:
            tmpCSV.write("1{0}2{0}3\n".format(delimiter))
            tmpCSV.write("4{0}5{0}6\n".format(delimiter))
            tmpCSV.flush()

            fromCSV = UML.createData("Matrix", data=tmpCSV.name)
            assert fromCSV == wanted

def test_createData_csv_inputSeparatorSpecified():
    wanted = UML.createData("Matrix", data=[[1,2,3], [4,5,6]])
    # instantiate from csv file
    for delimiter in [',', '\t', ' ', ':', ';', '|']:
        with tempfile.NamedTemporaryFile(mode='w') as tmpCSV:
            tmpCSV.write("1{0}2{0}3\n".format(delimiter))
            tmpCSV.write("4{0}5{0}6\n".format(delimiter))
            tmpCSV.flush()

            fromCSV = UML.createData("Matrix", data=tmpCSV.name, inputSeparator=delimiter)
            assert fromCSV == wanted

@raises(FileFormatException)
def test_createData_csv_inputSeparatorConfusion():
    with tempfile.NamedTemporaryFile(mode='w') as tmpCSV:
        tmpCSV.write("1,2;3\n")
        tmpCSV.write("4,5,6\n")
        tmpCSV.flush()

        fromCSV = UML.createData("Matrix", data=tmpCSV.name)


###################
### Other tests ###
###################

def test_createData_csv_nonremoval_efficiency():
    # setup the test fail trigger by replacing the function we don't want to
    # called

    def failFunction(arg1, arg2, arg3, arg4):
        assert False  # the function we didn't want to be called was called

    for t in returnTypes:
        fromList = UML.createData(returnType=t, data=[[1, 2, 3]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.flush()
            objName = 'fromCSV'

            try:
                backup = UML.helpers._removalCleanupAndSelectionOrdering
                UML.helpers._removalCleanupAndSelectionOrdering = failFunction
                fromCSV = UML.createData(returnType=t, data=tmpCSV.name, name=objName)
                assert fromList == fromCSV
            finally:
                UML.helpers._removalCleanupAndSelectionOrdering = backup


# tests for combination of one name set being specified and one set being
# in data.


# test that if both in comment and specified names are present, the
# specified names win out.


# unit tests demonstrating our file loaders can handle arbitrarly placed blank lines


# comment lines
