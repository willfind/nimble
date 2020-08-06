import tempfile
import numpy
import os
import sys
import copy
import itertools
import datetime
try:
    from unittest import mock #python >=3.3
except ImportError:
    import mock

from nose.tools import *
from nose.plugins.attrib import attr
import scipy.sparse
import pandas as pd
import h5py

import nimble
from nimble.exceptions import InvalidArgumentValue, InvalidArgumentType
from nimble.exceptions import FileFormatException
from nimble.core.data._dataHelpers import DEFAULT_PREFIX
from nimble.core.data._dataHelpers import isDatetime
from nimble.core._createHelpers import _intFloatOrString
from nimble.core._createHelpers import replaceNumpyValues
from nimble._utility import sparseMatrixToArray

# from .. import logger
from tests.helpers import oneLogEntryExpected

returnTypes = copy.copy(nimble.core.data.available)
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

###############################
# Raw data values correctness #
###############################

def test_data_raw_stringConversion_float():
    """
    """
    for t in returnTypes:
        values = []
        toTest = nimble.data(t, [['1','2','3'], ['4','5','6'], ['7','8','9']],
                             convertToType=float)
        for i in range(len(toTest.points)):
            for j in range(len(toTest.features)):
                values.append(toTest[i,j])
        assert all(isinstance(val, float) for val in values)

def test_data_raw_stringConversion_int():
    """
    """
    for t in returnTypes:
        values = []
        toTest = nimble.data(t, [['1','2','3'], ['4','5','6'], ['7','8','9']],
                             convertToType=int)
        for i in range(len(toTest.points)):
            for j in range(len(toTest.features)):
                values.append(toTest[i,j])
        assert all(isinstance(val, (int, numpy.integer)) for val in values)

def test_data_raw_noStringConversion():
    for t in returnTypes:
        values = []
        toTest = nimble.data(t, [['1','2','3'], ['4','5','6'], ['7','8','9']])
        for i in range(len(toTest.points)):
            for j in range(len(toTest.features)):
                values.append(toTest[i,j])
        assert all(isinstance(val, str) for val in values)

def test_data_raw_numericConversion_str():
    """
    """
    for t in returnTypes:
        values = []
        toTest = nimble.data(t, [[1, 2, 3], [4, 5, 6], [7 , 8, 9]],
                             convertToType=str)
        for i in range(len(toTest.points)):
            for j in range(len(toTest.features)):
                values.append(toTest[i,j])
        assert all(isinstance(val, str) for val in values)

def test_data_raw_numericConversion_float():
    """
    """
    for t in returnTypes:
        values = []
        toTest = nimble.data(t, [[1, 2, 3], [4, 5, 6], [7 , 8, 9]],
                             convertToType=float)
        for i in range(len(toTest.points)):
            for j in range(len(toTest.features)):
                values.append(toTest[i,j])
        assert all(isinstance(val, float) for val in values)

def test_data_raw_invalidPointOrFeatureNames():
    for t in returnTypes:
        try:
            pNames = NoIter(['1', '4'])
            toTest = nimble.data(t, [[1,2,3], [4,5,6]], pointNames=pNames)
            assert False # expected InvalidArgumentType
        except InvalidArgumentType:
            pass

        try:
            fNames = NoIter(['a', 'b', 'c'])
            toTest = nimble.data(t, [[1,2,3], [4,5,6]], featureNames=fNames)
            assert False # expected InvalidArgumentType
        except InvalidArgumentType:
            pass

def test_data_raw_pointAndFeatureIterators():
    for t in returnTypes:
        pNames = IterNext(['1', '4'])
        fNames = IterNext(['a', 'b', 'c'])
        rawData = [[1,2,3], [4,5,6]]
        toTest1 = nimble.data(t, rawData, pointNames=pNames,
                              featureNames=fNames)
        assert toTest1.points.getNames() == ['1', '4']
        assert toTest1.features.getNames() == ['a', 'b', 'c']

        pNames = GetItemOnly(['1', '4'])
        fNames = GetItemOnly(['a', 'b', 'c'])
        toTest2 = nimble.data(t, rawData, pointNames=pNames,
                              featureNames=fNames)
        assert toTest2.points.getNames() == ['1', '4']
        assert toTest2.features.getNames() == ['a', 'b', 'c']

def test_data_raw_datetime():
    for t in returnTypes:
        rawData = [[datetime.datetime(2020, 1, 1), -16, 2],
                   [numpy.datetime64('2020-01-02'), -24, -6],
                   [pd.Timestamp(year=2020, month=2, day=3), -30, -18]]
        toTest = nimble.data(t, rawData)
        for date in toTest.features[0].iterateElements():
            assert isDatetime(date)

################################
# File data values correctness #
################################

def test_data_CSV_data():
    """ Test of data() loading a csv file, default params """
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[1, 2, 3]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.flush()
            objName = 'fromCSV'
            fromCSV = nimble.data(returnType=t, source=tmpCSV.name, name=objName)

            assert fromList == fromCSV


def test_data_CSV_dataRandomExtension():
    """ Test of data() loading a csv file without csv extension """
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[1, 2, 3]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".foo", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.flush()
            objName = 'fromCSV'
            fromCSV = nimble.data(returnType=t, source=tmpCSV.name, name=objName)

            assert fromList == fromCSV


def test_data_CSV_data_noComment():
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[1, 2], [1, 2]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,#3\n")
            tmpCSV.write("1,2,3\n")
            tmpCSV.flush()
            objName = 'fromCSV'
            fromCSV = nimble.data(returnType=t, source=tmpCSV.name, name=objName, ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV


def test_data_CSV_data_ListOnly():
    fromList = nimble.data(returnType="List", source=[[1, 2, 'three'], [4, 5, 'six']])

    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,three\n")
        tmpCSV.write("4,5,six\n")
        tmpCSV.flush()
        objName = 'fromCSV'
        fromCSV = nimble.data(returnType="List", source=tmpCSV.name, name=objName)

        assert fromList == fromCSV


def test_data_CSV_data_ListOnly_noComment():
    fromList = nimble.data(returnType="List", source=[[1, 2, 'three'], [4, 5, '#six']])

    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,three\n")
        tmpCSV.write("4,5,#six\n")
        tmpCSV.flush()
        objName = 'fromCSV'
        fromCSV = nimble.data(returnType="List", source=tmpCSV.name, name=objName)

        assert fromList == fromCSV

def test_data_CSV_data_unicodeCharacters():
    """ Test of data() loading a csv file with unicode characters """
    for t in returnTypes:
        data = [['P', "\u2119"] ,['Y', "\u01B4" ],['T', "\u2602"],
                ['H', "\u210C"], ['O', "\u00F8"], ['N', "\u1F24"]]
        fromList = nimble.data(returnType=t, source=data)

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("P,\u2119\n")
            tmpCSV.write("Y,\u01B4\n")
            tmpCSV.write("T,\u2602\n")
            tmpCSV.write("H,\u210C\n")
            tmpCSV.write("O,\u00F8\n")
            tmpCSV.write("N,\u1F24\n")
            tmpCSV.flush()
            objName = 'fromCSV'
            fromCSV = nimble.data(returnType=t, source=tmpCSV.name, name=objName)

            assert fromList == fromCSV

def test_data_CSV_data_columnTypeHierarchy():
    """ Test of data() loading a csv file with various column types """
    for t in returnTypes:
        data = [[True,'False','True','False','TRUE','false',1,1.0,1.0,'1','1'],
                [False,'True','False','True','FALSE','true',2,2.0,2.0,'2','2'],
                [True,'False','True','False','TRUE','false',3,3.0,3.0,'3','3'],
                [False,'TRUE','false','1','FALSE','true',4,4.0,4.0,'4.0','False'],
                [True,'FALSE','true','0','TRUE','false',5,5.0,5.0,'5.0', 'True'],
                [False,'True','False','True','FALSE','true',6,6.0,6.0,'six','6']]
        fromList = nimble.data(returnType=t, source=data)

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("True,False,True,False,TRUE,false,1,1.0,1,1,1\n")
            tmpCSV.write("False,True,False,True,FALSE,true,2,2.0,2,2,2\n")
            tmpCSV.write("True,False,True,False,TRUE,false,3,3.0,3,3,3\n")
            tmpCSV.write("False,TRUE,false,1,FALSE,true,4,4.0,4.0,4.0,False\n")
            tmpCSV.write("True,FALSE,true,0,TRUE,false,5,5.0,5.0,5.0,True\n")
            tmpCSV.write("False,True,False,True,FALSE,true,6,6.0,6,six,6\n")
            tmpCSV.flush()
            objName = 'fromCSV'
            fromCSV = nimble.data(returnType=t, source=tmpCSV.name, name=objName)

            assert fromList == fromCSV

def test_data_CSV_data_columnTypeHierarchyWithNaN():
    """ Test of data() loading a csv file with various column types with nan values """
    for t in returnTypes:
        data = [[True,'False',1,1.0,1.0,'1'],
                [False,None,2,None,None,'2'],
                [True,'False',None,3.0,3.0,None],
                [None,'TRUE',None,4.0,4.0,None],
                [True,'FALSE',5,None,None,'5.0'],
                [None,None,6,6.0,6.0,'six']]
        fromList = nimble.data(returnType=t, source=data)

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("True,False,1,1.0,1,1\n")
            tmpCSV.write("False,,2,,,2\n")
            tmpCSV.write("True,False,,3.0,3,\n")
            tmpCSV.write(",TRUE,,4.0,4.0,\n")
            tmpCSV.write("True,FALSE,5,,,5.0\n")
            tmpCSV.write(",,6,6.0,6,six\n")
            tmpCSV.flush()
            objName = 'fromCSV'
            fromCSV = nimble.data(returnType=t, source=tmpCSV.name, name=objName)

            assert fromList == fromCSV

def test_data_CSV_data_emptyStringsNotMissing():
    """ Test of data() loading a csv file empty strings not treated as missing """
    for t in returnTypes:
        data = [[True,'False',1,'1'],
                [False,'',2,'2'],
                [True,'False',None,''],
                [None,'TRUE',4,''],
                [True,'FALSE',None,'5.0'],
                [None,'',6,'six']]
        fromList = nimble.data(returnType=t, source=data, treatAsMissing=[None])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("True,False,1,1\n")
            tmpCSV.write("False,,2,2\n")
            tmpCSV.write("True,False,,\n")
            tmpCSV.write(",TRUE,4,\n")
            tmpCSV.write("True,FALSE,,5.0\n")
            tmpCSV.write(",,6,six\n")
            tmpCSV.flush()
            objName = 'fromCSV'
            fromCSV = nimble.data(returnType=t, source=tmpCSV.name, name=objName,
                                  treatAsMissing=[None])

            assert fromList == fromCSV

def test_data_MTXArr_data():
    """ Test of data() loading a mtx (arr format) file, default params """
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[1, 2, 3]])

        # instantiate from mtx array file
        with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXArr:
            tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
            tmpMTXArr.write("1 3\n")
            tmpMTXArr.write("1\n")
            tmpMTXArr.write("2\n")
            tmpMTXArr.write("3\n")
            tmpMTXArr.flush()
            objName = 'fromMTXArr'
            fromMTXArr = nimble.data(returnType=t, source=tmpMTXArr.name, name=objName)

            if t is None and fromList.getTypeString() != fromMTXArr.getTypeString():
                assert fromList.isApproximatelyEqual(fromMTXArr)
            else:
                assert fromList == fromMTXArr

def test_data_MTXArr_dataRandomExtension():
    """ Test of data() loading a mtx (arr format) file without mtx extension """
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[1, 2, 3]])

        # instantiate from mtx array file
        with tempfile.NamedTemporaryFile(suffix=".foo", mode='w') as tmpMTXArr:
            tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
            tmpMTXArr.write("1 3\n")
            tmpMTXArr.write("1\n")
            tmpMTXArr.write("2\n")
            tmpMTXArr.write("3\n")
            tmpMTXArr.flush()
            objName = 'fromMTXArr'
            fromMTXArr = nimble.data(returnType=t, source=tmpMTXArr.name, name=objName)

            if t is None and fromList.getTypeString() != fromMTXArr.getTypeString():
                assert fromList.isApproximatelyEqual(fromMTXArr)
            else:
                assert fromList == fromMTXArr


def test_data_MTXCoo_data():
    """ Test of data() loading a mtx (coo format) file, default params """
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[1, 2, 3]])

        # instantiate from mtx coordinate file
        with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXCoo:
            tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.write("1 1 1\n")
            tmpMTXCoo.write("1 2 2\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.flush()
            objName = 'fromMTXCoo'
            fromMTXCoo = nimble.data(returnType=t, source=tmpMTXCoo.name, name=objName)

            if t is None and fromList.getTypeString() != fromMTXCoo.getTypeString():
                assert fromList.isApproximatelyEqual(fromMTXCoo)
            else:
                assert fromList == fromMTXCoo

def test_data_MTXCoo_dataRandomExtension():
    """ Test of data() loading a mtx (coo format) file without mtx extension """
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[1, 2, 3]])

        # instantiate from mtx coordinate file
        with tempfile.NamedTemporaryFile(suffix=".foo", mode='w') as tmpMTXCoo:
            tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.write("1 1 1\n")
            tmpMTXCoo.write("1 2 2\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.flush()
            objName = 'fromMTXCoo'
            fromMTXCoo = nimble.data(returnType=t, source=tmpMTXCoo.name, name=objName)

            if t is None and fromList.getTypeString() != fromMTXCoo.getTypeString():
                assert fromList.isApproximatelyEqual(fromMTXCoo)
            else:
                assert fromList == fromMTXCoo


@raises(FileFormatException)
def test_data_CSV_unequalRowLength_short():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write('1,2,3,4\n')
        tmpCSV.write('4,5,6\n')
        tmpCSV.flush()

        nimble.data(returnType="List", source=tmpCSV.name)

@raises(FileFormatException)
def test_data_CSV_unequalRowLength_long():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("4,5,6,7\n")
        tmpCSV.flush()

        nimble.data(returnType="List", source=tmpCSV.name)


@raises(FileFormatException)
def test_data_CSV_unequalRowLength_definedByNames():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("one,two,three\n")
        tmpCSV.write("11,22,33,44\n")
        tmpCSV.write("4,5,6,7\n")
        tmpCSV.flush()

        nimble.data(returnType="List", source=tmpCSV.name, featureNames=True)


def test_data_CSV_unequalRowLength_position():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("#ignore\n")
        tmpCSV.write("1,2,3,4,0,0,0,0\n")
        tmpCSV.write("\n")
        tmpCSV.write("11,22,33,44,0,0,0,0\n")
        tmpCSV.write("4,5,6,0,0,0\n")
        tmpCSV.flush()

        try:
            nimble.data(returnType="List", source=tmpCSV.name, featureNames=True)
            assert False  # the previous call should have raised an exception
        except FileFormatException as ffe:
            # print(ffe.message)

            # We expect a message of the format:
            #
            assert '1' in ffe.message  # defining line
            assert '4' in ffe.message  # offending line
            # offending line number comes before defining line number
            assert ffe.message.index('4') < ffe.message.index('1')

            assert '8' in ffe.message  # expected length
            assert '6' in ffe.message  # offending length
            # offending length comes before expected length
            assert ffe.message.index('6') < ffe.message.index('8')

def test_data_HDF5_data():
    """ """
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[[[1, 2], [3, 4]],
                                                    [[1, 2], [3, 4]],],
                                                   [[[1, 2], [3, 4]],
                                                    [[1, 2], [3, 4]]]])
        # HDF5 commonly uses two extensions .hdf5 and .h5
        for suffix in ['.hdf5', '.h5']:
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmpHDF:
                arr = numpy.array([[1, 2], [3, 4]])
                hdfFile = h5py.File(tmpHDF, 'w')
                one = hdfFile.create_group('one')
                one.create_dataset('mtx1', data=arr)
                one.create_dataset('mtx2', data=arr)
                two = hdfFile.create_group('two')
                two.create_dataset('mtx1', data=arr)
                two.create_dataset('mtx2', data=arr)
                hdfFile.flush()
                hdfFile.close()
                tmpHDF.seek(0)
                fromHDF = nimble.data(returnType=t, source=tmpHDF.name)

                if t is None and fromList.getTypeString() != fromHDF.getTypeString():
                    assert fromList.isApproximatelyEqual(fromHDF)
                else:
                    assert fromList == fromHDF

def test_data_HDF5_dataRandomExtension():
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[[[1, 2], [3, 4]],
                                                    [[1, 2], [3, 4]],],
                                                   [[[1, 2], [3, 4]],
                                                    [[1, 2], [3, 4]]]])

        with tempfile.NamedTemporaryFile(suffix=".data") as tmpHDF:
            arr = numpy.array([[1, 2], [3, 4]])
            hdfFile = h5py.File(tmpHDF, 'w')
            one = hdfFile.create_group('one')
            one.create_dataset('mtx1', data=arr)
            one.create_dataset('mtx2', data=arr)
            two = hdfFile.create_group('two')
            two.create_dataset('mtx1', data=arr)
            two.create_dataset('mtx2', data=arr)
            hdfFile.flush()
            hdfFile.close()
            tmpHDF.seek(0)
            fromHDF = nimble.data(returnType=t, source=tmpHDF.name)

            if t is None and fromList.getTypeString() != fromHDF.getTypeString():
                assert fromList.isApproximatelyEqual(fromHDF)
            else:
                assert fromList == fromHDF

def test_data_HDF5_dataDifferentStructures():
    data = [[[[1, 2], [3, 4]],
             [[1, 2], [3, 4]]],
            [[[-1, -2], [-3, -4]],
             [[-1, -2], [-3, -4]]]]
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=data)

        # Case 1: file contains single Dataset with all data
        with tempfile.NamedTemporaryFile(suffix=".h5") as tmpHDF:
            hdfFile = h5py.File(tmpHDF, 'w')
            ds1 = hdfFile.create_dataset('data', data=numpy.array(data))
            hdfFile.flush()
            hdfFile.close()
            tmpHDF.seek(0)
            fromHDF = nimble.data(returnType=t, source=tmpHDF.name)

            if t is None and fromList.getTypeString() != fromHDF.getTypeString():
                assert fromList.isApproximatelyEqual(fromHDF)
            else:
                assert fromList == fromHDF

        # Case 2: Two Datasets
        with tempfile.NamedTemporaryFile(suffix=".h5") as tmpHDF:
            hdfFile = h5py.File(tmpHDF, 'w')
            hdfFile.create_dataset('mtx1', data=numpy.array(data)[0])
            hdfFile.create_dataset('mtx2', data=numpy.array(data)[1])
            hdfFile.flush()
            hdfFile.close()
            tmpHDF.seek(0)
            fromHDF = nimble.data(returnType=t, source=tmpHDF.name)

            if t is None and fromList.getTypeString() != fromHDF.getTypeString():
                assert fromList.isApproximatelyEqual(fromHDF)
            else:
                assert fromList == fromHDF

        # Case 3: Two groups containing two Datasets (matrices)
        # This is the stucture in other tests so we will not test here

        # Case 4: Two groups each containing two groups with two Datasets (vectors)
        with tempfile.NamedTemporaryFile(suffix=".hdf5") as tmpHDF:
            hdfFile = h5py.File(tmpHDF, 'w')
            zero = hdfFile.create_group('index0')
            zeroZero = zero.create_group('index0')
            zeroZero.create_dataset('index0', data=numpy.array(data)[0, 0, 0])
            zeroZero.create_dataset('index1', data=numpy.array(data)[0, 0, 1])
            zeroOne = zero.create_group('index1')
            zeroOne.create_dataset('index0', data=numpy.array(data)[0, 1, 0])
            zeroOne.create_dataset('index1', data=numpy.array(data)[0, 1, 1])
            one = hdfFile.create_group('index1')
            oneZero = one.create_group('index0')
            oneZero.create_dataset('index0', data=numpy.array(data)[1, 0, 0])
            oneZero.create_dataset('index1', data=numpy.array(data)[1, 0, 1])
            oneOne = one.create_group('index1')
            oneOne.create_dataset('index0', data=numpy.array(data)[1, 1, 0])
            oneOne.create_dataset('index1', data=numpy.array(data)[1, 1, 1])
            hdfFile.flush()
            hdfFile.close()
            tmpHDF.seek(0)
            fromHDF = nimble.data(returnType=t, source=tmpHDF.name)

            if t is None and fromList.getTypeString() != fromHDF.getTypeString():
                assert fromList.isApproximatelyEqual(fromHDF)
            else:
                assert fromList == fromHDF

############################
# Name and path attributes #
############################


def test_data_objName_and_path_CSV():
    for t in returnTypes:
        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.flush()

            objName = 'fromCSV'
            ret = nimble.data(returnType=t, source=tmpCSV.name, name=objName)
            assert ret.name == objName
            assert ret.path == tmpCSV.name
            assert ret.absolutePath == tmpCSV.name

            relExp = os.path.relpath(ret.absolutePath)
            assert ret.relativePath == relExp

            retDefName = nimble.data(returnType=t, source=tmpCSV.name)
            tokens = tmpCSV.name.rsplit(os.path.sep)
            assert retDefName.name == tokens[len(tokens) - 1]


def test_data_objName_and_path_MTXArr():
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
            ret = nimble.data(returnType=t, source=tmpMTXArr.name, name=objName)
            assert ret.name == objName
            assert ret.path == tmpMTXArr.name
            assert ret.absolutePath == tmpMTXArr.name

            relExp = os.path.relpath(ret.absolutePath)
            assert ret.relativePath == relExp

            retDefName = nimble.data(returnType=t, source=tmpMTXArr.name)
            tokens = tmpMTXArr.name.rsplit(os.path.sep)
            assert retDefName.name == tokens[len(tokens) - 1]


def test_data_objName_and_path_MTXCoo():
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
            ret = nimble.data(returnType=t, source=tmpMTXCoo.name, name=objName)
            assert ret.name == objName
            assert ret.path == tmpMTXCoo.name
            assert ret.absolutePath == tmpMTXCoo.name

            relExp = os.path.relpath(ret.absolutePath)
            assert ret.relativePath == relExp

            retDefName = nimble.data(returnType=t, source=tmpMTXCoo.name)
            tokens = tmpMTXCoo.name.rsplit(os.path.sep)
            assert retDefName.name == tokens[len(tokens) - 1]


###################################
# Point / Feature names from File #
###################################

def test_extractNames_CSV():
    """ Test of data() loading a csv file and extracting names """
    pNames = ['pn1']
    fNames = ['one', 'two', 'three']
    for t in returnTypes:
        fromList = nimble.data(
            returnType=t, source=[[1, 2, 3]], pointNames=pNames, featureNames=fNames)

        # instantiate from csv file
        tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv", mode='w')
        tmpCSV.write('ignore,one,two,three\n')
        tmpCSV.write("pn1,1,2,3\n")
        tmpCSV.flush()

        fromCSV = nimble.data(
            returnType=t, source=tmpCSV.name, pointNames=True, featureNames=True)
        tmpCSV.close()
        assert fromList == fromCSV


def test_names_AutoDetectedBlankLines_CSV():
    pNames = ['pn1']
    fNames = ['one', 'two', 'three']
    for t in returnTypes:
        fromList = nimble.data(
            returnType=t, source=[[1, 2, 3]], pointNames=pNames, featureNames=fNames)

        # instantiate from csv file
        tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv", mode='w')
        tmpCSV.write("\n")
        tmpCSV.write("\n")
        tmpCSV.write("pointNames,one,two,three\n")
        tmpCSV.write("pn1,1,2,3\n")
        tmpCSV.flush()

        fromCSV = nimble.data(returnType=t, source=tmpCSV.name)
        tmpCSV.close()
        assert fromList == fromCSV


def test_featNamesOnly_AutoDetectedBlankLines_CSV():
    fNames = ['one', 'two', 'three']
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[1, 2, 3]], featureNames=fNames)

        # instantiate from csv file
        tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv", mode='w')
        tmpCSV.write("\n")
        tmpCSV.write("\n")
        tmpCSV.write("one,two,three\n")
        tmpCSV.write("1,2,3\n")
        tmpCSV.flush()

        fromCSV = nimble.data(returnType=t, source=tmpCSV.name)
        tmpCSV.close()
        assert fromList == fromCSV

def test_pointNames_AutoDetected_from_specified_featNames_CSV():
    fNames = ['one', 'two', 'three']
    pNames = ['pn1']
    for t in returnTypes:
        fromList = nimble.data(
            returnType=t, source=[[1, 2, 3]], pointNames=pNames, featureNames=fNames)

        # instantiate from csv file
        tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv", mode='w')
        tmpCSV.write("\n")
        tmpCSV.write("\n")
        tmpCSV.write("pointNames,one,two,three\n")
        tmpCSV.write("pn1,1,2,3\n")
        tmpCSV.flush()
        fromCSV = nimble.data(returnType=t, source=tmpCSV.name, featureNames=True)
        tmpCSV.close()
        assert fromList == fromCSV


def test_specifiedIgnore_overides_autoDetectBlankLine_CSV():
    for t in returnTypes:
        data = [[0, 1, 2, 3], [10, 11, 12, 13]]
        fromList = nimble.data(returnType=t, source=data)

        # instantiate from csv file
        tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv", mode='w')
        tmpCSV.write("\n")
        tmpCSV.write("\n")
        tmpCSV.write('0,1,2,3\n')
        tmpCSV.write("10,11,12,13\n")
        tmpCSV.flush()
        fromCSV = nimble.data(
            returnType=t, source=tmpCSV.name, pointNames=False, featureNames=False)
        tmpCSV.close()
        assert fromList == fromCSV


def helper_auto(rawStr, rawType, returnType, pointNames, featureNames):
    """
    Writes a CSV and reads it in using nimble.data, fixing in place the given arguments,
    returning the resultant object

    """
    if rawType == 'csv':
        tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv", mode='w')
        tmpCSV.write(rawStr)
        tmpCSV.flush()
        ret = nimble.data(returnType=returnType, source=tmpCSV.name,
                          pointNames=pointNames, featureNames=featureNames)
        tmpCSV.close()
    else:
        fnameRow = list(map(_intFloatOrString, rawStr.split('\n')[0].split(',')))
        dataRow = list(map(_intFloatOrString, rawStr.split('\n')[1].split(',')))
        lolFromRaw = [fnameRow, dataRow]
        baseObj = nimble.data("List", lolFromRaw, pointNames=False, featureNames=False)
        finalRaw = baseObj.copy(to=rawType)
        ret = nimble.data(returnType=returnType, source=finalRaw,
                          pointNames=pointNames, featureNames=featureNames)

    return ret

def test_automaticByType_fnames_rawAndCSV():
    availableRaw = ['csv', 'pythonlist', 'numpyarray', 'numpymatrix', 'scipycoo']
    for (rawT, retT) in itertools.product(availableRaw, returnTypes):
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
    """ Test data() accepted inputs for pointNames and featureNames """
    for t in returnTypes:
        # pNames and fNames triggered for automatic
        raw1 = [['pointNames', 'fname0','fname1','fname2'],
                ['pname0', 0, 1, 2]]
        testAuto = nimble.data(t, raw1, pointNames='automatic', featureNames='automatic')
        testTrue = nimble.data(t, raw1, pointNames=True, featureNames=True)
        testFalse = nimble.data(t, raw1, pointNames=False, featureNames=False)
        testNone = nimble.data(t, raw1, pointNames=None, featureNames=None)

        assert testAuto == testTrue
        assert testAuto != testFalse
        assert testFalse == testNone

        # pNames not triggered, fNames triggered for automatic
        raw2 = [['either', 'fname0','fname1','fname2'],
                [99, 0, 1, 2]]
        testAuto = nimble.data(t, raw2, pointNames='automatic', featureNames='automatic')
        testTrue = nimble.data(t, raw2, pointNames=True, featureNames=True)
        testFalse = nimble.data(t, raw2, pointNames=False, featureNames=False)
        testNone = nimble.data(t, raw2, pointNames=None, featureNames=None)

        assert testAuto != testTrue
        assert testAuto != testFalse
        assert testTrue != testFalse
        assert testFalse == testNone

        # no names triggered for automatic
        raw3 = [[-1, 9, 8, 7],
                [99, 0, 1, 2]]
        testAuto = nimble.data(t, raw3, pointNames='automatic', featureNames='automatic')
        testTrue = nimble.data(t, raw3, pointNames=True, featureNames=True)
        testFalse = nimble.data(t, raw3, pointNames=False, featureNames=False)
        testNone = nimble.data(t, raw3, pointNames=None, featureNames=None)

        assert testAuto != testTrue
        assert testAuto == testFalse
        assert testFalse == testNone


def test_namesInComment_MTXArr():
    """ Test of data() loading a mtx (arr format) file and comment Names """
    pNames = ['pn1']
    fNames = ['one', 'two', 'three']
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[1, 2, 3]], pointNames=pNames, featureNames=fNames)

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
        fromMTXArr = nimble.data(returnType=t, source=tmpMTXArr.name)
        tmpMTXArr.close()
        if t is None and fromList.getTypeString() != fromMTXArr.getTypeString():
            assert fromList.isApproximatelyEqual(fromMTXArr)
        else:
            assert fromList == fromMTXArr


def test_namesInComment_MTXCoo():
    """ Test of data() loading a mtx (coo format) file and comment Names """
    pNames = ['pn1']
    fNames = ['one', 'two', 'three']
    for t in returnTypes:
        fromList = nimble.data(
            returnType=t, source=[[1, 2, 3]], pointNames=pNames, featureNames=fNames)

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
        fromMTXCoo = nimble.data(returnType=t, source=tmpMTXCoo.name)
        tmpMTXCoo.close()
        if t is None and fromList.getTypeString() != fromMTXCoo.getTypeString():
            assert fromList.isApproximatelyEqual(fromMTXCoo)
        else:
            assert fromList == fromMTXCoo


def test_extractNames_MTXArr():
    """ Test of data() loading a mtx (arr format) file and extracting names """
    pNames = ['11']
    fNames = ['1', '2', '3']
    for t in returnTypes:
        fromList = nimble.data(
            returnType=t, source=[[21, 22, 23]], pointNames=pNames, featureNames=fNames)

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

        fromMTXArr = nimble.data(
            returnType=t, source=tmpMTXArr.name, pointNames=True, featureNames=True)
        tmpMTXArr.close()
        if t is None and fromList.getTypeString() != fromMTXArr.getTypeString():
            assert fromList.isApproximatelyEqual(fromMTXArr)
        else:
            assert fromList == fromMTXArr


def test_extractNames_MTXCoo():
    """ Test of data() loading a mtx (coo format) file and extracting names """
    pNames = ['21']
    fNames = ['1', '2', '3']
    for t in returnTypes:
        fromList = nimble.data(
            returnType=t, source=[[22, -5, 23]], pointNames=pNames, featureNames=fNames)

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
        fromMTXCoo = nimble.data(
            returnType=t, source=tmpMTXCoo.name, pointNames=True, featureNames=True)
        tmpMTXCoo.close()
        if t is None and fromList.getTypeString() != fromMTXCoo.getTypeString():
            assert fromList.isApproximatelyEqual(fromMTXCoo)
        else:
            assert fromList == fromMTXCoo

def test_extractNames_HDF():
    pNames = ['one', 'two']
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[[[1, 2], [3, 4]],
                                                    [[1, 2], [3, 4]],],
                                                   [[[1, 2], [3, 4]],
                                                    [[1, 2], [3, 4]]]],
                               pointNames=pNames)

        with tempfile.NamedTemporaryFile(suffix=".data") as tmpHDF:
            arr = numpy.array([[1, 2], [3, 4]])
            hdfFile = h5py.File(tmpHDF, 'w')
            one = hdfFile.create_group('one')
            one.create_dataset('mtx1', data=arr)
            one.create_dataset('mtx2', data=arr)
            two = hdfFile.create_group('two')
            two.create_dataset('mtx1', data=arr)
            two.create_dataset('mtx2', data=arr)
            hdfFile.flush()
            hdfFile.close()
            tmpHDF.seek(0)
            fromHDF = nimble.data(returnType=t, source=tmpHDF.name,
                                  pointNames=True)

            if t is None and fromList.getTypeString() != fromHDF.getTypeString():
                assert fromList.isApproximatelyEqual(fromHDF)
            else:
                assert fromList == fromHDF


@raises(InvalidArgumentValue)
def test_csv_extractNames_duplicatePointName():
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write('ignore,one,two,three\n')
        tmpCSV.write("pn1,1,2,3\n")
        tmpCSV.write("pn1,11,22,33\n")
        tmpCSV.flush()

        nimble.data(returnType="List", source=tmpCSV.name, pointNames=True)


@raises(InvalidArgumentValue)
def test_csv_extractNames_duplicateFeatureName():
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write('one,two,one\n')
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.flush()

        nimble.data(returnType="List", source=tmpCSV.name, featureNames=True)


def test_csv_roundtrip_autonames():
    for retType in returnTypes:
        data = [[1, 0, 5, 12], [0, 1, 3, 17], [0, 0, 8, 22]]
        pnames = ['p0','p1','p2']
        fnames = ['f0','f1','f2', 'f3']

        withFnames = nimble.data(retType, data, featureNames=fnames)
        withBoth = nimble.data(retType, data, featureNames=fnames, pointNames=pnames)

        with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSVFnames:
            withFnames.writeFile(tmpCSVFnames.name, 'csv', includeNames=True)
            fromFileFnames = nimble.data(returnType=retType, source=tmpCSVFnames.name)
            assert fromFileFnames == withFnames

        with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSVBoth:
            withBoth.writeFile(tmpCSVBoth.name, 'csv', includeNames=True)
            fromFileBoth = nimble.data(returnType=retType, source=tmpCSVBoth.name)
            assert fromFileBoth == withBoth

def test_hdf_roundtrip_autonames():
    for t in returnTypes:
        pNames = ['one', 'two']
        data = [[[[1, 2], [3, 4]], [[1, 2], [3, 4]]],
                [[[-1, -2], [-3, -4]], [[-1, -2], [-3, -4]]]]
        withPNames = nimble.data(t, data, pointNames=pNames)

        with tempfile.NamedTemporaryFile(suffix=".hdf5") as tmpHDF:
            withPNames.writeFile(tmpHDF.name, includeNames=True)
            fromFile = nimble.data(t, tmpHDF.name)
            assert withPNames == fromFile

##################################
# Point / Feature names from Raw #
##################################


def test_extractNames_pythonList():
    """ Test of data() given python list, extracting names """
    pNames = ['pn1']
    fNames = ['one', '2', 'three']

    for t in returnTypes:
        inDataRaw = [['foo', 'one', 2, 'three'], ['pn1', 1, -1, -3]]
        specRaw = [[1, -1, -3]]
        inData = nimble.data(
            returnType=t, source=inDataRaw, pointNames=True, featureNames=True)
        specified = nimble.data(
            returnType=t, source=specRaw, pointNames=pNames, featureNames=fNames)
        assert inData == specified


def test_extractNames_NPArray():
    """ Test of data() given numpy array, extracting names """
    pNames = ['11']
    fNames = ['21', '22', '23']

    for t in returnTypes:
        inDataRaw = numpy.array([[-111, 21, 22, 23], [11, 1, -1, -3]])
        specRaw = numpy.array([[1, -1, -3]])
        inData = nimble.data(
            returnType=t, source=inDataRaw, pointNames=True, featureNames=True)
        specified = nimble.data(
            returnType=t, source=specRaw, pointNames=pNames, featureNames=fNames)
        assert inData == specified


def test_extractNames_NPMatrix():
    """ Test of data() given numpy matrix, extracting names """
    pNames = ['11']
    fNames = ['21', '22', '23']

    for t in returnTypes:
        inDataRaw = numpy.array([[-111, 21, 22, 23], [11, 1, -1, -3]])
        specRaw = numpy.matrix([[1, -1, -3]])
        inData = nimble.data(
            returnType=t, source=inDataRaw, pointNames=True, featureNames=True)
        specified = nimble.data(
            returnType=t, source=specRaw, pointNames=pNames, featureNames=fNames)
        assert inData == specified


def test_extractNames_CooSparse():
    """ Test of data() given scipy Coo matrix, extracting names """
    pNames = ['11']
    fNames = ['21', '22', '23']

    for t in returnTypes:
        inDataRaw = numpy.array([[-111, 21, 22, 23], [11, 1, -1, -3]])
        inDataRaw = scipy.sparse.coo_matrix(inDataRaw)
        specRaw = numpy.array([[1, -1, -3]])
        specRaw = scipy.sparse.coo_matrix(specRaw)
        inData = nimble.data(
            returnType=t, source=inDataRaw, pointNames=True, featureNames=True)
        specified = nimble.data(
            returnType=t, source=specRaw, pointNames=pNames, featureNames=fNames)
        assert inData == specified


def test_extractNames_CscSparse():
    """ Test of data() given scipy Csc matrix, extracting names """
    pNames = ['11']
    fNames = ['21', '22', '23']

    for t in returnTypes:
        inDataRaw = numpy.array([[-111, 21, 22, 23], [11, 1, -1, -3]])
        inDataRaw = scipy.sparse.csc_matrix(inDataRaw)
        specRaw = numpy.array([[1, -1, -3]])
        specRaw = scipy.sparse.csc_matrix(specRaw)
        inData = nimble.data(
            returnType=t, source=inDataRaw, pointNames=True, featureNames=True)
        specified = nimble.data(
            returnType=t, source=specRaw, pointNames=pNames, featureNames=fNames)
        assert inData == specified


def test_extractNames_pandasDataFrame():
    pNames = ['11']
    fNames = ['21', '22', '23']

    for t in returnTypes:
        inDataRaw = pd.DataFrame([[1, -1, -3]], index=[11], columns=[21, 22, 23])
        specRaw = pd.DataFrame([[1, -1, -3]])
        inData = nimble.data(
            returnType=t, source=inDataRaw, pointNames=True, featureNames=True)
        specified = nimble.data(
            returnType=t, source=specRaw, pointNames=pNames, featureNames=fNames)
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
        inData = nimble.data(
            returnType=t, source=rawData, pointNames=names, featureNames=names)

        if isinstance(rawData, list):
            rawData == rawDataCopy
        elif scipy.sparse.isspmatrix(rawData):
            numpy.testing.assert_array_equal(sparseMatrixToArray(rawData),
                                             sparseMatrixToArray(rawDataCopy))
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


def test_data_CSV_passedOpen():
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[1, 2, 3]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.flush()
            objName = 'fromCSV'
            openFile = open(tmpCSV.name, 'r')
            fromCSV = nimble.data(returnType=t, source=openFile, name=objName)
            openFile.close()

            assert fromList == fromCSV

            assert fromCSV.path == openFile.name
            assert fromCSV.absolutePath == openFile.name
            assert fromCSV.relativePath == os.path.relpath(openFile.name)

            openFile = open(openFile.name, 'r')
            namelessOpenFile = NamelessFile(openFile)
            fromCSV = nimble.data(
                returnType=t, source=namelessOpenFile)
            openFile.close()
            namelessOpenFile.close()
            assert fromCSV.name.startswith(nimble.core.data._dataHelpers.DEFAULT_NAME_PREFIX)
            assert fromCSV.path is None
            assert fromCSV.absolutePath is None
            assert fromCSV.relativePath is None


def test_data_MTXArr_passedOpen():
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[1, 2, 3]])

        # instantiate from mtx array file
        with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXArr:
            tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
            tmpMTXArr.write("1 3\n")
            tmpMTXArr.write("1\n")
            tmpMTXArr.write("2\n")
            tmpMTXArr.write("3\n")
            tmpMTXArr.flush()
            objName = 'fromMTXArr'
            openFile = open(tmpMTXArr.name, 'r')
            fromMTXArr = nimble.data(returnType=t, source=openFile, name=objName)
            openFile.close()

            if t is None and fromList.getTypeString() != fromMTXArr.getTypeString():
                assert fromList.isApproximatelyEqual(fromMTXArr)
            else:
                assert fromList == fromMTXArr

            assert fromMTXArr.path == openFile.name
            assert fromMTXArr.absolutePath == openFile.name
            assert fromMTXArr.relativePath == os.path.relpath(openFile.name)

            openFile = open(tmpMTXArr.name, 'r')
            namelessOpenFile = NamelessFile(openFile)
            fromMTXArr = nimble.data(
                returnType=t, source=namelessOpenFile)
            openFile.close()
            namelessOpenFile.close()
            assert fromMTXArr.name.startswith(
                nimble.core.data._dataHelpers.DEFAULT_NAME_PREFIX)
            assert fromMTXArr.path is None
            assert fromMTXArr.absolutePath is None
            assert fromMTXArr.relativePath is None


def test_data_MTXCoo_passedOpen():
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[1, 2, 3]])

        # instantiate from mtx coordinate file
        with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXCoo:
            tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.write("1 1 1\n")
            tmpMTXCoo.write("1 2 2\n")
            tmpMTXCoo.write("1 3 3\n")
            tmpMTXCoo.flush()
            objName = 'fromMTXCoo'
            openFile = open(tmpMTXCoo.name, 'r')
            fromMTXCoo = nimble.data(returnType=t, source=openFile, name=objName)
            openFile.close()

            if t is None and fromList.getTypeString() != fromMTXCoo.getTypeString():
                assert fromList.isApproximatelyEqual(fromMTXCoo)
            else:
                assert fromList == fromMTXCoo

            assert fromMTXCoo.path == openFile.name
            assert fromMTXCoo.absolutePath == openFile.name
            assert fromMTXCoo.relativePath == os.path.relpath(openFile.name)

            openFile = open(tmpMTXCoo.name, 'r')
            namelessOpenFile = NamelessFile(openFile)
            fromMTXCoo = nimble.data(
                returnType=t, source=namelessOpenFile)
            openFile.close()
            namelessOpenFile.close()
            assert fromMTXCoo.name.startswith(
                nimble.core.data._dataHelpers.DEFAULT_NAME_PREFIX)
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
            self.status_code = status_code
            self.ok = ok
            self.reason = reason
            if content is not None:
                self.content = content
                try:
                    self.text = content.decode(encoding)
                    self.encoding = encoding
                except UnicodeDecodeError:
                    self.text = str(self.content, errors='replace')
                    self.encoding = None
            else:
                self.content = None
                self.text = None
                self.encoding = None
            self.apparent_encoding = encoding

    if args[0] == 'http://mockrequests.nimble/CSVNoExtension':
        return MockResponse(b'1,2,3\n4,5,6', 200)
    elif args[0] == 'http://mockrequests.nimble/CSVAmbiguousExtension.data':
        return MockResponse(b'1,2,3\n4,5,6', 200)
    elif args[0] == 'http://mockrequests.nimble/CSV.csv':
        return MockResponse(b'1,2,3\n4,5,6', 200)
    elif args[0] == 'http://mockrequests.nimble/CSVcarriagereturn.csv':
        return MockResponse(b'1,2,3\r4,5,6', 200)
    elif args[0] == 'http://mockrequests.nimble/CSVunicodetest.csv':
        return MockResponse(b'1,2,\xc2\xa1\n4,5,6', 200)
    elif args[0] == 'http://mockrequests.nimble/CSVquotednewline.csv':
        # csv allows for newline characters in field values within double quotes
        return MockResponse(b'1,2,"a/nb"\n4,5,6', 200)
    elif args[0].startswith('http://mockrequests.nimble/MTX'):
        mtx = b'%%MatrixMarket matrix coordinate real general\n'
        mtx += b'2 3 6\n1 1 1\n1 2 2\n1 3 3\n2 1 4\n2 2 5\n2 3 6'
        return MockResponse(mtx, 200)
    elif args[0].startswith('http://mockrequests.nimble/HDF'):
        data = [[[[1, 2], [3, 4]]], [[[-1, -2], [-3, -4]]]]
        with tempfile.NamedTemporaryFile(suffix=".data") as tmpHDF:
            hdfFile = h5py.File(tmpHDF, 'w')
            ds1 = hdfFile.create_dataset('data', data=numpy.array(data))
            hdfFile.flush()
            hdfFile.close()
            tmpHDF.seek(0)
            return MockResponse(tmpHDF.read(), 200)

    return MockResponse(None, 404, False, 'Not Found')

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_data_http_CSVNoExtension(mock_get):
    for t in returnTypes:
        exp = nimble.data(returnType=t, source=[[1,2,3],[4,5,6]])
        url = 'http://mockrequests.nimble/CSVNoExtension'
        fromWeb = nimble.data(returnType=t, source=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_data_http_CSVAmbiguousExtension(mock_get):
    for t in returnTypes:
        exp = nimble.data(returnType=t, source=[[1,2,3],[4,5,6]])
        url = 'http://mockrequests.nimble/CSVAmbiguousExtension.data'
        fromWeb = nimble.data(returnType=t, source=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_data_http_CSVFileOK(mock_get):
    for t in returnTypes:
        exp = nimble.data(returnType=t, source=[[1,2,3],[4,5,6]])
        url = 'http://mockrequests.nimble/CSV.csv'
        fromWeb = nimble.data(returnType=t, source=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_data_http_CSVCarriageReturn(mock_get):
    for t in returnTypes:
        exp = nimble.data(returnType=t, source=[[1,2,3],[4,5,6]])
        url = 'http://mockrequests.nimble/CSVcarriagereturn.csv'
        fromWeb = nimble.data(returnType=t, source=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_data_http_CSVNonUnicodeValues(mock_get):
    for t in returnTypes:
        exp = nimble.data(returnType=t, source=[[1,2,"\u00A1"],[4,5,'6']])
        url = 'http://mockrequests.nimble/CSVunicodetest.csv'
        fromWeb = nimble.data(returnType=t, source=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_data_http_CSVQuotedNewLine(mock_get):
    for t in returnTypes:
        exp = nimble.data(returnType=t, source=[[1,2,"a/nb"],[4,5,'6']])
        url = 'http://mockrequests.nimble/CSVquotednewline.csv'
        fromWeb = nimble.data(returnType=t, source=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_data_http_CSVPathsWithUrl(mock_get):
    for t in returnTypes:
        url = 'http://mockrequests.nimble/CSVNoExtension'
        fromWeb = nimble.data(returnType=t, source=url)
        assert fromWeb.absolutePath == url
        assert fromWeb.relativePath == None

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_data_http_MTXNoExtension(mock_get):
    for t in returnTypes:
        # None returnType for url will default to Sparse so use coo_matrix for data
        data = scipy.sparse.coo_matrix([[1,2,3],[4,5,6]])
        exp = nimble.data(returnType=t, source=data)
        url = 'http://mockrequests.nimble/MTXNoExtension'
        fromWeb = nimble.data(returnType=t, source=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_data_http_MTXAmbiguousExtension(mock_get):
    for t in returnTypes:
        # None returnType for url will default to Sparse so use coo_matrix for data
        data = scipy.sparse.coo_matrix([[1,2,3],[4,5,6]])
        exp = nimble.data(returnType=t, source=data)
        url = 'http://mockrequests.nimble/MTXAmbiguousExtension.data'
        fromWeb = nimble.data(returnType=t, source=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_data_http_MTXFileOK(mock_get):
    for t in returnTypes:
        # None returnType for url will default to Sparse so use coo_matrix for data
        data = scipy.sparse.coo_matrix([[1,2,3],[4,5,6]])
        exp = nimble.data(returnType=t, source=data)
        url = 'http://mockrequests.nimble/MTX.mtx'
        fromWeb = nimble.data(returnType=t, source=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_data_http_MTXPathsWithUrl(mock_get):
    for t in returnTypes:
        data = scipy.sparse.coo_matrix([[1,2,3],[4,5,6]])
        url = 'http://mockrequests.nimble/MTXNoExtension'
        fromWeb = nimble.data(returnType=t, source=url)
        assert fromWeb.absolutePath == url
        assert fromWeb.relativePath == None

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_data_http_HDFNoExtension(mock_get):
    for t in returnTypes:
        data = [[[[1, 2], [3, 4]]], [[[-1, -2], [-3, -4]]]]
        exp = nimble.data(returnType=t, source=data)
        url = 'http://mockrequests.nimble/HDFNoExtension'
        fromWeb = nimble.data(returnType=t, source=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_data_http_HDFAmbiguousExtension(mock_get):
    for t in returnTypes:
        data = [[[[1, 2], [3, 4]]], [[[-1, -2], [-3, -4]]]]
        exp = nimble.data(returnType=t, source=data)
        url = 'http://mockrequests.nimble/HDFAmbiguousExtension.data'
        fromWeb = nimble.data(returnType=t, source=url)
        assert fromWeb == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_data_http_HDFFileOK(mock_get):
    for t in returnTypes:
        data = [[[[1, 2], [3, 4]]], [[[-1, -2], [-3, -4]]]]
        exp = nimble.data(returnType=t, source=data)
        url1 = 'http://mockrequests.nimble/HDF.hdf5'
        fromWeb1 = nimble.data(returnType=t, source=url1)
        url2 = 'http://mockrequests.nimble/HDF.h5'
        fromWeb2 = nimble.data(returnType=t, source=url2)
        assert fromWeb1 == fromWeb2 == exp

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_data_http_HDFPathsWithUrl(mock_get):
    for t in returnTypes:
        data = [[[[1, 2], [3, 4]]], [[[-1, -2], [-3, -4]]]]
        url = 'http://mockrequests.nimble/HDFNoExtension'
        fromWeb = nimble.data(returnType=t, source=url)
        assert fromWeb.absolutePath == url
        assert fromWeb.relativePath == None

@mock.patch('requests.get', side_effect=mocked_requests_get)
def test_data_http_linkError(mock_get):
    for t in returnTypes:
        try:
            url = 'http://mockrequests.nimble/linknotfound.csv'
            fromWeb = nimble.data(returnType=t, source=url)
            assert False # expected InvalidArgumentValue
        except InvalidArgumentValue:
            pass

###################################
# ignoreNonNumericalFeatures flag #
###################################

def test_data_ignoreNonNumericalFeaturesCSV():
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[1, 3], [5, 7]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,two,3.0,four\n")
            tmpCSV.write("5,six,7,8\n")
            tmpCSV.flush()

            fromCSV = nimble.data(
                returnType=t, source=tmpCSV.name, ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV

            # sanity check
            fromCSV = nimble.data(returnType=t, source=tmpCSV.name)
            assert len(fromCSV.features) == 4


def test_data_CSV_ignoreNonNumerical_removalCleanup_hard():
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[1, 3], [5, 7], [11, 12], [13, 14]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3.0,4.0,1\n")
            tmpCSV.write("5,six,7,8,1\n")
            tmpCSV.write("11,6,12,eight,1.0\n")
            tmpCSV.write("13,one,14,9,who?\n")
            tmpCSV.flush()

            fromCSV = nimble.data(
                returnType=t, source=tmpCSV.name, ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV

            # sanity check
            fromCSV = nimble.data(returnType=t, source=tmpCSV.name)
            assert len(fromCSV.features) == 5


def test_data_CSV_ignoreNonNumerical_removalCleanup_easy():
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[1, 3], [5, 7], [11, 12], [13, 14]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,two,3.0,four,one\n")
            tmpCSV.write("5,6,7,8,1\n")
            tmpCSV.write("11,6,12,8,1.0\n")
            tmpCSV.write("13,1,14,9,2\n")
            tmpCSV.flush()

            fromCSV = nimble.data(
                returnType=t, source=tmpCSV.name, ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV

            # sanity check
            fromCSV = nimble.data(returnType=t, source=tmpCSV.name)
            assert len(fromCSV.features) == 5


def test_data_ignoreNonNumericalFeaturesCSV_noEffect():
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[1, 2, 3, 4], [5, 6, 7, 8]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3,4\n")
            tmpCSV.write("5,6,7,8\n")
            tmpCSV.flush()

            fromCSV = nimble.data(
                returnType=t, source=tmpCSV.name, ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV

            fromCSV = nimble.data(returnType=t, source=tmpCSV.name)
            assert len(fromCSV.features) == 4


def test_CSV_ignoreNonNumericalFeatures_featureNamesDontTrigger():
    for t in returnTypes:
        fnames = ['1', '2', '3', 'four']
        fromList = nimble.data(returnType=t, featureNames=fnames, source=[[5, 6, 7, 8]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3,four\n")
            tmpCSV.write("5,6,7,8\n")
            tmpCSV.flush()

            fromCSV = nimble.data(
                returnType=t, source=tmpCSV.name, featureNames=True,
                ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV


def test_CSV_ignoreNonNumericalFeatures_featureNamesAdjusted():
    for t in returnTypes:
        fNames = ["1", "2", "3"]
        data = [[1, 2, 3], [5, 6, 7]]
        fromList = nimble.data(returnType=t, featureNames=fNames, source=data)

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3,4\n")
            tmpCSV.write("1,2,3,four\n")
            tmpCSV.write("5,6,7,H8\n")
            tmpCSV.flush()

            fromCSV = nimble.data(
                returnType=t, source=tmpCSV.name, featureNames=True,
                ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV


def test_CSV_data_ignoreNonNumericalFeatures_allRemoved():
    for t in returnTypes:
        pNames = ['single', 'dubs', 'trips']
        fromList = nimble.data(returnType=t, pointNames=pNames, source=[[], [], []])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write(",ones,twos,threes\n")
            tmpCSV.write("single,1A,2A,3A\n")
            tmpCSV.write("dubs,11,22A,33\n")
            tmpCSV.write("trips,111,222,333\n")
            tmpCSV.flush()

            fromCSV = nimble.data(
                returnType=t, source=tmpCSV.name, pointNames=True,
                featureNames=True, ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV


####################################################
# Difficult CSV Formatting: whitespace and quoting #
####################################################

def test_CSVformatting_simpleQuotedValues():
    for t in returnTypes:
        fromList = nimble.data(returnType=t, source=[[1, 2, 3, 4], [5, 6, 7, 8]])

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,\"2\",\"3\",4\n")
            tmpCSV.write("5,\"6\",\"7\",8\n")
            tmpCSV.flush()

            fromCSV = nimble.data(returnType=t, source=tmpCSV.name)

            assert fromList == fromCSV


def test_CSVformatting_specialCharsInQuotes():
    for t in returnTypes:
        fNames = ["1,ONE", "2;TWO", "3\t'EE'"]
        data = [[1, 2, 3], [5, 6, 7]]
        dataAll = [[1, 2, 3, 4], [5, 6, 7, 8]]
        fromList = nimble.data(returnType=t, featureNames=fNames[:3], source=data)

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("\"1,ONE\",\"2;TWO\",\"3\t'EE'\",\"4f\"\n")
            tmpCSV.write("1,2,3,four\n")
            tmpCSV.write("5,6,7,H8\n")
            tmpCSV.flush()

            fromCSV = nimble.data(
                returnType=t, source=tmpCSV.name, featureNames=True,
                ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV


def test_CSVformatting_emptyAndCommentLines():
    for t in returnTypes:
        data = [['1', 2, 3, 4], ['#11', 22, 33, 44], ['5', 6, 7, 8]]

        fromList = nimble.data(returnType=t, source=data)

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

            fromCSV = nimble.data(
                returnType=t, source=tmpCSV.name, featureNames=False)

            assert fromList == fromCSV


def test_CSVformatting_scientificNotation():
    for t in returnTypes:
        data = [[1., 2., 3.], [11., 22., 33.], [111., 222., 333.]]
        fromRaw = nimble.data(returnType=t, source=data)

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1.000000000e+00,2.000000000e+00,3.000000000e+00\n")
            tmpCSV.write("1.100000000e+01,2.200000000e+01,3.300000000e+01\n")
            tmpCSV.write("1.110000000e+02,2.220000000e+02,3.330000000e+02\n")
            tmpCSV.flush()

            fromCSV = nimble.data(returnType=t, source=tmpCSV.name)

            assert fromRaw == fromCSV


################################
# keepPoints, keepFeatures #
################################

def test_data_keepPF_AllPossibleNatOrder():
    filesForms = ['csv', 'mtx']
    for (t, f) in itertools.product(returnTypes, filesForms):
        data = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
        orig = nimble.data(returnType=t, source=data)
        with tempfile.NamedTemporaryFile(suffix="." + f) as tmpF:
            orig.writeFile(tmpF.name, fileFormat=f, includeNames=False)
            tmpF.flush()

            poss = [[0], [1], [2], [0, 1], [0, 2], [1, 2], 'all']
            for (pSel, fSel) in itertools.product(poss, poss):
                ret = nimble.data(
                    t, tmpF.name, keepPoints=pSel, keepFeatures=fSel)
                fromOrig = nimble.data(
                    t, orig.data, keepPoints=pSel, keepFeatures=fSel)

                assert ret == fromOrig


def test_data_keepPF_AllPossibleReverseOrder():
    filesForms = ['csv', 'mtx']
    for (t, f) in itertools.product(returnTypes, filesForms):
        data = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
        orig = nimble.data(returnType=t, source=data)
        with tempfile.NamedTemporaryFile(suffix="." + f) as tmpF:
            orig.writeFile(tmpF.name, fileFormat=f, includeNames=False)
            tmpF.flush()

            poss = [[0, 1], [0, 2], [1, 2]]
            for (pSel, fSel) in itertools.product(poss, poss):
                ret = nimble.data(
                    t, tmpF.name, keepPoints=pSel, keepFeatures=fSel)
                fromOrig = nimble.data(
                    t, orig.data, keepPoints=pSel, keepFeatures=fSel)

                assert ret == fromOrig

                pSelR = copy.copy(pSel)
                pSelR.reverse()
                fSelR = copy.copy(fSel)
                fSelR.reverse()

                retT = nimble.data(
                    t, tmpF.name, keepPoints=pSelR, keepFeatures=fSelR)
                fromOrigT = nimble.data(
                    t, orig.data, keepPoints=pSelR, keepFeatures=fSelR)

                assert retT != ret
                assert retT == fromOrigT
                assert fromOrigT != fromOrig


def test_data_keepPF_AllPossibleWithNames_extracted():
    data = [[1., 2., 3.], [11., 22., 33.], [111., 222., 333.]]
    orig = nimble.data(returnType="List", source=data)
    filesForms = ['csv', 'mtx']
    for (t, f) in itertools.product(returnTypes, filesForms):
        with tempfile.NamedTemporaryFile(suffix="." + f) as tmpF:
            orig.writeFile(tmpF.name, fileFormat=f, includeNames=False)
            tmpF.flush()

            poss = [[0], [1], [0, 1], [1, 0], 'all']
            for (pSel, fSel) in itertools.product(poss, poss):
                toUse = orig.copy(to="pythonlist")
                fromOrig = nimble.data(
                    t, toUse, keepPoints=pSel, keepFeatures=fSel,
                    pointNames=True, featureNames=True)

                ret = nimble.data(
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

                retN = nimble.data(
                    t, tmpF.name, keepPoints=pSelUse,
                    keepFeatures=fSelUse, pointNames=True,
                    featureNames=True)

                assert ret == fromOrig
                assert retN == fromOrig


def test_data_keepPF_AllPossibleWithNames_fullNamesListProvided():
    pnames = ["11.", "111.", "1111"]
    fnames = ["2.", "3.", "4."]
    data = [[22., 33., 44.], [222., 333., 444.], [2222., 3333., 4444.]]
    orig = nimble.data(
        returnType="List", source=data, pointNames=pnames, featureNames=fnames)

    filesForms = ['csv', 'mtx']
    for (t, f) in itertools.product(returnTypes, filesForms):
        with tempfile.NamedTemporaryFile(suffix="." + f) as tmpF:
            orig.writeFile(tmpF.name, fileFormat=f, includeNames=False)
            tmpF.flush()

            poss = [[0], [1], [0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1], 'all']
            for (pSel, fSel) in itertools.product(poss, poss):
                toUseData = orig.copy(to="pythonlist")

                fromOrig = nimble.data(
                    t, toUseData, keepPoints=pSel, keepFeatures=fSel,
                    pointNames=pnames, featureNames=fnames)

                ret = nimble.data(
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

                retN = nimble.data(
                    t, tmpF.name, keepPoints=pSelUse,
                    keepFeatures=fSelUse, pointNames=pnames,
                    featureNames=fnames)

                assert ret == fromOrig
                assert retN == fromOrig


def test_data_keepPF_AllPossibleWithNames_fullNamesDictProvided():
    pnames = {"11.": 0, "111.": 1, "1111.": 2}
    fnames = {"2.": 0, "3.": 1, "4.": 2}
    data = [[22., 33., 44.], [222., 333., 444.], [2222., 3333., 4444.]]
    orig = nimble.data(
        returnType="List", source=data, pointNames=pnames, featureNames=fnames)

    filesForms = ['csv', 'mtx']
    for (t, f) in itertools.product(returnTypes, filesForms):
        with tempfile.NamedTemporaryFile(suffix="." + f) as tmpF:
            orig.writeFile(tmpF.name, fileFormat=f, includeNames=False)
            tmpF.flush()

            poss = [[0], [1], [0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1], 'all']
            for (pSel, fSel) in itertools.product(poss, poss):
                toUseData = orig.copy(to="pythonlist")

                fromOrig = nimble.data(
                    t, toUseData, keepPoints=pSel, keepFeatures=fSel,
                    pointNames=pnames, featureNames=fnames)

                ret = nimble.data(
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

                retN = nimble.data(
                    t, tmpF.name, keepPoints=pSelUse,
                    keepFeatures=fSelUse, pointNames=pnames,
                    featureNames=fnames)

                assert ret == fromOrig
                assert retN == fromOrig


def test_data_keepPF_AllCombosWithExactNamesProvided():
    pnames = ["11.", "111.", "1111."]
    fnames = ["2.", "3.", "4."]
    data = [[22., 33., 44.], [222., 333., 444.], [2222., 3333., 4444.]]
    orig = nimble.data(
        returnType="List", source=data, pointNames=pnames, featureNames=fnames)

    filesForms = ['csv', 'mtx']
    for (t, f) in itertools.product(returnTypes, filesForms):
        with tempfile.NamedTemporaryFile(suffix="." + f) as tmpF:
            orig.writeFile(tmpF.name, fileFormat=f, includeNames=False)
            tmpF.flush()

            toUseData = orig.copy(to="pythonlist")
            pSel = [0, 1]
            fSel = [1, 0]
            pNameSel = ["11.", "111."]
            fNameSel = ["3.", "2."]

            fromOrig = nimble.data(
                t, toUseData, keepPoints=pSel, keepFeatures=fSel,
                pointNames=pNameSel, featureNames=fNameSel)

            ret = nimble.data(
                t, tmpF.name, keepPoints=pSel, keepFeatures=fSel,
                pointNames=pNameSel, featureNames=fNameSel)

            assert ret == fromOrig

            pSel = ["11.", "111."]
            # using names should fail because we do not have full
            # access to the names for every data point
            try:
                retN = nimble.data(
                    t, tmpF.name, keepPoints=pSel, pointNames=pNameSel,
                    featureNames=fNameSel)
                assert False # expected InvalidArgumentValue
            except InvalidArgumentValue:
                pass

            fSel = ["3.", "2."]
            # using names should fail because we do not have full
            # access to the names for every data point
            try:
                retN = nimble.data(
                    t, tmpF.name, keepFeatures=fSel, pointNames=pNameSel,
                    featureNames=fNameSel)
                assert False # expected InvalidArgumentValue
            except InvalidArgumentValue:
                pass

            # keepPoints/Features is not permitted to be the same length
            # as its respective axis when names are not extracted
            pSel = [1, 0, 2]
            try:
                ret = nimble.data(
                    t, tmpF.name, keepPoints=pSel, pointNames=pnames,
                    featureNames=fnames)
                assert False # expected InvalidArgumentValue
            except InvalidArgumentValue:
                pass

            pSel = ["111.", "11.", "1111."]
            try:
                retN = nimble.data(
                    t, tmpF.name, keepPoints=pSel, pointNames=pnames,
                    featureNames=fnames)
                assert False # expected InvalidArgumentValue
            except InvalidArgumentValue:
                pass

            fSel = [2, 1, 0]
            try:
                ret = nimble.data(
                    t, tmpF.name, keepFeatures=fSel, pointNames=pnames,
                    featureNames=fnames)
                assert False # expected InvalidArgumentValue
            except InvalidArgumentValue:
                pass

            fSel = ["3.", "2.", "4."]
            try:
                retN = nimble.data(
                    t, tmpF.name, keepFeatures=fSel, pointNames=pnames,
                    featureNames=fnames)
                assert False # expected InvalidArgumentValue
            except InvalidArgumentValue:
                pass


def test_data_csv_keepPoints_IndexingGivenFeatureNames():
    data = [[111, 222, 333]]
    fnames = ['1', '2', '3']
    wanted = nimble.data("Matrix", source=data, featureNames=fnames)
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        fromCSV = nimble.data(
            "Matrix", source=tmpCSV.name, keepPoints=[1], featureNames=True)

        raw = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
        fromRaw = nimble.data(
            "Matrix", source=raw, keepPoints=[1], featureNames=True)

        assert fromRaw == wanted
        assert fromCSV == wanted


# since the data helper for raw data is chained onto the
# helper for file input, we need tests to show that we don't
# just read all of the data into an object and then remove the
# stuff we don't want in the raw data helper. If these pass,
# unwanted data could still be stored in memory, but it limits
# that mistake to the file input helpers only.

def test_data_keepPF_csv_noUncessaryStorage():
    wanted = nimble.data("List", source=[[22], [222]])
    backup = nimble.core._createHelpers.initDataObject

    try:
        def fakeinitDataObject(
                returnType, rawData, pointNames, featureNames, name, path,
                keepPoints, keepFeatures, treatAsMissing, replaceMissingWith,
                reuseData=False, extracted=(None, None)):
            assert len(rawData) == 2
            assert len(rawData[0]) == 1
            return nimble.core.data.List(rawData)

        nimble.core._createHelpers.initDataObject = fakeinitDataObject

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.write("11,22,33\n")
            tmpCSV.write("111,222,333\n")
            tmpCSV.flush()

            fromCSV = nimble.data(
                "List", source=tmpCSV.name, keepPoints=[1, 2], keepFeatures=[1])
            assert fromCSV == wanted
    finally:
        nimble.core._createHelpers.initDataObject = backup

#def TODOtest_data_keepPF_mtxArr_noUncessaryStorage():
#	fromList = nimble.data(returnType='Matrix', source=[[2]])
#	backup = nimble.core._createHelpers.initDataObject
#
#	try:
#		def fakeinitDataObject(
#				returnType, rawData, pointNames, featureNames, name, path,
#				keepPoints, keepFeatures):
#			assert len(rawData) == 1
#			assert len(rawData[0]) == 1
#			return nimble.core.data.List(rawData)
#
#		nimble.core._createHelpers.initDataObject = fakeinitDataObject
#
#		# instantiate from mtx array file
#		with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXArr:
#			tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
#			tmpMTXArr.write("1 3\n")
#			tmpMTXArr.write("1\n")
#			tmpMTXArr.write("2\n")
#			tmpMTXArr.write("3\n")
#			tmpMTXArr.flush()
#			fromMTXArr = nimble.data(
#				returnType='Matrix', source=tmpMTXArr.name, keepPoints=[0],
#				keepFeatures=[1])
#
#			assert fromList == fromMTXArr
#	finally:
#		nimble.core._createHelpers.initDataObject = backup


#def TODOtest_data_keepPF_mtxCoo_noUncessaryStorage():
#	fromList = nimble.data(returnType='Matrix', source=[[2]])
#	backup = nimble.core._createHelpers.initDataObject
#
#	try:
#		def fakeinitDataObject(
#				returnType, rawData, pointNames, featureNames, name, path,
#				keepPoints, keepFeatures):
#			assert rawData.shape == (1,1)
#			return nimble.core.data.List(rawData)
#
#		nimble.core._createHelpers.initDataObject = fakeinitDataObject
#
#		# instantiate from mtx coordinate file
#		with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXCoo:
#			tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
#			tmpMTXCoo.write("1 3 3\n")
#			tmpMTXCoo.write("1 1 1\n")
#			tmpMTXCoo.write("1 2 2\n")
#			tmpMTXCoo.write("1 3 3\n")
#			tmpMTXCoo.flush()
#			fromMTXCoo = nimble.data(
#				returnType='Matrix', source=tmpMTXCoo.name, keepPoints=[0],
#				keepFeatures=[1])
#
#			assert fromList == fromMTXCoo
#
#	finally:
#		nimble.core._createHelpers.initDataObject = backup


def test_data_keepPF_csv_simple():
    wanted = nimble.data("Matrix", source=[[222], [22]])
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        fromCSV = nimble.data(
            "Matrix", source=tmpCSV.name, keepPoints=[2, 1], keepFeatures=[1])
        assert fromCSV == wanted


def test_data_keepPF_mtxArr_simple():
    fromList = nimble.data(returnType='Matrix', source=[[3]])

    # instantiate from mtx array file
    with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXArr:
        tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
        tmpMTXArr.write("2 2\n")
        tmpMTXArr.write("1\n")
        tmpMTXArr.write("2\n")
        tmpMTXArr.write("3\n")
        tmpMTXArr.write("4\n")
        tmpMTXArr.flush()
        fromMTXArr = nimble.data(
            returnType='Matrix', source=tmpMTXArr.name, keepPoints=[0],
            keepFeatures=[1])

        assert fromList == fromMTXArr


def test_data_keepPF_mtxCoo_simple():
    fromList = nimble.data(returnType='Matrix', source=[[2]])

    # instantiate from mtx coordinate file
    with tempfile.NamedTemporaryFile(suffix=".mtx", mode='w') as tmpMTXCoo:
        tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
        tmpMTXCoo.write("2 3 3\n")
        tmpMTXCoo.write("1 1 1\n")
        tmpMTXCoo.write("1 2 2\n")
        tmpMTXCoo.write("2 3 3\n")
        tmpMTXCoo.flush()
        fromMTXCoo = nimble.data(
            returnType='Matrix', source=tmpMTXCoo.name, keepPoints=[0],
            keepFeatures=[1])

        assert fromList == fromMTXCoo


def test_data_keepPF_pythonList_simple():
    wanted = nimble.data("Matrix", source=[[22, 33], [222, 333]])
    raw = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]

    fromList = nimble.data(
        "Matrix", source=raw, keepPoints=[1, 2], keepFeatures=[1, 2])
    assert fromList == wanted

    wanted = nimble.data("Matrix", source=[[333, 222], [33, 22]])
    fromList = nimble.data(
        "Matrix", source=raw, keepPoints=[2, 1], keepFeatures=[2, 1])
    assert fromList == wanted


def test_data_keepPF_npArray_simple():
    wanted = nimble.data("Matrix", source=[[22, 33], [222, 333]])
    rawList = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
    raw = numpy.array(rawList)

    fromNPArr = nimble.data(
        "Matrix", source=raw, keepPoints=[1, 2], keepFeatures=[1, 2])
    assert fromNPArr == wanted

    wanted = nimble.data("Matrix", source=[[333, 222], [33, 22]])
    fromNPArr = nimble.data(
        "Matrix", source=raw, keepPoints=[2, 1], keepFeatures=[2, 1])
    assert fromNPArr == wanted


def test_data_keepPF_npMatrix_simple():
    wanted = nimble.data("Matrix", source=[[22, 33], [222, 333]])
    rawList = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
    raw = numpy.matrix(rawList)

    fromList = nimble.data(
        "Matrix", source=raw, keepPoints=[1, 2], keepFeatures=[1, 2])
    assert fromList == wanted

    wanted = nimble.data("Matrix", source=[[333, 222], [33, 22]])
    fromList = nimble.data(
        "Matrix", source=raw, keepPoints=[2, 1], keepFeatures=[2, 1])
    assert fromList == wanted


def test_data_keepPF_spCoo_simple():
    wanted = nimble.data("Matrix", source=[[22, 33], [222, 333]])
    rawList = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
    raw = scipy.sparse.coo_matrix(rawList)

    fromCOO = nimble.data(
        "Matrix", source=raw, keepPoints=[1, 2], keepFeatures=[1, 2])
    assert fromCOO == wanted

    wanted = nimble.data("Matrix", source=[[333, 222], [33, 22]])
    fromCOO = nimble.data(
        "Matrix", source=raw, keepPoints=[2, 1], keepFeatures=[2, 1])
    assert fromCOO == wanted


def test_data_keepPF_spCsc_simple():
    wanted = nimble.data("Matrix", source=[[22, 33], [222, 333]])
    rawList = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
    raw = scipy.sparse.csc_matrix(rawList)

    fromCSC = nimble.data(
        "Matrix", source=raw, keepPoints=[1, 2], keepFeatures=[1, 2])
    assert fromCSC == wanted

    wanted = nimble.data("Matrix", source=[[333, 222], [33, 22]])
    fromCSC = nimble.data(
        "Matrix", source=raw, keepPoints=[2, 1], keepFeatures=[2, 1])
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

        nimble.data(
            returnType='List', source=tmpCSV.name, pointNames=True,
            featureNames=True, keepFeatures=[0, "fours"])


@raises(InvalidArgumentValue)
def test_keepPF_csv_ExceptionUnknownFeatureName_Provided():
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        nimble.data(
            returnType='List', source=tmpCSV.name,
            featureNames=['ones', 'twos', 'threes'], keepFeatures=[0, "fours"])


@raises(InvalidArgumentValue)
def test_csv_keepFeatures_indexNotInFile():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        nimble.data(
            returnType='List', source=tmpCSV.name, pointNames=False,
            featureNames=False, keepPoints=[1, 2], keepFeatures=[1, 42])


@raises(InvalidArgumentValue)
def test_csv_keepPoints_indexNotInFile():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        nimble.data(
            returnType='List', source=tmpCSV.name, pointNames=False,
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

        nimble.data(
            returnType='List', source=tmpCSV.name, pointNames=True,
            featureNames=True, keepPoints=[1, "quads"])


@raises(InvalidArgumentValue)
def test_keepPF_csv_ExceptionUnknownPointName_provided():
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        nimble.data(
            returnType='List', source=tmpCSV.name,
            pointNames=['single', 'dubs', 'trips'], keepPoints=[1, "quads"])


@raises(InvalidArgumentValue)
def test_csv_keepPoints_noNamesButNameSpecified():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        nimble.data(
            returnType='List', source=tmpCSV.name, pointNames=False,
            featureNames=False, keepPoints=['dubs', 1], keepFeatures=[2])


@raises(InvalidArgumentValue)
def test_csv_keepFeatures_noNamesButNameSpecified():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        nimble.data(
            returnType='List', source=tmpCSV.name, pointNames=False,
            featureNames=False, keepFeatures=["threes"])


def test_csv_keepFeatures_duplicatesInList():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("pns,ones,twos,threes\n")
        tmpCSV.write("single,1,2,3\n")
        tmpCSV.write("dubs,11,22,33\n")
        tmpCSV.write("trips,111,222,333\n")
        tmpCSV.flush()

        try:
            nimble.data(
                returnType='List', source=tmpCSV.name, pointNames=True,
                featureNames=True, keepFeatures=[1, 1])
            assert False
        except InvalidArgumentValue:
            pass
        try:
            nimble.data(
                returnType='List', source=tmpCSV.name, pointNames=True,
                featureNames=True, keepFeatures=[1, 'twos'])
            assert False
        except InvalidArgumentValue:
            pass
        try:
            nimble.data(
                returnType='List', source=tmpCSV.name, pointNames=True,
                featureNames=True, keepFeatures=['threes', 'threes'])
            assert False
        except InvalidArgumentValue:
            pass
        try:
            nimble.data(
                returnType='List', source=tmpCSV.name, pointNames=True,
                featureNames=['ones', 'twos', 'threes'], keepFeatures=[1, 'twos'])
            assert False
        except InvalidArgumentValue:
            pass
        try:
            nimble.data(
                returnType='List', source=tmpCSV.name, pointNames=True,
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
            nimble.data(
                returnType='List', source=tmpCSV.name, pointNames=True,
                featureNames=True, keepPoints=[1, 1])
            assert False
        except InvalidArgumentValue:
            pass
        try:
            nimble.data(
                returnType='List', source=tmpCSV.name, pointNames=True,
                featureNames=True, keepPoints=[1, 'dubs'])
            assert False
        except InvalidArgumentValue:
            pass
        try:
            nimble.data(
                returnType='List', source=tmpCSV.name, pointNames=True,
                featureNames=True, keepPoints=['trips', 'trips'])
            assert False
        except InvalidArgumentValue:
            pass
        try:
            nimble.data(
                returnType='List', source=tmpCSV.name,
                pointNames=['single', 'dubs', 'trips'], featureNames=True,
                keepPoints=[1, 'dubs'])
            assert False
        except InvalidArgumentValue:
            pass
        try:
            nimble.data(
                returnType='List', source=tmpCSV.name,
                pointNames=['single', 'dubs', 'trips'], featureNames=True,
                keepPoints=['trips', 'trips'])
            assert False
        except InvalidArgumentValue:
            pass


def test_data_csv_keepPF_and_ignoreFlag():
    for t in returnTypes:
        fnames = ['threes']
        pnames = ['trips', 'dubs']
        data = [[333], [33]]
        fromList = nimble.data(
            returnType=t, source=data, pointNames=pnames, featureNames=fnames)

        # instantiate from csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("pns,ones,twos,threes\n")
            tmpCSV.write("single,1,2,3A\n")
            tmpCSV.write("dubs,11,22A,33\n")
            tmpCSV.write("trips,111,222,333\n")
            tmpCSV.flush()

            fromCSV = nimble.data(
                returnType=t, source=tmpCSV.name, pointNames=True,
                featureNames=True, keepPoints=[2, 'dubs'],
                keepFeatures=[1, 'threes'], ignoreNonNumericalFeatures=True)

            assert fromList == fromCSV


def test_data_keepPoints_csv_endAfterAllFound():
    wanted = nimble.data("Matrix", source=[[11, 22, 33], [1, 2, 3]])
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        # This line has an extra value - if it was actually read by the
        # csv input helper, it would cause an exception to be raised.
        tmpCSV.write("111,222,333,444\n")
        tmpCSV.flush()

        fromCSV = nimble.data("Matrix", source=tmpCSV.name, keepPoints=[1, 0])
        assert fromCSV == wanted


def test_data_keepPF_csv_nameAlignment_allNames():
    for t in nimble.core.data.available:
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

            fromCSVL = nimble.data(t, source=tmpCSV.name, pointNames=pNamesL,
                                   featureNames=fNamesL, keepPoints=[2, 1],
                                   keepFeatures=[1, 0])
            fromCSVD = nimble.data(t, source=tmpCSV.name, pointNames=pNamesD,
                                   featureNames=fNamesD, keepPoints=[2, 1],
                                   keepFeatures=[1, 0])

        keptPNames = ['third', 'second']
        keptFNames = ['two', 'one']
        keptData = [[222, 111], [22, 11]]
        expected = nimble.data(t, keptData, keptPNames, keptFNames)

        assert fromCSVL == expected
        assert fromCSVD == expected


def test_data_keepPF_csv_nameAlignment_keptNames():
    for t in nimble.core.data.available:
        # instantiate from csv file
        keptPNames = ['third', 'second']
        keptFNames = ['two', 'one']
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
            tmpCSV.write("1,2,3\n")
            tmpCSV.write("11,22,33\n")
            tmpCSV.write("111,222,333\n")
            tmpCSV.flush()

            fromCSVL = nimble.data(t, source=tmpCSV.name, pointNames=keptPNames,
                                   featureNames=keptFNames, keepPoints=[2, 1],
                                   keepFeatures=[1, 0])
            fromCSVD = nimble.data(t, source=tmpCSV.name, pointNames=keptPNames,
                                   featureNames=keptFNames, keepPoints=[2, 1],
                                   keepFeatures=[1, 0])

        keptPNames = ['third', 'second']
        keptFNames = ['two', 'one']
        keptData = [[222, 111], [22, 11]]
        expected = nimble.data(t, keptData, keptPNames, keptFNames)

        assert fromCSVL == expected
        assert fromCSVD == expected


@raises(InvalidArgumentValue)
def test_data_csv_keepPoints_keepingAllPointNames_index():
    data = [[111, 222, 333], [11, 22, 33], [1, 2, 3]]
    pnames = ['1', '2', '3']
    wanted = nimble.data("Matrix", source=data, pointNames=pnames)
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        # cannot assume that pnames contains all pointNames for data
        fromCSV = nimble.data(
            "Matrix", source=tmpCSV.name, pointNames=pnames, keepPoints=[2, 1, 0])


@raises(InvalidArgumentValue)
def test_data_csv_keepPoints_keepingAllPointNames_names():
    data = [[1, 2, 3], [11, 22, 33], [111, 222, 333]]
    pnames = ['1', '2', '3']
    wanted = nimble.data("Matrix", source=data, pointNames=pnames)
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        # cannot assume that pnames contains all pointNames for data
        fromCSV = nimble.data(
            "Matrix", source=tmpCSV.name, pointNames=pnames, keepPoints=['3', '2', '1'])


@raises(InvalidArgumentValue)
def test_data_csv_keepFeatures_keepingAllFeatureNames_index():
    data = [[2, 3, 1], [22, 33, 11], [222, 333, 111]]
    fnames = ['2', '3', '1']
    wanted = nimble.data("Matrix", source=data, featureNames=fnames)
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        # assume featureNames passed aligns with order of keepFeatures
        fromCSV = nimble.data(
            "Matrix", source=tmpCSV.name, featureNames=fnames, keepFeatures=[1, 2, 0])


@raises(InvalidArgumentValue)
def test_data_csv_keepFeatures_keepingAllFeatureNames_names():
    data = [[2, 3, 1], [22, 33, 11], [222, 333, 111]]
    fnames = ['b', 'c', 'a']
    wanted = nimble.data("Matrix", source=data, featureNames=fnames)
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        # assume featureNames passed aligns with order of keepFeatures
        fromCSV = nimble.data(
            "Matrix", source=tmpCSV.name, featureNames=['a', 'b', 'c'], keepFeatures=['b', 'c' ,'a'])


def test_data_csv_keepFeatures_reordersFeatureNames_fnamesTrue():
    data = [[22, 33, 11], [222, 333, 111]]
    fnames = ['2', '3', '1']
    wanted = nimble.data("Matrix", source=data, featureNames=fnames)
    # instantiate from csv file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w') as tmpCSV:
        tmpCSV.write("1,2,3\n")
        tmpCSV.write("11,22,33\n")
        tmpCSV.write("111,222,333\n")
        tmpCSV.flush()

        # reordered based on keepFeatures since featureNames extracted
        fromCSVNames = nimble.data(
            "Matrix", source=tmpCSV.name, featureNames=True, keepFeatures=fnames)
        assert fromCSVNames == wanted

        # reordered based on keepFeatures since featureNames extracted
        fromCSVIndex = nimble.data(
            "Matrix", source=tmpCSV.name, featureNames=True, keepFeatures=[1, 2, 0])
        assert fromCSVIndex == wanted

######################
### inputSeparator ###
######################

def test_data_csv_inputSeparatorAutomatic():
    wanted = nimble.data("Matrix", source=[[1,2,3], [4,5,6]])
    # instantiate from csv file
    for delimiter in [',', '\t', ' ', ':', ';', '|']:
        with tempfile.NamedTemporaryFile(mode='w') as tmpCSV:
            tmpCSV.write("1{0}2{0}3\n".format(delimiter))
            tmpCSV.write("4{0}5{0}6\n".format(delimiter))
            tmpCSV.flush()

            fromCSV = nimble.data("Matrix", source=tmpCSV.name)
            assert fromCSV == wanted

def test_data_csv_inputSeparatorSpecified():
    wanted = nimble.data("Matrix", source=[[1,2,3], [4,5,6]])
    # instantiate from csv file
    for delimiter in [',', '\t', ' ', ':', ';', '|']:
        with tempfile.NamedTemporaryFile(mode='w') as tmpCSV:
            tmpCSV.write("1{0}2{0}3\n".format(delimiter))
            tmpCSV.write("4{0}5{0}6\n".format(delimiter))
            tmpCSV.flush()

            fromCSV = nimble.data("Matrix", source=tmpCSV.name, inputSeparator=delimiter)
            assert fromCSV == wanted

@raises(FileFormatException)
def test_data_csv_inputSeparatorConfusion():
    with tempfile.NamedTemporaryFile(mode='w') as tmpCSV:
        tmpCSV.write("1,2;3\n")
        tmpCSV.write("4,5,6\n")
        tmpCSV.flush()

        fromCSV = nimble.data("Matrix", source=tmpCSV.name)

@raises(InvalidArgumentValue)
def test_data_csv_inputSeparatorNot1Character():
    with tempfile.NamedTemporaryFile(mode='w') as tmpCSV:
        tmpCSV.write("1,,2,,3\n")
        tmpCSV.write("4,,5,,6\n")
        tmpCSV.flush()

        fromCSV = nimble.data("Matrix", source=tmpCSV.name, inputSeparator=',,')


#########################################
# treatAsMissing and replaceMissingWith #
#########################################

def test_missingDefaults():
    for t in returnTypes:
        nan = numpy.nan
        data = [[1, 2, float('nan')], [numpy.nan, 5, 6], [7, None, 9], ["", "nan", "None"]]
        toTest = nimble.data(t, data)
        expData = [[1, 2, nan], [nan, 5, 6], [7, nan, 9], [nan, nan, nan]]
        expRet = nimble.data(t, expData)
        assert toTest == expRet

def test_handmadeReplaceMissingWith():
    for t in returnTypes:
        data = [[1, 2, float('nan')], [numpy.nan, 5, 6], [7, None, 9], ["", "nan", "None"]]
        toTest = nimble.data(t, data, replaceMissingWith=0)
        expData = [[1, 2, 0], [0, 5, 6], [7, 0, 9], [0, 0, 0]]
        expRet = nimble.data(t, expData)
        assert toTest == expRet

def test_numericalReplaceMissingWithNonNumeric():
    for t in returnTypes:
        data = [[1, 2, None], [None, 5, 6], [7, None, 9], [None, None, None]]
        toTest = nimble.data(t, data, replaceMissingWith="Missing")
        expData = [[1, 2, "Missing"], ["Missing", 5, 6], [7, "Missing", 9], ["Missing", "Missing", "Missing"]]
        expRet = nimble.data(t, expData)
        assert toTest == expRet

def test_handmadeTreatAsMissing():
    for t in returnTypes:
        nan = numpy.nan
        data = [[1, 2, ""], [nan, 5, 6], [7, "", 9], [nan, "nan", "None"]]
        missingList = [nan, "", 5]
        assert numpy.array(missingList).dtype != numpy.object_
        toTest = nimble.data(t, data, treatAsMissing=missingList)
        expData = [[1, 2, nan], [nan, nan, 6], [7, nan, 9], [nan, "nan", "None"]]
        expRet = nimble.data(t, expData, treatAsMissing=None)
        assert toTest == expRet

def test_handmadeConsiderAndReplaceMissingWith():
    for t in returnTypes:
        data = [[1, 2, "NA"], ["NA", 5, 6], [7, "NA", 9], ["NA", "NA", "NA"]]
        toTest = nimble.data(t, data, treatAsMissing=["NA"], replaceMissingWith=0)
        expData = [[1, 2, 0], [0, 5, 6], [7, 0, 9], [0, 0, 0]]
        expRet = nimble.data(t, expData)
        assert toTest == expRet

def test_replaceDataTypeMismatch():
    for t in returnTypes:
        data = [[1, 2, 99], [99, 5, 6], [7, 99, 9], [99, 99, 99]]
        toTest = nimble.data(t, data, treatAsMissing=[99], replaceMissingWith="")
        expData = [[1, 2, ""], ["", 5, 6], [7, "", 9], ["", "", ""]]
        expRet = nimble.data(t, expData, treatAsMissing=None)
        assert toTest == expRet

def test_keepNanAndReplaceAlternateMissing():
    for t in returnTypes:
        nan = numpy.nan
        data = [[1, 2, "NA"], [numpy.nan, 5, 6], [7, "NA", 9], ["NA", numpy.nan, "NA"]]
        toTest = nimble.data(t, data, treatAsMissing=["NA"], replaceMissingWith=-1)
        expData = [[1, 2, -1], [nan, 5, 6], [7, -1, 9], [-1, nan, -1]]
        expRet = nimble.data(t, expData, treatAsMissing=None)
        assert toTest == expRet

def test_treatAsMissingIsNone():
    for t in returnTypes:
        nan = numpy.nan
        data = [[1, 2, None], [None, 5, 6], [7, None, 9], ["", numpy.nan, ""]]
        toTest = nimble.data(t, data, treatAsMissing=None)
        notExpData = [[1,2, nan], [nan, 5, 6], [7, nan, 9], [nan, nan, nan]]
        notExpRet = nimble.data(t, notExpData, treatAsMissing=None)
        assert toTest != notExpRet

def test_DataOutputWithMissingDataTypes1D():
    for t in returnTypes:
        nan = numpy.nan
        expListOutput = [[1.0, 2.0, nan]]
        expMatrixOutput = numpy.array(expListOutput)
        expDataFrameOutput = pd.DataFrame(expListOutput)
        expSparseOutput = scipy.sparse.coo_matrix(expListOutput)

        orig1 = nimble.data(t, [1,2,"None"])
        orig2 = nimble.data(t, (1,2,"None"))
        orig3 = nimble.data(t, {'a':1, 'b':2, 'c':"None"})
        orig3.features.sort()
        orig4 = nimble.data(t, [{'a':1, 'b':2, 'c':"None"}])
        orig4.features.sort()
        orig5 = nimble.data(t, numpy.array([1,2,"None"], dtype=object))
        orig6 = nimble.data(t, numpy.matrix([1,2,"None"], dtype=object))
        orig7 = nimble.data(t, pd.DataFrame([[1,2,"None"]]))
        orig8 = nimble.data(t, pd.Series([1,2,"None"]))
        orig9 = nimble.data(t, scipy.sparse.coo_matrix(numpy.array([1,2,"None"], dtype=object)))
        orig10 = nimble.data(t, scipy.sparse.csc_matrix(numpy.array([1,2,float('nan')])))
        orig11 = nimble.data(t, scipy.sparse.csr_matrix(numpy.array([1,2,float('nan')])))
        try:
            orig12 = nimble.data(t, pd.DataFrame([[1,2,"None"]], dtype='Sparse[object]'))
        except TypeError:
            orig12 = nimble.data(t, pd.SparseDataFrame([[1,2,"None"]]))

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
        expMatrixOutput = numpy.array(expListOutput, dtype=object)
        expDataFrameOutput = pd.DataFrame(expMatrixOutput)
        expSparseOutput = scipy.sparse.coo_matrix(expMatrixOutput)

        orig1 = nimble.data(t, [[1,2,'None'], [3,4,'b']])
        orig2 = nimble.data(t, ((1,2,'None'), (3,4,'b')))
        orig3 = nimble.data(t, {'a':[1,3], 'b':[2,4], 'c':['None', 'b']})
        orig3.features.sort()
        orig4 = nimble.data(t, [{'a':1, 'b':2, 'c':'None'}, {'a':3, 'b':4, 'c':'b'}])
        orig4.features.sort()
        orig5 = nimble.data(t, numpy.array([[1,2,'None'], [3,4,'b']], dtype=object))
        orig6 = nimble.data(t, numpy.matrix([[1,2,'None'], [3,4,'b']], dtype=object))
        orig7 = nimble.data(t, pd.DataFrame([[1,2,'None'], [3,4,'b']]))
        orig8 = nimble.data(t, scipy.sparse.coo_matrix(numpy.array([[1,2,'None'], [3,4,'b']], dtype=object)))
        try:
            orig9 = nimble.data(t, pd.DataFrame([[1,2,'None'], [3,4,'b']], dtype='Sparse[object]'))
        except TypeError:
            orig9 = nimble.data(t, pd.SparseDataFrame([[1,2,'None'], [3,4,'b']]))

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

def test_replaceNumpyValues_dtypePreservation():
    for t in returnTypes:
        data = numpy.array([[True, False, True], [False, True, False]])
        toTest = nimble.data(t, data, replaceMissingWith=2,
                             treatAsMissing=[False])
        # should upcast to int, since replaceMissingWith is int
        if hasattr(toTest.data, 'dtype'):
            assert toTest.data.dtype == int
        assert toTest[0, 0] == True # could be 1 or True depending on type
        assert toTest[0, 1] == 2

        data = numpy.array([[1, 0, 1], [0, 1, 0]])
        toTest = nimble.data(t, data, replaceMissingWith=numpy.nan,
                             treatAsMissing=[None])
        # should skip attempted replacement because no treatAsMissing values
        if hasattr(toTest.data, 'dtype'):
            assert toTest.data.dtype == int
        ints = (int, numpy.integer)
        assert all(isinstance(val, ints) for val in toTest.iterateElements())

        toTest = nimble.data(t, data, replaceMissingWith=numpy.nan,
                             treatAsMissing=[0])
        # should upcast to float, since replaceMissingWith is float
        if hasattr(toTest.data, 'dtype'):
            assert toTest.data.dtype == float
        assert toTest[0, 0] == True # could be 1.0 or True depending on type
        assert numpy.isnan(toTest[0, 1])


        toTest = nimble.data(t, data, replaceMissingWith='x',
                             treatAsMissing=[0])
        # should upcast to object, since replaceMissingWith is a string
        if hasattr(toTest.data, 'dtype'):
            assert toTest.data.dtype == numpy.object_
        assert toTest[0, 0] == True
        assert toTest[0, 1] == 'x'

#################
# Logging count #
#################
def test_data_logCount():
    """Test data adds one entry to the log for each return type"""

    @oneLogEntryExpected
    def byType(rType):
        toTest = nimble.data(rType, [[1,2,3], [4,5,6], [7,8,9]])

    for t in returnTypes:
        byType(t)

def makeTensorData(matrix):
    rank3List = [matrix, matrix, matrix]
    rank4List = [rank3List, rank3List, rank3List]
    rank5List = [rank4List, rank4List, rank4List]
    rank3Array = numpy.array(rank3List)
    rank4Array = numpy.array(rank4List)
    rank5Array = numpy.array(rank5List)
    rank3Array2D = numpy.empty((3, 3), dtype='O')
    for i, lst in enumerate(rank3List):
        rank3Array2D[i] = lst
    rank4Array2D = numpy.empty((3, 3), dtype='O')
    for i, lst in enumerate(rank4List):
        rank4Array2D[i] = lst
    rank5Array2D = numpy.empty((3, 3), dtype='O')
    for i, lst in enumerate(rank5List):
        rank5Array2D[i] = lst
    rank3DF = pd.DataFrame(rank3Array2D)
    rank4DF = pd.DataFrame(rank4Array2D)
    rank5DF = pd.DataFrame(rank5Array2D)
    rank3COO = scipy.sparse.coo_matrix(rank3Array2D)
    rank4COO = scipy.sparse.coo_matrix(rank4Array2D)
    rank5COO = scipy.sparse.coo_matrix(rank5Array2D)

    tensors = [rank3List, rank4List, rank5List, rank3Array, rank4Array, rank5Array,
               rank3Array2D, rank4Array2D, rank5Array2D, rank3DF, rank4DF, rank5DF]
    # cannot construct high dimension empty tensors for sparse
    if rank3Array.shape[-1] > 0:
        tensors.extend([rank3COO, rank4COO, rank5COO])

    return tensors


def test_data_multidimensionalData():
    vector1 = [0, 1, 2, 3, 0]
    vector2 = [4, 5, 0, 6, 7]
    vector3 = [8, 0, 9, 0, 8]
    matrix = [vector1, vector2, vector3]

    tensors = makeTensorData(matrix)

    emptyTensors = makeTensorData([[], [], []])

    expPoints = 3
    for retType in returnTypes:
        for idx, tensor in enumerate(tensors):
            toTest = nimble.data(retType, tensor)
            expShape = [3, 3, 5]
            for i in range(idx % 3):
                expShape.insert(0, 3)
            expFeatures = numpy.prod(expShape[1:])
            assert toTest._shape == expShape
            assert toTest._pointCount == expPoints
            assert toTest._featureCount == expFeatures

        for idx, tensor in enumerate(emptyTensors):
            toTest = nimble.data(retType, tensor)
            expShape = [3, 3, 0]
            for i in range(idx % 3):
                expShape.insert(0, 3)
            expFeatures = numpy.prod(expShape[1:])
            assert toTest._shape == expShape
            assert toTest._pointCount == expPoints
            assert toTest._featureCount == expFeatures

def test_data_multidimensionalData_pointNames():
    vector1 = [0, 1, 2, 3, 0]
    vector2 = [4, 5, 0, 6, 7]
    vector3 = [8, 0, 9, 0, 8]
    matrix = [vector1, vector2, vector3]

    tensors = makeTensorData(matrix)

    ptNames = ['a', 'b', 'c']
    for retType in returnTypes:
        for tensor in tensors:
            toTest = nimble.data(retType, tensor, pointNames=ptNames)
            assert toTest.points.getNames() == ptNames

def test_data_multidimensionalData_featureNames():
    vector1 = [0, 1, 2, 3, 0]
    vector2 = [4, 5, 0, 6, 7]
    vector3 = [8, 0, 9, 0, 8]
    matrix = [vector1, vector2, vector3]

    tensors = makeTensorData(matrix)

    for retType in returnTypes:
        for idx, tensor in enumerate(tensors):
            flattenedLen = 15
            for i in range(idx % 3):
                flattenedLen *= 3
            ftNames = ['ft' + str(x) for x in range(flattenedLen)]
            toTest = nimble.data(retType, tensor, featureNames=ftNames)
            assert toTest.features.getNames() == ftNames

def test_data_multidimensionalData_listsOfMultiDimensionalObjects():
    for (rType1, rType2) in itertools.product(returnTypes, returnTypes):
        arr1D = numpy.array([1, 2, 3, 0])
        nim1D = nimble.data(rType1, [1, 2, 3, 0])

        fromListArr1D = nimble.data(rType2, [arr1D, arr1D, arr1D])
        assert fromListArr1D._shape == [3, 4]
        fromListNim1D = nimble.data(rType2, [nim1D, nim1D, nim1D])
        assert fromListNim1D._shape == [3, 4]

        arr2D = fromListArr1D.data
        coo2D = scipy.sparse.coo_matrix([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 3, 0]])
        df2D = pd.DataFrame([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 3, 0]])
        nim2D = fromListNim1D

        fromListArr2D = nimble.data(rType2, [arr2D, arr2D, arr2D])
        assert fromListArr2D._shape == [3, 3, 4]
        fromListCoo2D = nimble.data(rType2, [coo2D, coo2D, coo2D])

        assert fromListCoo2D._shape == [3, 3, 4]
        fromListDF2D = nimble.data(rType2, [df2D, df2D, df2D])
        assert fromListDF2D._shape == [3, 3, 4]
        fromListNim2D = nimble.data(rType2, [nim2D, nim2D, nim2D])
        assert fromListNim2D._shape == [3, 3, 4]

        nim3D = fromListNim2D
        fromListNim3D = nimble.data(rType2, [nim3D, nim3D])
        assert fromListNim3D._shape == [2, 3, 3, 4]

# tests for combination of one name set being specified and one set being
# in data.


# test that if both in comment and specified names are present, the
# specified names win out.


# unit tests demonstrating our file loaders can handle arbitrarly placed blank lines


# comment lines
