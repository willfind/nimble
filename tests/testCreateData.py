
from nose.tools import *
import tempfile
import scipy
import numpy
import os
import copy

import UML

#retTypes = ['List', 'Matrix', 'Sparse', None]  # None for auto
retTypes = copy.copy(UML.data.available)
retTypes.append(None)

###########################
# Data values correctness #
###########################

def test_createData_CSV_data():
	""" Test of createData() loading a csv file, default params """
	for t in retTypes:
		fromList = UML.createData(retType=t, data=[[1,2,3]])

		# instantiate from csv file
		with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
			tmpCSV.write("1,2,3\n")
			tmpCSV.flush()
			objName = 'fromCSV'
			fromCSV = UML.createData(retType=t, data=tmpCSV.name, name=objName)

			assert fromList == fromCSV

def test_createData_MTXArr_data():
	""" Test of createData() loading a mtx (arr format) file, default params """
	for t in retTypes:
		fromList = UML.createData(retType=t, data=[[1,2,3]])

		# instantiate from mtx array file
		with tempfile.NamedTemporaryFile(suffix=".mtx") as tmpMTXArr:
			tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
			tmpMTXArr.write("1 3\n")
			tmpMTXArr.write("1\n")
			tmpMTXArr.write("2\n")
			tmpMTXArr.write("3\n")
			tmpMTXArr.flush()
			objName = 'fromMTXArr'
			fromMTXArr = UML.createData(retType=t, data=tmpMTXArr.name, name=objName)
		
			if t is None and fromList.getTypeString() != fromMTXArr.getTypeString():
				assert fromList.isApproximatelyEqual(fromMTXArr)
			else:
				assert fromList == fromMTXArr


def test_createData_MTXCoo_data():
	""" Test of createData() loading a mtx (coo format) file, default params """
	for t in retTypes:
		fromList = UML.createData(retType=t, data=[[1,2,3]])

		# instantiate from mtx coordinate file
		with tempfile.NamedTemporaryFile(suffix=".mtx") as tmpMTXCoo:
			tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
			tmpMTXCoo.write("1 3 3\n")
			tmpMTXCoo.write("1 1 1\n")
			tmpMTXCoo.write("1 2 2\n")
			tmpMTXCoo.write("1 3 3\n")
			tmpMTXCoo.flush()
			objName = 'fromMTXCoo'
			fromMTXCoo = UML.createData(retType=t, data=tmpMTXCoo.name, name=objName)

			if t is None and fromList.getTypeString() != fromMTXCoo.getTypeString():
				assert fromList.isApproximatelyEqual(fromMTXCoo)
			else:
				assert fromList == fromMTXCoo


############################
# Name and path attributes #
############################


def test_createData_objName_and_path_CSV():
	for t in retTypes:
		# instantiate from csv file
		with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
			tmpCSV.write("1,2,3\n")
			tmpCSV.flush()

			objName = 'fromCSV'
			ret = UML.createData(retType=t, data=tmpCSV.name, name=objName)
			assert ret.name == objName
			assert ret.path == tmpCSV.name

			retDefName = UML.createData(retType=t, data=tmpCSV.name)
			tokens = tmpCSV.name.rsplit(os.path.sep)
			assert retDefName.name == tokens[len(tokens)-1]


def test_createData_objName_and_path_MTXArr():
	for t in retTypes:
		# instantiate from mtx array file
		with tempfile.NamedTemporaryFile(suffix=".mtx") as tmpMTXArr:
			tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
			tmpMTXArr.write("1 3\n")
			tmpMTXArr.write("1\n")
			tmpMTXArr.write("2\n")
			tmpMTXArr.write("3\n")
			tmpMTXArr.flush()
			
			objName = 'fromMTXArr'
			ret = UML.createData(retType=t, data=tmpMTXArr.name, name=objName)
			assert ret.name == objName
			assert ret.path == tmpMTXArr.name

			retDefName = UML.createData(retType=t, data=tmpMTXArr.name)
			tokens = tmpMTXArr.name.rsplit(os.path.sep)
			assert retDefName.name == tokens[len(tokens)-1]
			

def test_createData_objName_and_path_MTXCoo():
	for t in retTypes:
		# instantiate from mtx coordinate file
		with tempfile.NamedTemporaryFile(suffix=".mtx") as tmpMTXCoo:
			tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
			tmpMTXCoo.write("1 3 3\n")
			tmpMTXCoo.write("1 1 1\n")
			tmpMTXCoo.write("1 2 2\n")
			tmpMTXCoo.write("1 3 3\n")
			tmpMTXCoo.flush()

			objName = 'fromMTXCoo'
			ret = UML.createData(retType=t, data=tmpMTXCoo.name, name=objName)
			assert ret.name == objName
			assert ret.path == tmpMTXCoo.name

			retDefName = UML.createData(retType=t, data=tmpMTXCoo.name)
			tokens = tmpMTXCoo.name.rsplit(os.path.sep)
			assert retDefName.name == tokens[len(tokens)-1]

			

###################################
# Point / Feature names from File #
###################################

def test_extractNames_CSV():
	""" Test of createData() loading a csv file and extracting names """
	pNames = ['pn1'] 
	fNames = ['one', 'two', 'three']
	for t in retTypes:
		fromList = UML.createData(retType=t, data=[[1,2,3]], pointNames=pNames, featureNames=fNames)

		# instantiate from csv file
		tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv")
		tmpCSV.write("1,2,pn1,3\n")
		tmpCSV.write('one,two,ignore,three')
		tmpCSV.flush()
		fromCSV = UML.createData(retType=t, data=tmpCSV.name, pointNames=2, featureNames=1)
		tmpCSV.close()
		assert fromList == fromCSV


def test_names_AutoDetected_CSV():
	pNames = ['pn1']
	fNames = ['one', 'two', 'three']
	for t in retTypes:
		fromList = UML.createData(retType=t, data=[[1,2,3]], pointNames=pNames, featureNames=fNames)

		# instantiate from csv file
		tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv")
		tmpCSV.write("\n")
		tmpCSV.write("\n")
		tmpCSV.write("point_names,one,two,three\n")
		tmpCSV.write("pn1,1,2,3\n")
		tmpCSV.flush()

		fromCSV = UML.createData(retType=t, data=tmpCSV.name)
		tmpCSV.close()
		assert fromList == fromCSV


def test_featNamesOnly_AutoDetected_CSV():
	fNames = ['one', 'two', 'three']
	for t in retTypes:
		fromList = UML.createData(retType=t, data=[[1,2,3]], featureNames=fNames)

		# instantiate from csv file
		tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv")
		tmpCSV.write("\n")
		tmpCSV.write("\n")
		tmpCSV.write("one,two,three\n")
		tmpCSV.write("1,2,3\n")
		tmpCSV.flush()

		fromCSV = UML.createData(retType=t, data=tmpCSV.name)
		tmpCSV.close()
		assert fromList == fromCSV

def test_pointNames_AutoDetected_from_specified_featNames_CSV():
	fNames = ['one', 'two', 'three']
	pNames = ['pn1']
	for t in retTypes:
		fromList = UML.createData(retType=t, data=[[1,2,3]], pointNames=pNames, featureNames=fNames)

		# instantiate from csv file
		tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv")
		tmpCSV.write("\n")
		tmpCSV.write("\n")
		tmpCSV.write("point_names,one,two,three\n")
		tmpCSV.write("pn1,1,2,3\n")
		tmpCSV.flush()
		fromCSV = UML.createData(retType=t, data=tmpCSV.name, featureNames=0)
		tmpCSV.close()
		assert fromList == fromCSV


def test_extractNames_overides_autoDetect_CSV():
	pNames = ['pn1'] 
	fNames = ['one', 'two', 'three']
	for t in retTypes:
		fromList = UML.createData(retType=t, data=[[1,2,3]], pointNames=pNames, featureNames=fNames)

		# instantiate from csv file
		tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv")
		tmpCSV.write("\n")
		tmpCSV.write("\n")
		tmpCSV.write("1,2,pn1,3\n")
		tmpCSV.write('one,two,ignore,three')
		tmpCSV.flush()
		fromCSV = UML.createData(retType=t, data=tmpCSV.name, pointNames=2, featureNames=1)
		tmpCSV.close()
		assert fromList == fromCSV



def test_namesInComment_MTXArr():
	""" Test of createData() loading a mtx (arr format) file and comment Names """
	pNames = ['pn1']
	fNames = ['one', 'two', 'three']
	for t in retTypes:
		fromList = UML.createData(retType=t, data=[[1,2,3]], pointNames=pNames, featureNames=fNames)

		# instantiate from mtx array file
		tmpMTXArr = tempfile.NamedTemporaryFile(suffix=".mtx")
		tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
		tmpMTXArr.write("%#pn1\n")
		tmpMTXArr.write("%#one,two,three\n")
		tmpMTXArr.write("1 3\n")
		tmpMTXArr.write("1\n")
		tmpMTXArr.write("2\n")
		tmpMTXArr.write("3\n")
		tmpMTXArr.flush()
		fromMTXArr = UML.createData(retType=t, data=tmpMTXArr.name)
		tmpMTXArr.close()
		if t is None and fromList.getTypeString() != fromMTXArr.getTypeString():
			assert fromList.isApproximatelyEqual(fromMTXArr)
		else:
			assert fromList == fromMTXArr

def test_namesInComment_MTXCoo():
	""" Test of createData() loading a mtx (coo format) file and comment Names """
	pNames = ['pn1']
	fNames = ['one', 'two', 'three']
	for t in retTypes:
		fromList = UML.createData(retType=t, data=[[1,2,3]], pointNames=pNames, featureNames=fNames)

		# instantiate from mtx coordinate file
		tmpMTXCoo = tempfile.NamedTemporaryFile(suffix=".mtx")
		tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
		tmpMTXCoo.write("%#pn1\n")
		tmpMTXCoo.write("%#one,two,three\n")
		tmpMTXCoo.write("1 3 3\n")
		tmpMTXCoo.write("1 1 1\n")
		tmpMTXCoo.write("1 2 2\n")
		tmpMTXCoo.write("1 3 3\n")
		tmpMTXCoo.flush()
		fromMTXCoo = UML.createData(retType=t, data=tmpMTXCoo.name)
		tmpMTXCoo.close()
		if t is None and fromList.getTypeString() != fromMTXCoo.getTypeString():
			assert fromList.isApproximatelyEqual(fromMTXCoo)
		else:
			assert fromList == fromMTXCoo


def test_extractNames_MTXArr():
	""" Test of createData() loading a mtx (arr format) file and extracting names """
	pNames = ['11']
	fNames = ['21', '22', '23']
	for t in retTypes:
		fromList = UML.createData(retType=t, data=[[1,2,3]], pointNames=pNames, featureNames=fNames)

		# instantiate from mtx array file
		tmpMTXArr = tempfile.NamedTemporaryFile(suffix=".mtx")
		tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
		tmpMTXArr.write("2 4\n")
		tmpMTXArr.write("1\n")
		tmpMTXArr.write("21\n")
		tmpMTXArr.write("2\n")
		tmpMTXArr.write("22\n")
		tmpMTXArr.write("11\n")
		tmpMTXArr.write("-4\n")
		tmpMTXArr.write("3\n")
		tmpMTXArr.write("23\n")
		tmpMTXArr.flush()
		fromMTXArr = UML.createData(retType=t, data=tmpMTXArr.name, pointNames=2, featureNames=1)
		tmpMTXArr.close()
		if t is None and fromList.getTypeString() != fromMTXArr.getTypeString():
			assert fromList.isApproximatelyEqual(fromMTXArr)
		else:
			assert fromList == fromMTXArr



def test_extractNames_MTXCoo():
	""" Test of createData() loading a mtx (coo format) file and extracting names """
	pNames = ['11']
	fNames = ['21', '22', '23']
	for t in retTypes:
		fromList = UML.createData(retType=t, data=[[1,2,3]], pointNames=pNames, featureNames=fNames)

		# instantiate from mtx coordinate file
		tmpMTXCoo = tempfile.NamedTemporaryFile(suffix=".mtx")
		tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
		tmpMTXCoo.write("2 4 8\n")
		tmpMTXCoo.write("1 1 1\n")
		tmpMTXCoo.write("1 2 2\n")
		tmpMTXCoo.write("1 3 11\n")
		tmpMTXCoo.write("1 4 3\n")
		tmpMTXCoo.write("2 1 21.1\n")
		tmpMTXCoo.write("2 2 22\n")
		tmpMTXCoo.write("2 3 -5\n")
		tmpMTXCoo.write("2 4 23\n")
		tmpMTXCoo.flush()
		fromMTXCoo = UML.createData(retType=t, data=tmpMTXCoo.name, pointNames=2, featureNames=1)
		tmpMTXCoo.close()
		if t is None and fromList.getTypeString() != fromMTXCoo.getTypeString():
			assert fromList.isApproximatelyEqual(fromMTXCoo)
		else:
			assert fromList == fromMTXCoo




##################################
# Point / Feature names from Raw #
##################################


def test_extractNames_pythonList():
	""" Test of createData() given python list, extracting names """
	pNames = ['pn1'] 
	fNames = ['one', '2', 'three']

	for t in retTypes:
		inDataRaw = [['one', 2, 'three', 'foo'], [1,-1, -3, 'pn1']]
		specRaw = [[1,-1,-3]]
		inData = UML.createData(retType=t, data=inDataRaw, pointNames=3, featureNames=0)
		specified = UML.createData(retType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
		assert inData == specified

def test_extractNames_NPArray():
	""" Test of createData() given numpy array, extracting names """
	pNames = ['11'] 
	fNames = ['21', '22', '23']

	for t in retTypes:
		inDataRaw = numpy.array([[21,22,23,-111],[1,-1,-3,11]])
		specRaw = numpy.array([[1,-1,-3]])
		inData = UML.createData(retType=t, data=inDataRaw, pointNames=3, featureNames=0)
		specified = UML.createData(retType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
		assert inData == specified


def test_extractNames_NPMatrix():
	""" Test of createData() given numpy matrix, extracting names """
	pNames = ['11'] 
	fNames = ['21', '22', '23']

	for t in retTypes:
		inDataRaw = numpy.matrix([[21,22,23,-111],[1,-1,-3,11]])
		specRaw = numpy.matrix([[1,-1,-3]])
		inData = UML.createData(retType=t, data=inDataRaw, pointNames=3, featureNames=0)
		specified = UML.createData(retType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
		assert inData == specified


def test_extractNames_CooSparse():
	""" Test of createData() given scipy Coo matrix, extracting names """
	pNames = ['11'] 
	fNames = ['21', '22', '23']

	for t in retTypes:
		inDataRaw = numpy.array([[21,22,23,-111],[1,-1,-3,11]])
		inDataRaw = scipy.sparse.coo_matrix(inDataRaw)
		specRaw = numpy.array([[1,-1,-3]])
		specRaw = scipy.sparse.coo_matrix(specRaw)

		inData = UML.createData(retType=t, data=inDataRaw, pointNames=3, featureNames=0)
		specified = UML.createData(retType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
		assert inData == specified

def test_extractNames_CscSparse():
	""" Test of createData() given scipy Coo matrix, extracting names """
	pNames = ['11'] 
	fNames = ['21', '22', '23']

	for t in retTypes:
		inDataRaw = numpy.array([[21,22,23,-111],[1,-1,-3,11]])
		inDataRaw = scipy.sparse.coo_matrix(inDataRaw)
		specRaw = numpy.array([[1,-1,-3]])
		specRaw = scipy.sparse.coo_matrix(specRaw)

		inData = UML.createData(retType=t, data=inDataRaw, pointNames=3, featureNames=0)
		specified = UML.createData(retType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
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
	for t in retTypes:
		fromList = UML.createData(retType=t, data=[[1,2,3]])

		# instantiate from csv file
		with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
			tmpCSV.write("1,2,3\n")
			tmpCSV.flush()
			objName = 'fromCSV'
			openFile = open(tmpCSV.name, 'rU')
			fromCSV = UML.createData(retType=t, data=openFile, name=objName)
			openFile.close()

			assert fromList == fromCSV

			openFile = open(openFile.name, 'rU')
			namelessOpenFile = NamelessFile(openFile)
			fromCSV = UML.createData(retType=t, data=namelessOpenFile, fileType='csv')
			assert fromCSV.name.startswith(UML.data.dataHelpers.DEFAULT_NAME_PREFIX)
			assert fromCSV.path is None

def test_createData_MTXArr_passedOpen():
	for t in retTypes:
		fromList = UML.createData(retType=t, data=[[1,2,3]])

		# instantiate from mtx array file
		with tempfile.NamedTemporaryFile(suffix=".mtx") as tmpMTXArr:
			tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
			tmpMTXArr.write("1 3\n")
			tmpMTXArr.write("1\n")
			tmpMTXArr.write("2\n")
			tmpMTXArr.write("3\n")
			tmpMTXArr.flush()
			objName = 'fromMTXArr'
			openFile = open(tmpMTXArr.name, 'rU')
			fromMTXArr = UML.createData(retType=t, data=openFile, name=objName)
			openFile.close()

			if t is None and fromList.getTypeString() != fromMTXArr.getTypeString():
				assert fromList.isApproximatelyEqual(fromMTXArr)
			else:
				assert fromList == fromMTXArr

			openFile = open(tmpMTXArr.name, 'rU')
			namelessOpenFile = NamelessFile(openFile)
			fromMTXArr = UML.createData(retType=t, data=namelessOpenFile, fileType='mtx')
			assert fromMTXArr.name.startswith(UML.data.dataHelpers.DEFAULT_NAME_PREFIX)
			assert fromMTXArr.path is None


def test_createData_MTXCoo_passedOpen():
	for t in retTypes:
		fromList = UML.createData(retType=t, data=[[1,2,3]])

		# instantiate from mtx coordinate file
		with tempfile.NamedTemporaryFile(suffix=".mtx") as tmpMTXCoo:
			tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
			tmpMTXCoo.write("1 3 3\n")
			tmpMTXCoo.write("1 1 1\n")
			tmpMTXCoo.write("1 2 2\n")
			tmpMTXCoo.write("1 3 3\n")
			tmpMTXCoo.flush()
			objName = 'fromMTXCoo'
			openFile = open(tmpMTXCoo.name, 'rU')
			fromMTXCoo = UML.createData(retType=t, data=openFile, name=objName)
			openFile.close()

			if t is None and fromList.getTypeString() != fromMTXCoo.getTypeString():
				assert fromList.isApproximatelyEqual(fromMTXCoo)
			else:
				assert fromList == fromMTXCoo

			openFile = open(tmpMTXCoo.name, 'rU')
			namelessOpenFile = NamelessFile(openFile)
			fromMTXCoo = UML.createData(retType=t, data=namelessOpenFile, fileType='mtx')
			assert fromMTXCoo.name.startswith(UML.data.dataHelpers.DEFAULT_NAME_PREFIX)
			assert fromMTXCoo.path is None





# tests for combination of one name set being specified and one set being
# in data.


# test that if both in comment and specified names are present, the
# specified names win out.



# test fileType parameter : overide from extension, or no
# extension data


# unit tests demonstrating our file loaders can handle arbitrarly placed blank lines
# comment lines
