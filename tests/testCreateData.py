
from nose.tools import *
import tempfile
import scipy
import numpy
import os
import copy

import UML

#returnTypes = ['Matrix', 'Sparse', None]  # None for auto
returnTypes = copy.copy(UML.data.available)
returnTypes.append(None)

###########################
# Data values correctness #
###########################

def test_createData_CSV_data():
	""" Test of createData() loading a csv file, default params """
	for t in returnTypes:
		fromList = UML.createData(returnType=t, data=[[1,2,3]])

		# instantiate from csv file
		with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
			tmpCSV.write("1,2,3\n")
			tmpCSV.flush()
			objName = 'fromCSV'
			fromCSV = UML.createData(returnType=t, data=tmpCSV.name, name=objName)

			assert fromList == fromCSV

def test_createData_CSV_data_noComment():
	for t in returnTypes:
		fromList = UML.createData(returnType=t, data=[[1,2], [1,2]])

		# instantiate from csv file
		with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
			tmpCSV.write("1,2,#3\n")
			tmpCSV.write("1,2,3\n")
			tmpCSV.flush()
			objName = 'fromCSV'
			fromCSV = UML.createData(returnType=t, data=tmpCSV.name, name=objName, ignoreNonNumericalFeatures=True)

			assert fromList == fromCSV


def test_createData_CSV_data_ListOnly():
	fromList = UML.createData(returnType="List", data=[[1,2,'three'], [4,5,'six']])

	# instantiate from csv file
	with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
		tmpCSV.write("1,2,three\n")
		tmpCSV.write("4,5,six\n")
		tmpCSV.flush()
		objName = 'fromCSV'
		fromCSV = UML.createData(returnType="List", data=tmpCSV.name, name=objName)

		assert fromList == fromCSV

def test_createData_CSV_data_ListOnly_noComment():
	fromList = UML.createData(returnType="List", data=[[1,2,'three'], [4,5,'#six']])

	# instantiate from csv file
	with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
		tmpCSV.write("1,2,three\n")
		tmpCSV.write("4,5,#six\n")
		tmpCSV.flush()
		objName = 'fromCSV'
		fromCSV = UML.createData(returnType="List", data=tmpCSV.name, name=objName)

		assert fromList == fromCSV

def test_createData_MTXArr_data():
	""" Test of createData() loading a mtx (arr format) file, default params """
	for t in returnTypes:
		fromList = UML.createData(returnType=t, data=[[1,2,3]])

		# instantiate from mtx array file
		with tempfile.NamedTemporaryFile(suffix=".mtx") as tmpMTXArr:
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
		fromList = UML.createData(returnType=t, data=[[1,2,3]])

		# instantiate from mtx coordinate file
		with tempfile.NamedTemporaryFile(suffix=".mtx") as tmpMTXCoo:
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


############################
# Name and path attributes #
############################


def test_createData_objName_and_path_CSV():
	for t in returnTypes:
		# instantiate from csv file
		with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
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
			assert retDefName.name == tokens[len(tokens)-1]


def test_createData_objName_and_path_MTXArr():
	for t in returnTypes:
		# instantiate from mtx array file
		with tempfile.NamedTemporaryFile(suffix=".mtx") as tmpMTXArr:
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
			assert retDefName.name == tokens[len(tokens)-1]
			

def test_createData_objName_and_path_MTXCoo():
	for t in returnTypes:
		# instantiate from mtx coordinate file
		with tempfile.NamedTemporaryFile(suffix=".mtx") as tmpMTXCoo:
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
			assert retDefName.name == tokens[len(tokens)-1]

			

###################################
# Point / Feature names from File #
###################################

def test_extractNames_CSV():
	""" Test of createData() loading a csv file and extracting names """
	pNames = ['pn1'] 
	fNames = ['one', 'two', 'three']
	for t in returnTypes:
		fromList = UML.createData(returnType=t, data=[[1,2,3]], pointNames=pNames, featureNames=fNames)

		# instantiate from csv file
		tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv")
		tmpCSV.write("1,2,pn1,3\n")
		tmpCSV.write('one,two,ignore,three')
		tmpCSV.flush()

		fromCSV = UML.createData(returnType=t, data=tmpCSV.name, pointNames=2, featureNames=1)
		tmpCSV.close()
		assert fromList == fromCSV


def test_names_AutoDetected_CSV():
	pNames = ['pn1']
	fNames = ['one', 'two', 'three']
	for t in returnTypes:
		fromList = UML.createData(returnType=t, data=[[1,2,3]], pointNames=pNames, featureNames=fNames)

		# instantiate from csv file
		tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv")
		tmpCSV.write("\n")
		tmpCSV.write("\n")
		tmpCSV.write("point_names,one,two,three\n")
		tmpCSV.write("pn1,1,2,3\n")
		tmpCSV.flush()

		fromCSV = UML.createData(returnType=t, data=tmpCSV.name)
		tmpCSV.close()
		assert fromList == fromCSV


def test_featNamesOnly_AutoDetected_CSV():
	fNames = ['one', 'two', 'three']
	for t in returnTypes:
		fromList = UML.createData(returnType=t, data=[[1,2,3]], featureNames=fNames)

		# instantiate from csv file
		tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv")
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
		fromList = UML.createData(returnType=t, data=[[1,2,3]], pointNames=pNames, featureNames=fNames)

		# instantiate from csv file
		tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv")
		tmpCSV.write("\n")
		tmpCSV.write("\n")
		tmpCSV.write("point_names,one,two,three\n")
		tmpCSV.write("pn1,1,2,3\n")
		tmpCSV.flush()
		fromCSV = UML.createData(returnType=t, data=tmpCSV.name, featureNames=0)
		tmpCSV.close()
		assert fromList == fromCSV


def test_extractNames_overides_autoDetect_CSV():
	pNames = ['pn1'] 
	fNames = ['one', 'two', 'three']
	for t in returnTypes:
		fromList = UML.createData(returnType=t, data=[[1,2,3]], pointNames=pNames, featureNames=fNames)

		# instantiate from csv file
		tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv")
		tmpCSV.write("\n")
		tmpCSV.write("\n")
		tmpCSV.write("1,2,pn1,3\n")
		tmpCSV.write('one,two,ignore,three')
		tmpCSV.flush()
		fromCSV = UML.createData(returnType=t, data=tmpCSV.name, pointNames=2, featureNames=1)
		tmpCSV.close()
		assert fromList == fromCSV



def test_namesInComment_MTXArr():
	""" Test of createData() loading a mtx (arr format) file and comment Names """
	pNames = ['pn1']
	fNames = ['one', 'two', 'three']
	for t in returnTypes:
		fromList = UML.createData(returnType=t, data=[[1,2,3]], pointNames=pNames, featureNames=fNames)

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
		fromList = UML.createData(returnType=t, data=[[1,2,3]], pointNames=pNames, featureNames=fNames)

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
		fromMTXCoo = UML.createData(returnType=t, data=tmpMTXCoo.name)
		tmpMTXCoo.close()
		if t is None and fromList.getTypeString() != fromMTXCoo.getTypeString():
			assert fromList.isApproximatelyEqual(fromMTXCoo)
		else:
			assert fromList == fromMTXCoo


def test_extractNames_MTXArr():
	""" Test of createData() loading a mtx (arr format) file and extracting names """
	pNames = ['11']
	fNames = ['21', '22', '23']
	for t in returnTypes:
		fromList = UML.createData(returnType=t, data=[[1,2,3]], pointNames=pNames, featureNames=fNames)

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
		fromMTXArr = UML.createData(returnType=t, data=tmpMTXArr.name, pointNames=2, featureNames=1)
		tmpMTXArr.close()
		if t is None and fromList.getTypeString() != fromMTXArr.getTypeString():
			assert fromList.isApproximatelyEqual(fromMTXArr)
		else:
			assert fromList == fromMTXArr



def test_extractNames_MTXCoo():
	""" Test of createData() loading a mtx (coo format) file and extracting names """
	pNames = ['11']
	fNames = ['21', '22', '23']
	for t in returnTypes:
		fromList = UML.createData(returnType=t, data=[[1,2,3]], pointNames=pNames, featureNames=fNames)

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
		fromMTXCoo = UML.createData(returnType=t, data=tmpMTXCoo.name, pointNames=2, featureNames=1)
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

	for t in returnTypes:
		inDataRaw = [['one', 2, 'three', 'foo'], [1,-1, -3, 'pn1']]
		specRaw = [[1,-1,-3]]
		inData = UML.createData(returnType=t, data=inDataRaw, pointNames=3, featureNames=0)
		specified = UML.createData(returnType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
		assert inData == specified

def test_extractNames_NPArray():
	""" Test of createData() given numpy array, extracting names """
	pNames = ['11'] 
	fNames = ['21', '22', '23']

	for t in returnTypes:
		inDataRaw = numpy.array([[21,22,23,-111],[1,-1,-3,11]])
		specRaw = numpy.array([[1,-1,-3]])
		inData = UML.createData(returnType=t, data=inDataRaw, pointNames=3, featureNames=0)
		specified = UML.createData(returnType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
		assert inData == specified


def test_extractNames_NPMatrix():
	""" Test of createData() given numpy matrix, extracting names """
	pNames = ['11'] 
	fNames = ['21', '22', '23']

	for t in returnTypes:
		inDataRaw = numpy.matrix([[21,22,23,-111],[1,-1,-3,11]])
		specRaw = numpy.matrix([[1,-1,-3]])
		inData = UML.createData(returnType=t, data=inDataRaw, pointNames=3, featureNames=0)
		specified = UML.createData(returnType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
		assert inData == specified


def test_extractNames_CooSparse():
	""" Test of createData() given scipy Coo matrix, extracting names """
	pNames = ['11'] 
	fNames = ['21', '22', '23']

	for t in returnTypes:
		inDataRaw = numpy.array([[21,22,23,-111],[1,-1,-3,11]])
		inDataRaw = scipy.sparse.coo_matrix(inDataRaw)
		specRaw = numpy.array([[1,-1,-3]])
		specRaw = scipy.sparse.coo_matrix(specRaw)

		inData = UML.createData(returnType=t, data=inDataRaw, pointNames=3, featureNames=0)
		specified = UML.createData(returnType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
		assert inData == specified

def test_extractNames_CscSparse():
	""" Test of createData() given scipy Coo matrix, extracting names """
	pNames = ['11'] 
	fNames = ['21', '22', '23']

	for t in returnTypes:
		inDataRaw = numpy.array([[21,22,23,-111],[1,-1,-3,11]])
		inDataRaw = scipy.sparse.coo_matrix(inDataRaw)
		specRaw = numpy.array([[1,-1,-3]])
		specRaw = scipy.sparse.coo_matrix(specRaw)

		inData = UML.createData(returnType=t, data=inDataRaw, pointNames=3, featureNames=0)
		specified = UML.createData(returnType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
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
		fromList = UML.createData(returnType=t, data=[[1,2,3]])

		# instantiate from csv file
		with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
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
			fromCSV = UML.createData(returnType=t, data=namelessOpenFile, fileType='csv')
			assert fromCSV.name.startswith(UML.data.dataHelpers.DEFAULT_NAME_PREFIX)
			assert fromCSV.path is None
			assert fromCSV.absolutePath is None
			assert fromCSV.relativePath is None


def test_createData_MTXArr_passedOpen():
	for t in returnTypes:
		fromList = UML.createData(returnType=t, data=[[1,2,3]])

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
			fromMTXArr = UML.createData(returnType=t, data=namelessOpenFile, fileType='mtx')
			assert fromMTXArr.name.startswith(UML.data.dataHelpers.DEFAULT_NAME_PREFIX)
			assert fromMTXArr.path is None
			assert fromMTXArr.absolutePath is None
			assert fromMTXArr.relativePath is None


def test_createData_MTXCoo_passedOpen():
	for t in returnTypes:
		fromList = UML.createData(returnType=t, data=[[1,2,3]])

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
			fromMTXCoo = UML.createData(returnType=t, data=namelessOpenFile, fileType='mtx')
			assert fromMTXCoo.name.startswith(UML.data.dataHelpers.DEFAULT_NAME_PREFIX)
			assert fromMTXCoo.path is None
			assert fromMTXCoo.absolutePath is None
			assert fromMTXCoo.relativePath is None




###################################
# ignoreNonNumericalFeatures flag #
###################################


def test_createData_ignoreNonNumericalFeaturesCSV():
	for t in returnTypes:
		fromList = UML.createData(returnType=t, data=[[1,3], [5,7]])

		# instantiate from csv file
		with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
			tmpCSV.write("1,two,3.0,four\n")
			tmpCSV.write("5,six,7,8\n")
			tmpCSV.flush()

			fromCSV = UML.createData(returnType=t, data=tmpCSV.name, ignoreNonNumericalFeatures=True)

			assert fromList == fromCSV

			if t == 'List':
				fromCSV = UML.createData(returnType=t, data=tmpCSV.name)
				assert fromCSV.featureCount == 4

def test_createData_ignoreNonNumerical_removalCleanup_hard():
	for t in returnTypes:
		fromList = UML.createData(returnType=t, data=[[1,3], [5,7],[11,12],[13,14]])

		# instantiate from csv file
		with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
			tmpCSV.write("1,2,3.0,4.0,1\n")
			tmpCSV.write("5,six,7,8,1\n")
			tmpCSV.write("11,6,12,eight,1.0\n")
			tmpCSV.write("13,one,14,9,who?\n")
			tmpCSV.flush()

			fromCSV = UML.createData(returnType=t, data=tmpCSV.name, ignoreNonNumericalFeatures=True)

			assert fromList == fromCSV

			if t == 'List':
				fromCSV = UML.createData(returnType=t, data=tmpCSV.name)
				assert fromCSV.featureCount == 5

def test_createData_ignoreNonNumerical_removalCleanup_easy():
	for t in returnTypes:
		fromList = UML.createData(returnType=t, data=[[1,3], [5,7],[11,12],[13,14]])

		# instantiate from csv file
		with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
			tmpCSV.write("1,two,3.0,four,one\n")
			tmpCSV.write("5,6,7,8,1\n")
			tmpCSV.write("11,6,12,8,1.0\n")
			tmpCSV.write("13,1,14,9,2\n")
			tmpCSV.flush()

			fromCSV = UML.createData(returnType=t, data=tmpCSV.name, ignoreNonNumericalFeatures=True)

			assert fromList == fromCSV

			if t == 'List':
				fromCSV = UML.createData(returnType=t, data=tmpCSV.name)
				assert fromCSV.featureCount == 5



def test_createData_ignoreNonNumericalFeaturesCSV_noEffect():
	for t in returnTypes:
		fromList = UML.createData(returnType=t, data=[[1,2,3,4], [5,6,7,8]])

		# instantiate from csv file
		with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
			tmpCSV.write("1,2,3,4\n")
			tmpCSV.write("5,6,7,8\n")
			tmpCSV.flush()

			fromCSV = UML.createData(returnType=t, data=tmpCSV.name, ignoreNonNumericalFeatures=True)

			assert fromList == fromCSV

			if t == 'List':
				fromCSV = UML.createData(returnType=t, data=tmpCSV.name)
				assert fromCSV.featureCount == 4


def test_ignoreNonNumericalFeatures_featureNamesDontTrigger():
	for t in returnTypes:
		fnames = ['1','2','3','four']
		fromList = UML.createData(returnType=t, featureNames=fnames, data=[[5,6,7,8]])

		# instantiate from csv file
		with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
			tmpCSV.write("1,2,3,four\n")
			tmpCSV.write("5,6,7,8\n")
			tmpCSV.flush()

			fromCSV = UML.createData(returnType=t, data=tmpCSV.name, featureNames=0, ignoreNonNumericalFeatures=True)

			assert fromList == fromCSV


def test_ignoreNonNumericalFeatures_featureNamesAdjusted():
	for t in returnTypes:
		fNames = ["1", "2", "3"]
		data = [[1,2,3], [5,6,7]]
		fromList = UML.createData(returnType=t, featureNames=fNames, data=data)

		# instantiate from csv file
		with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
			tmpCSV.write("1,2,3,4\n")
			tmpCSV.write("1,2,3,four\n")
			tmpCSV.write("5,6,7,H8\n")
			tmpCSV.flush()

			fromCSV = UML.createData(returnType=t, data=tmpCSV.name, featureNames=0, ignoreNonNumericalFeatures=True)

			assert fromList == fromCSV

def test_createData_ignoreNonNumericalFeatures_allRemoved():
	for t in returnTypes:
		pNames = ['single', 'dubs', 'trips']
		fromList = UML.createData(returnType=t, pointNames=pNames, data=[[],[],[]])

		# instantiate from csv file
		with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
			tmpCSV.write(",ones,twos,threes\n")
			tmpCSV.write("single,1A,2A,3A\n")
			tmpCSV.write("dubs,11,22A,33\n")
			tmpCSV.write("trips,111,222,333\n")
			tmpCSV.flush()

			fromCSV = UML.createData(returnType=t, data=tmpCSV.name, pointNames=0, featureNames=0, ignoreNonNumericalFeatures=True)

			assert fromList == fromCSV


####################################################
# Difficult CSV Formatting: whitespace and quoting #
####################################################

def test_formatting_simpleQuotedValues():
	for t in returnTypes:
		fromList = UML.createData(returnType=t, data=[[1,2,3,4], [5,6,7,8]])

		# instantiate from csv file
		with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
			tmpCSV.write("1,\"2\",\"3\",4\n")
			tmpCSV.write("5,\"6\",\"7\",8\n")
			tmpCSV.flush()

			fromCSV = UML.createData(returnType=t, data=tmpCSV.name)

			assert fromList == fromCSV

def test_formatting_specialCharsInQuotes():
	for t in returnTypes:
		fNames = ["1,ONE", "2;TWO", "3\t'EE'"]
		data = [[1,2,3], [5,6,7]]
		dataAll = [[1,2,3,4], [5,6,7,8]]
		fromList = UML.createData(returnType=t, featureNames=fNames[:3], data=data)

		# instantiate from csv file
		with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
			tmpCSV.write("\"1,ONE\",\"2;TWO\",\"3\t'EE'\",\"4f\"\n")
			tmpCSV.write("1,2,3,four\n")
			tmpCSV.write("5,6,7,H8\n")
			tmpCSV.flush()

			fromCSV = UML.createData(returnType=t, data=tmpCSV.name, featureNames=0, ignoreNonNumericalFeatures=True)

			assert fromList == fromCSV


def test_formatting_emptyAndCommentLines():
	for t in returnTypes:
		data = [[1,2,3,4],[5,6,7,8]]
		fromList = UML.createData(returnType=t, data=data)

		# instantiate from csv file
		with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
			tmpCSV.write("#1,2,3,4\n")
			tmpCSV.write("1,2,3,4\n")
			tmpCSV.write("#second line\n")
			tmpCSV.write("\n")
			tmpCSV.write("5,6,7,8\n")
			tmpCSV.write("\n")
			tmpCSV.write("#END\n")
			tmpCSV.flush()

			fromCSV = UML.createData(returnType=t, data=tmpCSV.name)

			assert fromList == fromCSV


# tests for combination of one name set being specified and one set being
# in data.


# test that if both in comment and specified names are present, the
# specified names win out.



# test fileType parameter : overide from extension, or no
# extension data


# unit tests demonstrating our file loaders can handle arbitrarly placed blank lines
# comment lines
