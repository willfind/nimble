
from nose.tools import *
import tempfile
import scipy
import numpy

import UML

retTypes = ['List', 'Matrix', 'Sparse', None]  # None for auto

def test_simpleCSV():
	""" Test of createData() loading a csv file, default params """
	for t in retTypes:
		fromList = UML.createData(retType=t, data=[[1,2,3]])

		# instantiate from csv file
		tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv")
		tmpCSV.write("1,2,3\n")
		tmpCSV.flush()
		fromCSV = UML.createData(retType=t, data=tmpCSV.name)
		tmpCSV.close()
		assert fromList == fromCSV

def test_simpleMTXArr():
	""" Test of createData() loading a mtx (arr format) file, default params """
	for t in retTypes:
		fromList = UML.createData(retType=t, data=[[1,2,3]])

		# instantiate from mtx array file
		tmpMTXArr = tempfile.NamedTemporaryFile(suffix=".mtx")
		tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
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


def test_simpleMTXCoo():
	""" Test of createData() loading a mtx (coo format) file, default params """
	for t in retTypes:
		fromList = UML.createData(retType=t, data=[[1,2,3]])

		# instantiate from mtx coordinate file
		tmpMTXCoo = tempfile.NamedTemporaryFile(suffix=".mtx")
		tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
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

def test_namesInCommentCSV():
	""" Test of createData() loading a csv file and comment Names """
	pNames = ['pn1']
	fNames = ['one', 'two', 'three']
	for t in retTypes:
		fromList = UML.createData(retType=t, data=[[1,2,3]], pointNames=pNames, featureNames=fNames)

		# instantiate from csv file
		tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv")
		tmpCSV.write("#pn1\n")
		tmpCSV.write("#one,two,three\n")
		tmpCSV.write("1,2,3\n")
		tmpCSV.flush()
		fromCSV = UML.createData(retType=t, data=tmpCSV.name)
		tmpCSV.close()
		assert fromList == fromCSV

def test_namesInCommentMTXArr():
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

def test_namesInCommentMTXCoo():
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

def test_extractNamesCSV():
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


def test_extractNamesMTXArr():
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




def test_extractNamesMTXCoo():
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



def test_extractNamesList():
	""" Test of createData() given python list, extracting names """
	pNames = ['pn1'] 
	fNames = ['one', '2', 'three']

	for t in retTypes:
		inDataRaw = [['one', 2, 'three', 'foo'], [1,-1, -3, 'pn1']]
		specRaw = [[1,-1,-3]]
		inData = UML.createData(retType=t, data=inDataRaw, pointNames=3, featureNames=0)
		specified = UML.createData(retType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
		assert inData == specified

def test_extractNamesNPArray():
	""" Test of createData() given numpy array, extracting names """
	pNames = ['11'] 
	fNames = ['21', '22', '23']

	for t in retTypes:
		inDataRaw = numpy.array([[21,22,23,-111],[1,-1,-3,11]])
		specRaw = numpy.array([[1,-1,-3]])
		inData = UML.createData(retType=t, data=inDataRaw, pointNames=3, featureNames=0)
		specified = UML.createData(retType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
		assert inData == specified


def test_extractNamesNPMatrix():
	""" Test of createData() given numpy matrix, extracting names """
	pNames = ['11'] 
	fNames = ['21', '22', '23']

	for t in retTypes:
		inDataRaw = numpy.matrix([[21,22,23,-111],[1,-1,-3,11]])
		specRaw = numpy.matrix([[1,-1,-3]])
		inData = UML.createData(retType=t, data=inDataRaw, pointNames=3, featureNames=0)
		specified = UML.createData(retType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
		assert inData == specified


def test_extractNamesCooSparse():
	""" Test of createData() given scipy Coo matrix, extracting names """
	pNames = ['11'] 
	fNames = ['21', '22', '23']

	for t in retTypes:
		inDataRaw = numpy.array([[21,22,23,-111],[1,-1,-3,11]])
		inDataRaw = scipy.sparse.coo_matrix(inDataRaw)
		specRaw = numpy.array([[1,-1,-3]])
		specRaw = scipy.sparse.coo_matrix(specRaw)
#		import pdb
#		pdb.set_trace()
		inData = UML.createData(retType=t, data=inDataRaw, pointNames=3, featureNames=0)
		specified = UML.createData(retType=t, data=specRaw, pointNames=pNames, featureNames=fNames)
		assert inData == specified


# tests for combination of one name set being specified and one set being
# in data.


# test that if both in comment and specified names are present, the
# specified names win out.



# test fileType parameter : overide from extension, or no
# extension data
