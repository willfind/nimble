"""
Class extending SparseBaseData, defining an object to hold and manipulate a scipy csc_matrix.

"""

from scipy.sparse import csc_matrix
from scipy.io import mmread
from scipy.io import mmwrite
import numpy

from base_data import *
from dense_matrix_data import DenseMatrixData
from sparse_data import *
from ..utility.custom_exceptions import ArgumentException



class CscSparseData(SparseData):


	def __init__(self, data=None, featureNames=None):
		self.data = csc_matrix(data)
		super(CscSparseData, self).__init__(self.data,featureNames)


	def _extractColumns_implementation(self,toExtract):
		"""
		

		"""

		converted = self.data.todense()
		ret = converted[:,toExtract]
		converted = numpy.delete(converted,toExtract,1)
		self.data = csc_matrix(converted)


		return CscSparseData(ret)


	
	def _transpose_implementation(self):
		"""

		"""

		print self.data
		self.data = self.data.transpose()
		print self.data



	def _convertToDenseMatrixData_implementation(self):
		""" Returns a DenseMatrixData object with the same data and featureNames as this object """
		return DenseMatrixData(self.data.todense(), self.featureNames)




def loadMM(inPath):
	"""
	Returns a CscSparseData object containing the data at the Market Matrix file specified by inPath.
	Uses the build in scipy function io.mmread().

	"""
	return CscSparseData(mmread(inPath))
	



def writeToMM(toWrite, outPath, includeFeatureNames=False):
	"""

	"""
	if includeFeatureNames:
		featureNameString = "#"
		for i in xrange(toWrite.columns()):
			featureNameString += toWrite.featureNamesInverse[i]
			if not i == toWrite.columns() - 1:
				featureNameString += ','
		
		mmwrite(target=outPath, a=toWrite.data, comment=featureNameString)		
	else:
		mmwrite(target=outPath, a=toWrite.data)








