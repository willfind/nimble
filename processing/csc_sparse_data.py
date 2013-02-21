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


	def __init__(self, data=None, featureNames=None, name=None, path=None):
		self.data = csc_matrix(data)
		super(CscSparseData, self).__init__(self.data, featureNames, name, path)


	def _extractFeatures_implementation(self,toExtract):
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
		self.data = self.data.transpose()



	def _toDenseMatrixData_implementation(self):
		""" Returns a DenseMatrixData object with the same data and featureNames as this object """
		return DenseMatrixData(self.data.todense(), self.featureNames)


	def _writeFileMTX_implementation(self, outPath, includeFeatureNames):
		if includeFeatureNames:
			featureNameString = "#"
			for i in xrange(self.features()):
				featureNameString += self.featureNamesInverse[i]
				if not i == self.features() - 1:
					featureNameString += ','
			
			mmwrite(target=outPath, a=self.data, comment=featureNameString)		
		else:
			mmwrite(target=outPath, a=self.data)

	def _copyReferences_implementation(self, other):
		if not isinstance(other, CscSparseData):
			raise ArgumentException("Other must be the same type as this object")

		self.data = other.data


###########
# Helpers #
###########


def _readFile(file):
	# TODO do some kind of checking as to the the input file format
	return _readMM(file)

def _readMM(file):
	"""
	Returns a CooSparseData object containing the data at the Market Matrix file specified by 
	the file parameter. Uses the build in scipy function io.mmread().

	"""
	return (mmread(file), None)
	
	



