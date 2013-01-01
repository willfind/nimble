"""
Class extending SparseBaseData, defining an object to hold and manipulate a scipy coo_matrix.

"""

from scipy.sparse import coo_matrix
from scipy.io import mmread
from scipy.io import mmwrite

from base_data import *
from dense_matrix_data import DenseMatrixData
from sparse_data import *
from ..utility.custom_exceptions import ArgumentException



class CooSparseData(SparseData):


	def __init__(self, data=None, featureNames=None):
		self.data = coo_matrix(data)
		super(CooSparseData, self).__init__(self.data,featureNames)


	def _extractColumns_implementation(self, toExtract, start, end, number, randomize):
		"""
		Function to extract columns according to the parameters, and return an object containing
		the removed columns with their featureName names from this object. The actual work is done by
		further helper functions, this determines which helper to call, and modifies the input
		to accomodate the number and randomize parameters, where number indicates how many of the
		possibilities should be extracted, and randomize indicates whether the choice of who to
		extract should be by order or uniform random.

		"""
		# single identifier
		if isinstance(toExtract, int):
			toExtract = [toExtract]	
		# list of identifiers
		if isinstance(toExtract, list):
			if number is None:
				number = len(toExtract)
			# if randomize, use random sample
			if randomize:
				toExtract = random.sample(toExtract, number)
			# else take the first number members of toExtract
			else:
				toExtract = toExtract[:number]
			# convert IDs if necessary
			toExtractIndices = []
			for value in toExtract:
				toExtractIndices.append(self._getIndex(value))
			return self._extractColumnsByList_implementation(toExtractIndices)
		# boolean function
		if hasattr(toExtract, '__call__'):
			if randomize:
				#apply to each
				raise NotImplementedError # TODO randomize in the extractRowByFunction case
			else:
				if number is None:
					number = self.rows()		
				return self._extractColumnsByFunction_implementation(toExtract, number)
		# by range
		if start is not None or end is not None:
			if start is None:
				start = 0
			if end is None:
				end = self.rows()
			if number is None:
				number = end - start
			if randomize:
				return self.extactColumnsByList(random.randrange(start,end,number))
			else:
				return self._extractColumnsByRange_implementation(start, end)


	def _extractColumnsByList_implementation(self,toExtract):
		"""
		

		"""
		extractData = []
		extractRows = []
		extractCols = []

		#walk through col listing and partition all data: extract, and kept, reusing the sparse matrix
		# underlying structure to save space
		copy = 0

		for i in xrange(len(self.data.col)):
			value = self.data.col[i]
			if value in toExtract:
				extractData.append(self.data.data[i])
				extractRows.append(self.data.row[i])
				extractCols.append(toExtract.index(value))
			else:
				self.data.data[copy] = self.data.data[i]				
				self.data.row[copy] = self.data.row[i]
				self.data.col[copy] = self.data.col[i] - _numLessThan(value, toExtract)
				copy = copy + 1

		# reinstantiate self
		# (cannot reshape coo matrices, so cannot do this in place)
		(rowShape, colShape) = self.data.shape
		self.data = coo_matrix( (self.data.data[0:copy],(self.data.row[0:copy],self.data.col[0:copy])), (rowShape, colShape - len(toExtract)))

		# instantiate return obj
		ret = coo_matrix((extractData,(extractRows,extractCols)),shape=(self.rows(), len(toExtract)))

		return CooSparseData(ret) 

	def _transpose_implementation(self):
		"""

		"""
		self.data = self.data.transpose()


	def _equals_implementation(self,other):
		if not isinstance(other, CooSparseData):
			return False
		if not self.data.data.tolist() == other.data.data.tolist():
			return False
		if not self.data.row.tolist() == other.data.row.tolist():
			return False
		if not self.data.col.tolist() == other.data.col.tolist():
			return False
		return True 

	def _columns_implementation(self):
		return self.data.shape[1]

	def _rows_implementation(self):
		return self.data.shape[0]


	def _convertToDenseMatrixData_implementation(self):
		""" Returns a DenseMatrixData object with the same data and featureNames as this object """
		return DenseMatrixData(self.data.todense(), self.featureNames)



def loadMM(inPath):
	"""
	Returns a CooSparseData object containing the data at the Market Matrix file specified by inPath.
	Uses the build in scipy function io.mmread().

	"""
	return CooSparseData(mmread(inPath))
	



def writeToMM(toWrite, outPath, includeFeatureNames):
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



def _numLessThan(value, toCheck): # TODO caching
	i = 0
	while i < len(toCheck):
		if toCheck[i] < value:
			i = i + 1
		else:
			break

	return i



