
from nose.tools import *
from UML import createRandomizedData
import numpy


globalPoints = 100
globalFeatures = 200
globalSparsity = .5

def testReturnsFundamentalsCorrect():
	"""
	function that tests
	-the size of the underlying data is consistent with that requested through our API
	-the data class requested (Matrix, Sparse, List) is that which you get back
	-the data fundamental data type used to store the value of (point, feature) pairs
		is what the user requests ('int' or 'float')
	Note:
	These tests are run for all combinations of the paramaters:
		supportedFundamentalTypes = ['int', 'float']
		returnTypes = ['Matrix','Sparse','List']
		sparsities = [0.0, 0.5, .99]
	"""

	supportedFundamentalTypes = ['int', 'float']
	returnTypes = ['Matrix','Sparse','List']
	sparsities = [0.0, 0.5, .99]

	nPoints = 100
	nFeatures = 200
	#sparsity = .5

	for curType in supportedFundamentalTypes:
		for curReturnType in returnTypes:
			for curSparsity in sparsities:

				returned = createRandomizedData(curReturnType, nPoints, nFeatures, curSparsity, numericType=curType)
				
				assert(returned.pointCount == nPoints)
				assert(returned.featureCount == nFeatures)

				#assert that the requested numerical type was returned
				assert type(returned[0,0] == curType)


#note: makes calls to Base.data with assumptions about underlying datatstructure for sparse data
def testSparsityReturnedPlausible():
	"""
	function that tests:
	-for a dataset with 1500 points and 2000 features (2M pairs) that the number of 
		zero entries is reasonably close to the amount requested.
	Notes:
	-Because the generation of zeros is done stochastically, exact numbers of zeros
		is not informative. Instead, the test checks that the ratio of zeros to all
		points (zeros and non zeros) is within 1 percent of the 1 - sparsity.
	-These tests are run for all combinations of the paramaters:
		supportedFundamentalTypes = ['int', 'float']
		returnTypes = ['Matrix','Sparse','List']
		sparsities = [0.0, 0.5, .99]
	"""
	supportedFundamentalTypes = ['int', 'float']
	returnTypes = ['Matrix','Sparse','List']
	sparsities = [0.0, 0.5, .99]

	nPoints = 800
	nFeatures = 1000
	#sparsity = .5

	for curType in supportedFundamentalTypes:
		for curReturnType in returnTypes:
			for curSparsity in sparsities:
				returned = createRandomizedData(curReturnType, nPoints, nFeatures, curSparsity, numericType=curType)

				if curReturnType.lower() == 'matrix' or curReturnType.lower() == 'list':
					nonZerosCount = numpy.count_nonzero(returned.copyAs('numpyarray'))
					actualSparsity = 1.0 - nonZerosCount/float(nPoints * nFeatures)
					difference = abs(actualSparsity - curSparsity)
					
					assert(difference < .01)
					
				else: #is sparse matrix
					nonZerosCount = returned.data.nnz
					actualSparsity = 1.0 - nonZerosCount/float(nPoints * nFeatures)
					difference = abs(actualSparsity - curSparsity)
					
					assert(difference < .01)

#todo check that sizes of retunred mats are waht you request via npoints and nfeates
