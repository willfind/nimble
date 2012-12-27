"""
Script that takes the path given as the first command line argument, loads it as a sparse matrix,
calls the SciKit-Learn logistic regression classifier on it as both the training and test set,
and then counts (and prints) the number of points it correctly classified.

"""

# PEP 366 'boilerplate', plus the necessary import of the top level package
if __name__ == "__main__" and __package__ is None:
	import sys
	# add UML parent directory to sys.path
	sys.path.append(sys.path[0].rsplit('/',2)[0])
	import UML
	import UML.applications
	__package__ = "UML.applications"

if __name__ == "__main__":
	import sys

	from ..interfaces.scikit_learn_interface import sciKitLearn as skl
	from ..processing.dense_matrix_data import DenseMatrixData
	from ..processing.coo_sparse_data import CooSparseData
	from ..processing.coo_sparse_data import loadMM as cooLoadMM

	# path to input specified by command line argument
	pathIn = sys.argv[1]

	sparseAll = cooLoadMM(pathIn)
	sparseY = sparseAll.extractColumns([0])
	sparseX = sparseAll

	ret = skl('LogisticRegression', trainData=sparseX, testData=sparseX, dependentVar=sparseY)

	ret.renameLabel(0,'result')

	Y = sparseY.convertToDenseMatrixData()

	total = 0.
	correct = 0.
	for i in xrange(ret.rows()):
		if ret.data[i,0] == Y.data[i,0]:
			correct = correct + 1
		total = total + 1

	print 'Correct ' + str(correct)
	print 'Total ' + str(total)
	print 'ratio ' + str(correct/total)

