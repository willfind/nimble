"""
Script that takes the path given as the first command line argument, loads it as a sparse matrix,
calls the SciKit-Learn logistic regression classifier on it as both the training and test set,
and then counts (and prints) the number of points it correctly classified.

"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
	import sys

	from UML import run
	from UML import data

	# path to input specified by command line argument
	pathIn = sys.argv[1]

	sparseAll = data('CooSparseData',file=pathIn)
	sparseY = sparseAll.extractFeatures([0])
	sparseX = sparseAll

	ret = run('sciKitLearn', 'LogisticRegression', trainData=sparseX, testData=sparseX, dependentVar=sparseY)

	ret.renameFeatureName(0, 'result')

	Y = sparseY.toDenseMatrixData()

	total = 0.
	correct = 0.
	for i in xrange(ret.points()):
		if ret.data[i,0] == Y.data[i,0]:
			correct = correct + 1
		total = total + 1

	print 'Correct ' + str(correct)
	print 'Total ' + str(total)
	print 'ratio ' + str(correct/total)

