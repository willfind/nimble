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
	from UML import create

	# path to input specified by command line argument
	pathIn = sys.argv[1]

	data1 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1],[0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]

	sparseAll = create('Sparse', data=pathIn)
	sparseY = sparseAll.extractFeatures(sparseAll.data.shape[1] - 1)
	sparseX = sparseAll

	sparseAll2 = create('Sparse', data=data1)
	sparseY2 = sparseAll2.extractFeatures(3)
	sparseX2 = sparseAll2

	ret = run('sciKitLearn.LinearRegression', trainData=sparseX, testData=sparseX, dependentVar=sparseY)

	ret2 = run('sciKitLearn.LinearRegression', trainData=sparseX2, testData=sparseX2, dependentVar=sparseY2)

	print "Raw results 1: " + str(ret.data)
	print "Raw results 2: " + str(ret2.data)

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

