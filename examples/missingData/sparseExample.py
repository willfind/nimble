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
	from UML import createData

	# path to input specified by command line argument
	pathIn = sys.argv[1]

	data1 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1],[0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]

	sparseAll = createData('Sparse', data=pathIn)
	sparseY = sparseAll.extractFeatures(sparseAll.data.shape[1] - 1)
	sparseX = sparseAll

	sparseAll2 = createData('Sparse', data=data1)
	sparseY2 = sparseAll2.extractFeatures(3)
	sparseX2 = sparseAll2

	ret = run('sciKitLearn.LinearRegression', trainX=sparseX, trainY=sparseY, testX=sparseX)

	ret2 = run('sciKitLearn.LinearRegression', trainX=sparseX2, trainY=sparseY2, testX=sparseX2)

	print "Raw results 1: " + str(ret.data)
	print "Raw results 2: " + str(ret2.data)

	ret.setFeatureName(0, 'result')

	Y = sparseY.copy(asType="Matrix")

	total = 0.
	correct = 0.
	for i in xrange(ret.pointCount):
		if ret.data[i,0] == Y.data[i,0]:
			correct = correct + 1
		total = total + 1

	print 'Correct ' + str(correct)
	print 'Total ' + str(total)
	print 'ratio ' + str(correct/total)

