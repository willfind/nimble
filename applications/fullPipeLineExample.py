"""
Short module demonstrating the full pipeline of train - test - log results.
"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
	from UML import run
	from UML import normalize
	from UML import data
	from UML.performance.runner import runAndTest
	from UML.performance.metric_functions import classificationError

	variables = ["x1","x2","x3"]
	data1 = [[1,0,0], [3,3,3], [5,0,0],]
	trainObj = data('DenseMatrixData', data1, variables)

	data2 = [[1,0,0],[1,1,1],[5,1,1], [3,4,4]]
	testObj = data('DenseMatrixData', data2)

	metricFuncs = []
	metricFuncs.append(classificationError)

	results = runAndTest('sciKitLearn.KMeans',trainObj, testObj, trainDependentVar=2, testDependentVar=2, arguments={'n_clusters':3}, performanceMetricFuncs=metricFuncs, sendToLog=True)
