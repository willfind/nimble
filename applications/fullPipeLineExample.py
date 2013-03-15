"""
Short module demonstrating the full pipeline of train - test - log results.
"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
	from UML import data
	from UML.performance.runner import runAndTest
	from UML.performance.runner import runOneVsOne
	from UML.performance.metric_functions import classificationError

	variables = ["x1","x2","x3", "label"]
	data1 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], 
	[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
	trainObj = data('DenseMatrixData', data1, variables)

	data2 = [[1,0,0,1],[0,1,0,2],[0,0,1,3]]
	testObj = data('DenseMatrixData', data2)

	metricFuncs = []
	metricFuncs.append(classificationError)

	#results1 = runAndTest('sciKitLearn.LogisticRegression',trainObj, testObj, trainDependentVar=3, testDependentVar=3, arguments=None, performanceMetricFuncs=metricFuncs)
	results2 = runOneVsOne('sciKitLearn.SVC',trainObj, testObj, trainDependentIndicator=3, testDependentIndicator=3, arguments={}, performanceMetricFuncs=metricFuncs)

	#print 'Standard run results: '+str(results1)
	print
	print 'One vs One results: '+str(results2)
