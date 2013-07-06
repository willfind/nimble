"""
Testing how type works with BaseData objects (based on numpy matrices) and
the applyFunctionToEachFeature() function in BaseData.
"""

import numpy as np

from allowImports import boilerplate
boilerplate()

from UML import create
from UML.logging.data_set_analyzer import produceFeaturewiseReport
from UML.logging.data_set_analyzer import produceAggregateReport

if __name__ == "__main__":
	variables = ["x","y","z"]
	variables2 = ['a', 'b', 'c', 'd']
	data1 = [[1,0,1], [3,3,3], [5,2,0]]
	trainObj1 = create('DenseMatrixData', data1, variables)
	data2 = np.array([[1,2,3, 0], [4,5,6, 100], [7,8,9, None]])
	trainObj2 = create('DenseMatrixData', data2, variables2)
	data3 = np.array([1, 4, 7])

	print "feature report: " + produceFeaturewiseReport(trainObj1)
	print "aggregate report: " + produceAggregateReport(trainObj1, displayDigits=2)

	print "feature report: " + produceFeaturewiseReport(trainObj2)
	print "aggregate report: " + produceAggregateReport(trainObj2, displayDigits=2)




