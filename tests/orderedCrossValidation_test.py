

import datetime

from UML import create
from UML import orderedCrossValidate


def createAll(inData, featureNames=None):
	rld = create("RowListData", inData, featureNames)
	dmd = create("DenseMatrixData", inData, featureNames)

	return (rld, dmd)

def returnNone(**args):
	return None

def checkResults(results, expStartTrain, expEndTrain, expStartTest, expEndTest):
#	print results
	for key in results:
		trialsList = results[key]
		for i in range(len(expStartTrain)):
			assert trialsList[i]["startTrain"] == expStartTrain[i]
			assert trialsList[i]["endTrain"] == expEndTrain[i]
			assert trialsList[i]["startTest"] == expStartTest[i]
			assert trialsList[i]["endTest"] == expEndTest[i]

def test_orderedCV_trialConstruction_actualData():
	""" Test of the orderedCrossValidate() passes the correct data to the functions"""
	rawX = [[1],[2],[3],[4],[5]]
	rawY = [[11],[22],[33],[44],[55]]

	toTestListX = createAll(rawX, ["value"])
	toTestListY = createAll(rawY, ["time"])

	def returnRawInput(**args):
		trainX = args['trainX'].toListOfLists()
		trainY = args['trainY'].toListOfLists()
		testX = args['testX'].toListOfLists()
		testY = args['testY'].toListOfLists()
		return (trainX, trainY, testX, testY)

	# call OCV for each data representation type
	for i in range(len(toTestListX)):
		dataX = toTestListX[i]
		dataY = toTestListY[i]
		results = orderedCrossValidate(dataX, dataY, [returnRawInput], "time", 1, 1, 1, 0, 1, 1)

	expectedTrials = []
	expectedTrials.append(([[1]],[[11]],[[2]],[[22]]))
	expectedTrials.append(([[2]],[[22]],[[3]],[[33]]))
	expectedTrials.append(([[3]],[[33]],[[4]],[[44]]))
	expectedTrials.append(([[4]],[[44]],[[5]],[[55]]))

	for key in results:
		trialsList = results[key]
		for i in range(len(trialsList)):
			print trialsList[i]["result"]
			print expectedTrials[i]
			assert trialsList[i]["result"] == expectedTrials[i]


def test_orderedCV_trialConstruction_multiValue_intParam():
	""" Test of the orderedCrossValidate() construction of non singleton trials"""
	rawX = [[1],[2],[3],[4],[5]]
	rawY = [[1],[2],[3],[4],[5]]

	toTestListX = createAll(rawX, ["value"])
	toTestListY = createAll(rawY, ["time"])

	# call OCV for each data representation type
	for i in range(len(toTestListX)):
		dataX = toTestListX[i]
		dataY = toTestListY[i]
		results = orderedCrossValidate(dataX, dataY, [returnNone], "time", 2, 2, 1, 0, 2, 2)

	expStartTrain = [0,1]
	expEndTrain = [1,2]
	expStartTest = [2,3]
	expEndTest = [3,4]

	checkResults(results, expStartTrain, expEndTrain, expStartTest, expEndTest)


def test_orderedCV_trialConstruction_noSplit_intParam():
	""" Test of the orderedCrossValidate() construction of trials which do not split the ordered value"""
	rawX = [[1],[2],[3],[4],[5],[6],[7],[8]]
	rawY = [[1],[1],[1],[2],[2],[2],[3],[3]]

	toTestListX = createAll(rawX, ["value"])
	toTestListY = createAll(rawY, ["time"])

	# call OCV for each data representation type
	for i in range(len(toTestListX)):
		dataX = toTestListX[i]
		dataY = toTestListY[i]
		results = orderedCrossValidate(dataX, dataY, [returnNone], "time", 2, 2, 1, 0, 2, 2)

	expStartTrain = [1,4]
	expEndTrain = [2,5]
	expStartTest = [3,6]
	expEndTest = [4,7]

	checkResults(results, expStartTrain, expEndTrain, expStartTest, expEndTest)


def test_orderedCV_trialConstruction_nonZeroGap_intParam():
	""" Test of the orderedCrossValidate() construction of trials with a nonzero gap size"""
	rawX = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
	rawY = [[1], [1], [1], [2], [2], [2], [3], [3], [3], [4]]

	toTestListX = createAll(rawX, ["value"])
	toTestListY = createAll(rawY, ["time"])

	# call OCV for each data representation type
	for i in range(len(toTestListX)):
		dataX = toTestListX[i]
		dataY = toTestListY[i]
		results = orderedCrossValidate(dataX, dataY, [returnNone], "time", 2, 2, 1, 2, 2, 2)

	expStartTrain = [1,4]
	expEndTrain = [2,5]
	expStartTest = [5,8]
	expEndTest = [6,9]

	checkResults(results, expStartTrain, expEndTrain, expStartTest, expEndTest)


def test_orderedCV_trialConstruction_minMaxDifferent_intParam():
	""" Test of the orderedCrossValidate() construction of trials where min and max sizes are different"""
	rawX = [[1], [2], [3], [4], [5], [6], [7]]
	rawY = [[1], [2], [3], [4], [5], [6], [7]]

	toTestListX = createAll(rawX, ["value"])
	toTestListY = createAll(rawY, ["time"])

	# call OCV for each data representation type
	for i in range(len(toTestListX)):
		dataX = toTestListX[i]
		dataY = toTestListY[i]
		results = orderedCrossValidate(dataX, dataY, [returnNone], "time", 1, 3, 1, 0, 1, 3)

	expStartTrain = [0,0,0,1,2,3]
	expEndTrain = [0,1,2,3,4,5]
	expStartTest = [1,2,3,4,5,6]
	expEndTest = [3,4,5,6,6,6]

	checkResults(results, expStartTrain, expEndTrain, expStartTest, expEndTest)




def test_orderedCV_trialConstruction_multiValue_dtParam():
	""" Test of the orderedCrossValidate() construction of non singleton trials with timedelta params"""
	rawX = [[1],[2],[3],[4],[5]]
	rawY = [[1],[2],[3],[4],[5]]

	toTestListX = createAll(rawX, ["value"])
	toTestListY = createAll(rawY, ["time"])

	minTrain = datetime.timedelta(1)
	maxTrain = datetime.timedelta(1)
	step = 1
	gap = 0
	minTest = datetime.timedelta(1)
	maxTest = datetime.timedelta(1)

#	set_trace()

	# call OCV for each data representation type
	for i in range(len(toTestListX)):
		dataX = toTestListX[i]
		dataY = toTestListY[i]
		results = orderedCrossValidate(dataX, dataY, [returnNone], "time", minTrain, maxTrain, step, gap, minTest, maxTest)

	expStartTrain = [0,1]
	expEndTrain = [1,2]
	expStartTest = [2,3]
	expEndTest = [3,4]

	checkResults(results, expStartTrain, expEndTrain, expStartTest, expEndTest)


def test_orderedCV_trialConstruction_noSplit_dtParam():
	""" Test of the orderedCrossValidate() construction of trials which do not split the ordered value with timedelta params"""
	rawX = [[1],[2],[3],[4],[5],[6],[7],[8]]
	rawY = [[1],[1],[1],[2],[2],[2],[3],[3]]

	toTestListX = createAll(rawX, ["value"])
	toTestListY = createAll(rawY, ["time"])

	minTrain = datetime.timedelta(1)
	maxTrain = datetime.timedelta(1)
	step = 1
	gap = 0
	minTest = datetime.timedelta(1)
	maxTest = datetime.timedelta(1)

	# call OCV for each data representation type
	for i in range(len(toTestListX)):
		dataX = toTestListX[i]
		dataY = toTestListY[i]
		results = orderedCrossValidate(dataX, dataY, [returnNone], "time", minTrain, maxTrain, step, gap, minTest, maxTest)

	expStartTrain = [1,4]
	expEndTrain = [2,5]
	expStartTest = [3,6]
	expEndTest = [4,7]

	checkResults(results, expStartTrain, expEndTrain, expStartTest, expEndTest)


def test_orderedCV_trialConstruction_nonZeroGap_dtParam():
	""" Test of the orderedCrossValidate() construction of trials with a nonzero gap size with timedelta params"""
	rawX = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
	rawY = [[1], [1], [1], [2], [2], [2], [3], [3], [3], [4]]

	toTestListX = createAll(rawX, ["value"])
	toTestListY = createAll(rawY, ["time"])

	minTrain = datetime.timedelta(1)
	maxTrain = datetime.timedelta(1)
	step = 1
	gap = datetime.timedelta(1)
	minTest = datetime.timedelta(1)
	maxTest = datetime.timedelta(1)

	# call OCV for each data representation type
	for i in range(len(toTestListX)):
		dataX = toTestListX[i]
		dataY = toTestListY[i]
		results = orderedCrossValidate(dataX, dataY, [returnNone], "time", minTrain, maxTrain, step, gap, minTest, maxTest)

	expStartTrain = [1,4]
	expEndTrain = [2,5]
	expStartTest = [5,8]
	expEndTest = [6,9]

	checkResults(results, expStartTrain, expEndTrain, expStartTest, expEndTest)


def test_orderedCV_trialConstruction_minMaxDifferent_dtParam():
	""" Test of the orderedCrossValidate() construction of trials where min and max sizes are different with timedelta params"""
	rawX = [[1], [2], [3], [4], [5], [6], [7]]
	rawY = [[1], [2], [3], [4], [5], [6], [7]]

	toTestListX = createAll(rawX, ["value"])
	toTestListY = createAll(rawY, ["time"])

	minTrain = datetime.timedelta(0)
	maxTrain = datetime.timedelta(2)
	step = 1
	gap = 0
	minTest = datetime.timedelta(0)
	maxTest = datetime.timedelta(2)

	# call OCV for each data representation type
	for i in range(len(toTestListX)):
		dataX = toTestListX[i]
		dataY = toTestListY[i]
		results = orderedCrossValidate(dataX, dataY, [returnNone], "time", minTrain, maxTrain, step, gap, minTest, maxTest)

	expStartTrain = [0,0,0,1,2,3]
	expEndTrain = [0,1,2,3,4,5]
	expStartTest = [1,2,3,4,5,6]
	expEndTest = [3,4,5,6,6,6]

	checkResults(results, expStartTrain, expEndTrain, expStartTest, expEndTest)


def _orderedCV_trialConstruction_nonIntTime_dtParam():
	""" Test of the orderedCrossValidate() construction of trials with noninteger time with timedelta params"""
	rawX = [[1], [2], [3], [4], [5], [6], [7]]
	rawY = [[1.], [2], [3], [4], [5], [6], [7]]

	toTestListX = createAll(rawX, ["value"])
	toTestListY = createAll(rawY, ["time"])

	minTrain = datetime.timedelta(0)
	maxTrain = datetime.timedelta(2)
	step = 1
	gap = 0
	minTest = datetime.timedelta(0)
	maxTest = datetime.timedelta(2)

	# call OCV for each data representation type
	for i in range(len(toTestListX)):
		dataX = toTestListX[i]
		dataY = toTestListY[i]
		results = orderedCrossValidate(dataX, dataY, [returnNone], "time", minTrain, maxTrain, step, gap, minTest, maxTest)

	expStartTrain = [0,0,0,1,2,3]
	expEndTrain = [0,1,2,3,4,5]
	expStartTest = [1,2,3,4,5,6]
	expEndTest = [3,4,5,6,6,6]

	checkResults(results, expStartTrain, expEndTrain, expStartTest, expEndTest)



