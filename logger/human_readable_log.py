import time
import numpy
import inspect
import types
from tableString import *

import UML
from UML.exceptions import ArgumentException
from uml_logger import UmlLogger


class HumanReadableLogger(UmlLogger):
	"""
	Class that handles various UML-specific logging tasks, printing in a more human-friendly
	format than the machine-friendly version of logging done by HumanReadableLogger.
	"""

	def __init__(self, logFileName=None):
		super(HumanReadableLogger, self).__init__(logFileName)


	def _logData_implementation(self, baseDataObject):
		"""
		Log information about a data object
		"""
		if baseDataObject is None:
			raise ArgumentException("logData() cannot do anything with a null data object")
		elif not isinstance(baseDataObject, UML.data.Base):
			raise ArgumentException("logData() requires an object of type UML.data.Base to work")

		self.logMessage('*'*37+" DATA "+'*'*37)
		self.logMessage("FEATURE REPORT")
		self.logMessage(baseDataObject.featureReport())
		self.logMessage("AGGREGATE REPORT")
		self.logMessage(baseDataObject.summaryReport())

	def _logLoad_implementation(self, dataFileName, baseDataType=None, name=None):
		"""
		Log information about the event of loading data from disk
		into a data object
		"""
		#initialize string that will contain entire data loading log message
		self.logMessage('*'*80)
		self.logMessage("Loaded Data")
		if dataFileName is not None and dataFileName != '':
			self.logMessage("Data file: "+str(dataFileName))

		if baseDataType is not None and baseDataType != '':
			self.logMessage("Data container type: "+str(baseDataType))

		if name is not None and name != '':
			self.logMessage("Data name: "+str(name))


	def _logRun_implementation(self, trainX, trainY, testX, testY,
								function, metrics, predictions, performance, timer,
								extraInfo=None, numFolds=None):
		"""timer
			Convert a set of objects representing one run (data used for training, data used for
			testing, function representing a unique classifier {learnerName, parameters}, error metrics,
			and any additional info) into a list.  This list can be appended to a second list, to create
			a 2-dimensional table that can be passed to tableString().  The results of tableString can
			then be printed to the log.

			Information contained, by default, in the resulting list:
				pass)
				# of training data points
				# of testing data points
				# of features in training data
				# of features in testing data
				Function defining the classifer (learnerName, parameters, etc.)
				Error metrics computed based on predictions of classifier: name/function and
				numerical result)

				Any additional information contained in extraInfo dict (optional).
		"""

		tableList = []

		#Pack the function defining the learner
		tableList.append([["Learner", str(function)]])

		# TODO: need kind of call? train vs trainAndApply?
#		if numFolds is not None:
#			self.logMessage("# of folds: " + str(numFolds))

		tableList.append(_packDataInfo([trainX, trainY, testX, testY]))

		#if extraInfo is not null, we create a new table and add all values in
		#extraInfo
		if extraInfo is not None and extraInfo != {}:
			tableList.append(_packExtraInfo(extraInfo))

		#Print out the name/function text of the error metric being used (if there
		#is only one), or the rate & name/function text if more than one is being
		#used
		if metrics is not None:
			metricTable = _packMetricInfo(testY, metrics, predictions, performance)
			tableList.append(metricTable)

		if timer is not None:
			timerHeaders = []
			timerList = []
			for header in timer.cumulativeTimes.keys():
				duration = timer.calcRunTime(header)
				timerHeaders.append(header + " time")
				timerList.append("{0:.2f}".format(duration))
			tableList.append([timerHeaders, timerList])

		self._log_EntryOfTables(tableList)

	def _log_EntryOfTables(self, toLog):
		"""Takes a list of dicts specifying a sequence of tables to be 
		generated as strings and then written to the log file
		
		"""
		toOutput = '*'*80 + '\n'

		timestamp = [["Timestamp", time.strftime('%Y-%m-%d %H:%M:%S')]]
		toOutput += tableString(timestamp) + '\n'

		for table in toLog:
			rowHeaders = False
			if table[0][0] == 'Data':
				rowHeaders = True
			toOutput += tableString(table, rowHeaders, roundDigits=4) + '\n'

		self.logMessage(toOutput)


def _packMetricInfo(testY, metrics, predictions, performance):
	metricTable = []
	metricHeaders = []
	metricHeaders.append("Error Metric")
	metricHeaders.append("95% CI low")
	metricHeaders.append("Error Value")
	metricHeaders.append("95% CI high")
	metricTable.append(metricHeaders)
	for metric, result in zip(metrics,performance):
		metricRow = []
		metricString = str(metric.__name__)
		metricRow.append(metricString)

		intervalGenName = metricString + 'ConfidenceInterval'
		interval = None
		if hasattr(UML.calculate.confidence, intervalGenName):
			if testY is not None and predictions is not None:
				intervalGen = getattr(UML.calculate.confidence, intervalGenName)
				interval = intervalGen(testY, predictions)
		if interval is None:
			metricRow.append("")
			metricRow.append(result)
			metricRow.append("")
		else:
			metricRow.append(interval[0])
			metricRow.append(result)
			metricRow.append(interval[1])

		metricTable.append(metricRow)
	return metricTable


def _packExtraInfo(extraInfo):
	extraTableHeaders = []
	extraTableValues = []
	for key, value in extraInfo.iteritems():
		extraTableHeaders.append(str(key))
		if isinstance(value, types.FunctionType):
			extraTableValues.append(value.__name__)
		elif isinstance(value, UML.data.Base):
			extraTableValues.append("UML.data.Base: " + "(" + str(value.pointCount) + ", " + str(value.featureCount) + ")")
		else:
			extraTableValues.append(str(value))
	extraTable = [extraTableHeaders, extraTableValues]
	return extraTable

def _packDataInfo(dataObjects):
	assert len(dataObjects) == 4

	# TODO currently ignore labels Obj from data, should record it somehow instead
	for i in range(4):
		if isinstance(dataObjects[i], (int, basestring)):
			dataObjects[i] = None

	tableHeaders = ["Data"]

	# check to see if there are meaningful values of name and path for any
	# of the objects
	includeName = False
	includePath = False
	for d in dataObjects:
		if d is not None:
			if d.name is not None and d.nameIsNonDefault():
				includeName = True
			if d.path is not None:
				includePath = True

	# set up headers
	if includeName:
		tableHeaders.append("Name")
	if includePath:
		tableHeaders.append("Path")
	tableHeaders.append("# points")
	tableHeaders.append("# features")

	# pack row for each non None object
	rowHeaders = ['trainX', 'trainY', 'testX', 'testY']
	rawTable = [tableHeaders]
	for i in range(4):
		currRow = [rowHeaders[i]]
		d = dataObjects[i]
		if d is not None:
			if includeName:
				if d.name is not None and d.nameIsNonDefault():
					currRow.append(d.name)
				else:
					currRow.append("")
			if includePath:
				if d.path is not None:
					currRow.append(d.path)
				else:
					currRow.append("")
			currRow.append(d.pointCount)
			currRow.append(d.featureCount)

			rawTable.append(currRow)

	return rawTable


def testBasic():
	trainDataBase = numpy.array([(1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0)])
	testDataBase = numpy.array([(1.0, 1.0, 1.0), (0.0, 1.0, 0.0)])

	trainData1 = UML.createData('Sparse', trainDataBase)
	testData1 = UML.createData('Sparse', testDataBase)
	metrics = ["rootMeanSquareError", "meanAbsoluteError"]
	results = [0.50,0.45]
	extra = {"c":0.5, "folds":10, "tests": 20}

	testLogger = HumanReadableRunLog("/Users/rossnoren/UMLMisc/hrTest1.txt")

	functionObj = lambda x: x+1

	testLogger.logRun(trainData1, None, testData1, None, functionObj, metrics, None, results, 0.5, extra)

if __name__ == "__main__":
	main()
