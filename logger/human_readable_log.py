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


	def _logRun_implementation(self, trainData, trainLabels, testData, testLabels,
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
		#if the log file is not available, try to create it
		if not self.isAvailable:
			self.setup()

		self.logMessage('*'*80)

		#Add current time to the basic log table
		self.logMessage("Timestamp " + time.strftime('%Y-%m-%d %H:%M:%S'))

		#Write the function defining the classifier to the log. If it is
		#a string, write it directly.  Else use inspect to turn the function
		#into a string
		if isinstance(function, (str, unicode)):
			self.logMessage("Function: " + str(function))
		else:
			self.logMessage("Function: ")
			funcLines = inspect.getsourcelines(function)[0]
			for funcLine in funcLines:
				self.logMessage(str(funcLine))

		self.logMessage("")

		# Table formated info on training data, if available
		if trainData is not None:
			trainTable = _dataInfo(trainData, 'Train')
			self.logMessage(trainTable)

		# Table formated info on training data, if available
		if testData is not None:
			testTable = _dataInfo(testData, 'Test')			
			self.logMessage(testTable)

		if numFolds is not None:
			self.logMessage("# of folds: " + str(numFolds))
		
		#if extraInfo is not null, we create a new table and add all values in
		#extraInfo
		if extraInfo is not None:
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
			self.logMessage(tableString(extraTable, True, None, roundDigits=4))

		#Print out the name/function text of the error metric being used (if there
		#is only one), or the rate & name/function text if more than one is being
		#used
		if metrics is not None:
			metricTable = []
			metricHeaders = []
			metricHeaders.append("Error Metric")
			metricHeaders.append("95% CI low")
			metricHeaders.append("Error Value")
			metricHeaders.append("95% CI high")
			metricTable.append(metricHeaders)
			for metric, result in zip(metrics,performance):
				metricRow = []
				# first column: Error metric
				if inspect.isfunction(metric):
					metricString = metric.__name__
#					metricFuncLines = inspect.getsourcelines(metric)[0]
#					metricString = ""
#					for metricFuncLine in metricFuncLines:
#						metricString += metricFuncLine
				else:
					metricString = str(metric)
				metricRow.append(metricString)

#				# second column: Error value
#				metricRow.append(result)

				intervalGenName = metricString + 'ConfidenceInterval'
				interval = None
				if hasattr(UML.calculate.confidence, intervalGenName):
					if testLabels is not None and predictions is not None:
						intervalGen = getattr(UML.calculate.confidence, intervalGenName)
						interval = intervalGen(testLabels, predictions)
				if interval is None:
					metricRow.append("")
					metricRow.append(result)
					metricRow.append("")
				else:
					metricRow.append(interval[0])
					metricRow.append(result)
					metricRow.append(interval[1])

				
				metricTable.append(metricRow)
			self.logMessage(tableString(metricTable, True, None, roundDigits=4))

		if timer is not None:
			for header in timer.cumulativeTimes.keys():
				duration = timer.calcRunTime(header)
				self.logMessage(header+" time: " + "{0:.2f}".format(duration))

		self.logMessage("")

def _dataInfo(dataObj, objName):
	tableHeaders = []
	tableRow = []

	#if the data matrix was sourced from a file, add the file name and path
	if dataObj.name is not None and dataObj.nameIsNonDefault():
		tableHeaders.append(objName + " Data file")
		tableRow.append(dataObj.name)
	if dataObj.path is not None:
		tableHeaders.append(objName + " Data path")
		tableRow.append(dataObj.path)
	#add number of training points, # of of features to output list
	tableHeaders.append(objName + " points")
	tableRow.append(str(dataObj.pointCount))
	tableHeaders.append(objName + " features")
	tableRow.append(str(dataObj.featureCount))
	
	#Print table w/basic info to the log
	basicTable = [tableHeaders, tableRow]
	basicTableStr = tableString(basicTable, True, None, roundDigits=4)
	return basicTableStr


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
