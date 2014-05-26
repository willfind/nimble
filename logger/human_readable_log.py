import time
import numpy
import inspect
import types
from tableString import *

from UML.data import Base
from UML.data import Sparse
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
		elif not isinstance(baseDataObject, Base):
			raise ArgumentException("logData() requires an object of type Base to work")

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


	def _logRun_implementation(self, trainData, testData, function, metrics, results, timer, extraInfo=None, numFolds=None):
		"""
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

		tableHeaders = []
		tableRow = []

		#Add current time to the basic log table
		tableHeaders.append("Timestamp")
		tableRow.append(time.strftime('%Y-%m-%d %H:%M:%S'))

		if trainData is not None:
			#if the data matrix was sourced from a file, add the file name and path
			if trainData.name is not None:
				tableHeaders.append("Train Data file")
				tableRow.append(trainData.name)
			if trainData.path is not None:
				tableHeaders.append("Train Data path")
				tableRow.append(trainData.path)
			#add number of training points, # of of features to output list
			if trainData.data is not None:
				tableHeaders.append("Train points")
				tableRow.append(str(trainData.pointCount))
				tableHeaders.append("Train features")
				tableRow.append(str(trainData.featureCount))
			else:
				tableHeaders.append("Train points")
				tableHeaders.append("0")

		if testData is not None:
			#add name and path, if present
			if testData.name is not None and testData.name != trainData.name:
				tableHeaders.append("Test Data file")
				tableRow.append(testData.name)
			if testData.path is not None and testData.path != trainData.path:
				tableHeaders.append("Test Data path")
				tableRow.append(testData.path)
			#add number of training points, # of of features to output list
			if testData.data is not None:
				tableHeaders.append("Test points")
				tableRow.append(str(testData.pointCount))
				tableHeaders.append("Test features")
				tableRow.append(str(testData.featureCount))
			else:
				tableHeaders.append("Test points")
				tableHeaders.append("0")

		if numFolds is not None:
			tableHeaders.append("# of folds")
			tableRow.append(str(numFolds))


		if timer is not None:
			for header in timer.cumulativeTimes.keys():
				duration = timer.calcRunTime(header)
				tableHeaders.append(header+" time")
				tableRow.append("{0:.2f}".format(duration))

		#Print table w/basic info to the log
		basicTable = [tableHeaders, tableRow]
		basicTableStr = tableString(basicTable, True, None, roundDigits=4)
		self.logMessage(basicTableStr)

		#if extraInfo is not null, we create a new table and add all values in
		#extraInfo
		if extraInfo is not None:
			extraTableHeaders = []
			extraTableValues = []
			for key, value in extraInfo.iteritems():
				extraTableHeaders.append(str(key))
				if isinstance(value, types.FunctionType):
					extraTableValues.append(value.__name__)
				elif isinstance(value, Base):
					extraTableValues.append("Base: " + "(" + str(value.pointCount) + ", " + str(value.featureCount) + ")")
				else:
					extraTableValues.append(str(value))
			extraTable = [extraTableHeaders, extraTableValues]
			self.logMessage(tableString(extraTable, True, None, roundDigits=4))


		#Write the function defining the classifier to the log. If it is
		#a string, write it directly.  Else use inspect to turn the function
		#into a string
		self.logMessage("")
		self.logMessage("Function:")
		if isinstance(function, (str, unicode)):
			self.logMessage(str(function))
		else:
			funcLines = inspect.getsourcelines(function)[0]
			for funcLine in funcLines:
				self.logMessage(str(funcLine))

		#Print out the name/function text of the error metric being used (if there
		#is only one), or the rate & name/function text if more than one is being
		#used
		if metrics is not None:
			metricTable = []
			metricHeaders = []
			metricHeaders.append("\n\nError rate")
			metricHeaders.append("Error Metric")
			metricTable.append(metricHeaders)
			for metric, result in zip(metrics,results):
				metricRow = []
				metricRow.append(str(result))

				if inspect.isfunction(metric):
					metricFuncLines = inspect.getsourcelines(metric)[0]
					metricString = ""
					for metricFuncLine in metricFuncLines:
						metricString += metricFuncLine
				else:
					metricString = str(metric)
				metricRow.append(metricString)
				metricTable.append(metricRow)
			self.logMessage(tableString(metricTable, True, None, roundDigits=4))


def main():
	trainDataBase = numpy.array([(1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0)])
	testDataBase = numpy.array([(1.0, 1.0, 1.0), (0.0, 1.0, 0.0)])

	trainData1 = Sparse(trainDataBase)
	testData1 = Sparse(testDataBase)
	functionStr = """def f():
	return 0"""
	metrics = ["rootMeanSquareError", "meanAbsoluteError"]
	results = [0.50,0.45]
	extra = {"c":0.5, "folds":10, "tests": 20}

	testLogger = HumanReadableRunLog("/Users/rossnoren/UMLMisc/hrTest1.txt")
	testLogger.logRun(trainData1, testData1, functionStr, metrics, results, 0.5, extra)

	functionObj = lambda x: x+1

	testLogger.logRun(trainData1, testData1, functionObj, metrics, results, 0.5, extra)

if __name__ == "__main__":
	main()
