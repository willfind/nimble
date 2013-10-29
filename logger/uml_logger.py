import numpy

from UML.exceptions import ArgumentException

"""
	Handle logging of creating and testing classifiers.  Currently
	creates two versions of a log for each run:  one that is human-readable,
	and one that is machine-readable (csv).
	Should report, for each run:
		Size of input data
			# of features  (columns)
			# of points (rows)
			# points used for training
			# points used for testing

		Name of package (mlpy, scikit learn, etc.)
		Name of learningAlgorithm
		parameters of learningAlgorithm
		performance metric(s)
		results of performance metric
		Any additional information
"""
class UmlLogger(object):

	def __init__(self, logFileName=None):
		self.logFileName = logFileName
		self.isAvailable = False

	def setup(self, newFileName=None):
		"""
			Try to open the file that will be used for logging.  If newFileName
			is present, will reset the log file to use the new file name and Attempt
			to open it.  Otherwise, will use the file name provided when this was
			instantiated.  If successfully opens the file, set isAvailable to true.
		"""
		if newFileName is not None and isinstance(newFileName, (str, unicode)):
			self.logFileName = newFileName
		else: pass
		self.logFile = open(self.logFileName, 'a')
		self.isAvailable = True

	def logMessage(self, message, addNewLine=True):
		"""
			Generic function to write a message to this object's log file.  Does
			no formatting; just writes whatever is in 'message' to the file.  Attempt
			to open the log file if it has not yet been opened.  By default, adds a
			new line to each message sent, but if addNewLine is false, will not put
			the message on a new line.
		"""

		#if the log file hasn't been created, we try to create it now
		if self.isAvailable:
			pass
		else:
			self.setup()

		#Should we put this message on a new line?
		if addNewLine:
			message = "\n"+message
		else: pass
			
		self.logFile.write(message)
		self.logFile.flush()

	def logRun(self, trainData, testData, function, metrics, results, timer, extraInfo=None, numFolds=None):
		"""
		Send pertinent information about a cycle of train a learning algorithm and test its performance
		to the log file
		"""
		self._logRun_implementation(trainData, testData, function, metrics, results, timer, extraInfo, numFolds)


	def logData(self, baseDataObject):
		"""
		Send pertinent information about a data object that has been loaded/created to the log file
		"""
		self._logData_implementation(baseDataObject)


	def logLoad(self, dataFileName, baseDataType=None, name=None):
		"""
		Send pertinent information about the loading of some data set to the log file
		"""
		if dataFileName is None and baseDataType is None and name is None:
			raise ArgumentException("logLoad requires at least one non-None argument")
		else:
			self._logRun_implementation(dataFileName, baseDataType, name)

