import numpy

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
		Name of algorithm
		parameters of algorithm
		performance metric(s)
		results of performance metric


"""
class Logger(object):

	def __init__(self, logFileName=None):
		self.logFileName = logFileName
		self.isAvailable = False

	def setup(self, newFileName=None):
		if newFileName is not None and isinstance(newFileName, (str, unicode)):
			self.logFileName = newFileName
		else: pass
		self.logFile = open(self.logFileName, 'w')
		self.isAvailable = True

	def logMessage(self, message):
		"""
			Generic function to write a message to this object's log file.  Does
			no formatting; just writes whatever is in 'message' to the file.  Attempt
			to open the log file if it has not yet been opened.
		"""
		if self.isAvailable:
			self.writeMessage("\n"+message)
		else:
			self.setup()
			self.writeMessage("\n"+message)


	def writeMessage(self, stringToWrite):
		"""
			Helper function to make code cleaner.  Writes a string to
			this logger's log file.
		"""
		self.logFile.write(stringToWrite)
