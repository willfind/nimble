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
		"""
			Try to open the file that will be used for logging.  If newFileName
			is present, will reset the log file to use the new file name and Attempt
			to open it.  Otherwise, will use the file name provided when this was
			instantiated.  If successfully opens the file, set isAvailable to true.
		"""
		if newFileName is not None and isinstance(newFileName, (str, unicode)):
			self.logFileName = newFileName
		else: pass
		self.logFile = open(self.logFileName, 'w')
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

		#
		if addNewLine:
			message = "\n"+message
		else: pass
			
		self.logFile.write(message)

