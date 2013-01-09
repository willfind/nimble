import datetime
from logger import *
from ..utility.custom_exceptions import ArgumentException

class HumanReadableTestLog(Logger):

	def logTestRun(self, trainingData, testData, function, metrics):
		"""
			Write one (data + classifer + error metrics) combination to a log file
			in a human readable format.  Should include as much information as possible,
			to allow someone to reproduce the test.  Information included:
			# of training data points
			# of testing data points
			# of features in training data
			# of features in testing data
			Function defining the classifer (algorithm, parameters, etc.)
			Error metrics computed based on predictions of classifier: name/function and numerical
			result)
		"""
		#if the log file is not available, try to create it
		if not self.isAvailable:
			self.setUp()
		else: pass

		self.writeMessage('*'*80)
		self.writeMessage('\n')
		self.writeMessage(str(datetime.datetime.now()))
		
		self.writeMessage('\n')
		self.writeMessage('Number of training points: ')
		self.writeMessage(trainingData.data.shape()[0])
		self.writeMessage('\n')
		self.writeMessage('Number of testing points: ')
		self.writeMessage(testingData.data.shape()[0])
		self.writeMessage('\n')
		self.writeMessage('Number of training features: ')
		self.writeMessage(trainingData.data.shape()[1])
		self.writeMessage('\n')
		self.writeMessage('Number of testing features: ')
		self.writeMessage(testData.data.shape()[1])
		self.writeMessage('\n')
		
		#Write the function defining the classifier to the log. If it is
		#a string, write it directly.  Else use pickle to turn the function
		#into a string
		self.writeMessage('Classifier Training Function: ')
		self.writeMessage('\n')
		if isinstance(function, (str, unicode)):
			self.writeMessage(function)
		else:
			pickle(function, self.logFile)
		self.writeMessage('\n')

		#Write performance metrics
		for metric, result in metrics:
			self.writeMessage("METRIC FUNCTION: "+"\n")
			if isinstance(metric, (str, unicode)):
				self.writeMessage(metric)
			else:
				pickle(metric, self.logFile)

			self.writeMessage("\n")
			self.writeMessage("RESULT: {0:.4f}".format(result))

		self.writeMessage("\n\n")
		self.writeMessage('*'*80)








