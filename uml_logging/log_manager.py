"""
	A class that manages logging from a high level.  Creates two low-level logger objects -
	a human readable logger and a machine-readable logger - and passes run information to each
	of them.  The logs are put in the default location (home directory) unless the log path is provided
	when instantiated.  Likewise with the name of the log file: unless provided at instantiation, it
	is set to default value.
"""

import os
import datetime

from human_readable_test import HumanReadableRunLog
from machine_readable_test import MachineReadableRunLog


class LogManager(object):

	def __init__(self, logLocation=None, logName=None):
		if logLocation is None:
			logLocation = '../'

		if logName is None:
			currDate = datetime.datetime.now()
			logName = "uMLLog-" + currDate.strftime("%Y%m%d")

		fullLogDesignator = os.path.join(logLocation, logName)

		self.humanReadableLog = HumanReadableRunLog(fullLogDesignator + ".txt")
		self.machineReadableLog = MachineReadableRunLog(fullLogDesignator + ".mr")

	def logData(self, baseDataObject):
		"""
		Send information about a data set to the log(s).
		"""
		self.humanReadableLog.logData()
		self.machineReadableLog.logData()

	def logLoad(self, dataFilePath=None):
		"""
		Send information about the loading of a data set to the log(s).
		"""
		self.humanReadableLog.logLoad()
		self.machineReadableLog.logLoad()

	def logRun(self, trainData, testData, function, metrics, timer, extraInfo=None, numFolds=None):
		"""
			Pass the information about this run to both logs:  human and machine
			readable, which will write it out to the log files.
		"""
		self.humanReadableLog.logRun(trainData, testData, function, metrics, timer, extraInfo, numFolds)
		self.machineReadableLog.logRun(trainData, testData, function, metrics, timer, extraInfo, numFolds)
