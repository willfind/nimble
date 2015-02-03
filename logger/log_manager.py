"""
	A class that manages logging from a high level.  Creates two low-level logger objects -
	a human readable logger and a machine-readable logger - and passes run information to each
	of them.  The logs are put in the default location (home directory) unless the log path is provided
	when instantiated.  Likewise with the name of the log file: unless provided at instantiation, it
	is set to default value.
"""

import os
import datetime

import UML
from human_readable_log import HumanReadableLogger
from machine_readable_log import MachineReadableLogger


class LogManager(object):

	def __init__(self, logLocation, logName):
		fullLogDesignator = os.path.join(logLocation, logName)

		self.humanReadableLog = HumanReadableLogger(fullLogDesignator + ".txt")
		self.machineReadableLog = MachineReadableLogger(fullLogDesignator + ".mr")

	def logData(self, baseDataObject):
		"""
		Send information about a data set to the log(s).
		"""
		self.humanReadableLog.logData(baseDataObject)
		self.machineReadableLog.logData(baseDataObject)

	def logLoad(self, dataFilePath=None):
		"""
		Send information about the loading of a data set to the log(s).
		"""
		self.humanReadableLog.logLoad(dataFilePath)
		self.machineReadableLog.logLoad(dataFilePath)

	def logRun(self, trainData, trainLabels, testData, testLabels, function,
				metrics, predictions, performance, timer, extraInfo=None,
				numFolds=None):
		"""
			Pass the information about this run to both logs:  human and machine
			readable, which will write it out to the log files.
		"""
		self.humanReadableLog.logRun(trainData, trainLabels, testData, testLabels, function, metrics, predictions, performance, timer, extraInfo, numFolds)
		self.machineReadableLog.logRun(trainData, trainLabels, testData, testLabels, function, metrics, predictions, performance, timer, extraInfo, numFolds)

def initLoggerAndLogConfig():
	try:
		location = UML.settings.get("logger", "location")
	except:
		location = './logs-UML'
		UML.settings.set("logger", "location", location)
		UML.settings.saveChanges("logger", "location")
	finally:
	#	def cleanThenReInit(newLocation):

	#	UML.settings.hook("logger", "location")
		pass
	try:
		name = UML.settings.get("logger", "name")
	except:
		name = "log-UML"
		UML.settings.set("logger", "name", name)
		UML.settings.saveChanges("logger", "name")

	try:
		loggingEnabled = UML.settings.get("logger", "enabled")
	except:
		loggingEnabled = 'True'
		UML.settings.set("logger", "enabled", loggingEnabled)
		UML.settings.saveChanges("logger", "enabled")

	UML.logger.active = UML.logger.log_manager.LogManager(location, name)
