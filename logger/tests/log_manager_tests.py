"""
Unit tests for functionality defined strictly by the LogManager object,
as well as tests confirming the initialization of the logger and
associated configuration options

"""

import tempfile
import shutil
import os
import copy

import UML
from UML.helpers import generateClassificationData

def safetyWrapper(toWrap):
	"""Decorator which ensures the safety of the the UML.settings and
	the configuraiton file during the unit tests"""
	def wrapped(*args):
		backupFile = tempfile.TemporaryFile()
		configurationFile = open(os.path.join(UML.UMLPath, 'configuration.ini'), 'r')
		backupFile.write(configurationFile.read())
		configurationFile.close()
		
		backupChanges = copy.copy(UML.settings.changes)
		backupAvailable = copy.copy(UML.interfaces.available)

		try:
			toWrap(*args)
		finally:
			backupFile.seek(0)
			configurationFile = open(os.path.join(UML.UMLPath, 'configuration.ini'), 'w')
			configurationFile.write(backupFile.read())
			configurationFile.close()

			UML.settings = UML.configuration.loadSettings()
			UML.settings.changes = backupChanges
			UML.interfaces.available = backupAvailable

	wrapped.func_name = toWrap.func_name
	wrapped.__doc__ = toWrap.__doc__

	return wrapped 

@safetyWrapper
def test_logger_location_init():
	tempDirPath = tempfile.mkdtemp()
	try:
		location = os.path.join(tempDirPath, 'logs-UML')
		UML.settings.set("logger", "location", location)
		
		cData = generateClassificationData(2, 10, 5)
		((trainX, trainY), (testX, testY)) = cData
		learner = 'custom.KNNClassifier'
		metric = UML.calculate.fractionIncorrect

		# this will trigger a write to the log file
		UML.trainAndTest(learner, trainX, trainY, testX, testY, metric, useLog=True)

		assert os.path.exists(location)
		currName = UML.settings.get("logger", "name")

		hrPath = os.path.join(location, currName + '.txt')
		assert os.path.exists(hrPath)
		assert os.path.getsize(hrPath) > 0

		mrPath = os.path.join(location, currName + '.mr')
		assert os.path.exists(mrPath)
		assert os.path.getsize(mrPath) > 0

	# Have to clean up after the mkdtemp call
	finally:
		shutil.rmtree(tempDirPath)
