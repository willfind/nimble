"""
Unit tests for functionality defined strictly by the LogManager object,
as well as tests confirming the initialization of the logger and
associated configuration options

"""

from __future__ import absolute_import
import tempfile
import shutil
import os

import UML
from UML.helpers import generateClassificationData
from UML.configuration import configSafetyWrapper


@configSafetyWrapper
def test_logger_location_init():
    tempDirPath = tempfile.mkdtemp()
    backupName = UML.settings.get("logger", "name")
    backupLoc = UML.settings.get("logger", "location")
    try:
        location = os.path.join(tempDirPath, 'logs-UML')

        oldDirHR = UML.logger.active.humanReadableLog.logFileName
        oldDirMR = UML.logger.active.machineReadableLog.logFileName

        UML.settings.set("logger", "location", location)

        assert UML.logger.active.humanReadableLog.logFileName != oldDirHR
        assert UML.logger.active.machineReadableLog.logFileName != oldDirMR

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
