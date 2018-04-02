"""
    A class that manages logging from a high level.  Creates two low-level logger objects -
    a human readable logger and a machine-readable logger - and passes run information to each
    of them.  The logs are put in the default location (home directory) unless the log path is provided
    when instantiated.  Likewise with the name of the log file: unless provided at instantiation, it
    is set to default value.
"""

from __future__ import absolute_import
import os
import datetime

import UML
from .human_readable_log import HumanReadableLogger
from .machine_readable_log import MachineReadableLogger


class LogManager(object):
    def __init__(self, logLocation, logName):
        fullLogDesignator = os.path.join(logLocation, logName)

        # self.humanReadableLog = HumanReadableLogger(fullLogDesignator + ".txt")
        self.machineReadableLog = MachineReadableLogger(fullLogDesignator + ".mr")

    def cleanup(self):
        # self.humanReadableLog.cleanup()
        self.machineReadableLog.cleanup()

    def logData(self, baseDataObject):
        """
        Send information about a data set to the log(s).
        """
        # self.humanReadableLog.logData(baseDataObject)
        self.machineReadableLog.logData(baseDataObject)

    def logLoad(self, dataFilePath=None):
        """
        Send information about the loading of a data set to the log(s).
        """
        # self.humanReadableLog.logLoad(dataFilePath)
        self.machineReadableLog.logLoad(dataFilePath)

    def logRun(self, trainData, trainLabels, testData, testLabels, function,
               metrics, predictions, performance, timer, extraInfo=None,
               numFolds=None):
        """
            Pass the information about this run to both logs:  human and machine
            readable, which will write it out to the log files.
        """
        # self.humanReadableLog.logRun(trainData, trainLabels, testData, testLabels, function, metrics, predictions,
        #                              performance, timer, extraInfo, numFolds)
        self.machineReadableLog.logRun(trainData, trainLabels, testData, testLabels, function, metrics, predictions,
                                       performance, timer, extraInfo, numFolds)

    def logCrossValidation(self, trainData, trainLabels, learnerName, metric, performance,
                           timer, learnerArgs, folds=None):
        # self.humanReadableLog.logCrossValidation(trainData, trainLabels, learnerName, metric, performance, timer,
        #                                          learnerArgs, folds)
        self.machineReadableLog.logCrossValidation(trainData, trainLabels, learnerName, metric, performance, timer,
                                                   learnerArgs, folds)


def initLoggerAndLogConfig():
    """Sets up or reads configuration options associated with logging,
    and initializes the currently active logger object using those
    options.

    """
    try:
        location = UML.settings.get("logger", "location")
        if location == "":
            location = './logs-UML'
            UML.settings.set("logger", "location", location)
            UML.settings.saveChanges("logger", "location")
    except:
        location = './logs-UML'
        UML.settings.set("logger", "location", location)
        UML.settings.saveChanges("logger", "location")
    finally:
        def cleanThenReInit_Loc(newLocation):
            UML.logger.active.cleanup()
            currName = UML.settings.get("logger", 'name')
            UML.logger.active = UML.logger.uml_logger.UmlLogger(newLocation, currName)
        UML.settings.hook("logger", "location", cleanThenReInit_Loc)

    try:
        name = UML.settings.get("logger", "name")
    except:
        name = "log-UML"
        UML.settings.set("logger", "name", name)
        UML.settings.saveChanges("logger", "name")
    finally:
        def cleanThenReInit_Name(newName):
            UML.logger.active.cleanup()
            currLoc = UML.settings.get("logger", 'location')
            UML.logger.active = UML.logger.uml_logger.UmlLogger(currLoc, newName)

        UML.settings.hook("logger", "name", cleanThenReInit_Name)

    try:
        loggingEnabled = UML.settings.get("logger", "enabledByDefault")
    except:
        loggingEnabled = 'True'
        UML.settings.set("logger", "enabledByDefault", loggingEnabled)
        UML.settings.saveChanges("logger", "enabledByDefault")

    try:
        mirror = UML.settings.get("logger", "mirrorToStandardOut")
    except:
        mirror = 'False'
        UML.settings.set("logger", "mirrorToStandardOut", mirror)
        UML.settings.saveChanges("logger", "mirrorToStandardOut")

    try:
        deepCV = UML.settings.get("logger", 'enableCrossValidationDeepLogging')
    except:
        deepCV = 'False'
        UML.settings.set("logger", 'enableCrossValidationDeepLogging', deepCV)
        UML.settings.saveChanges("logger", 'enableCrossValidationDeepLogging')

    try:
        deepMulti = UML.settings.get("logger", 'enableMultiClassStrategyDeepLogging')
    except:
        deepMulti = 'False'
        UML.settings.set("logger", 'enableMultiClassStrategyDeepLogging', deepMulti)
        UML.settings.saveChanges("logger", 'enableMultiClassStrategyDeepLogging')
    print(location,name)
    UML.logger.active = UML.logger.uml_logger.UmlLogger(location, name)
