from __future__ import absolute_import
from __future__ import print_function
import os
import time
import six
import inspect
import numpy
import sqlite3
from datetime import datetime
from dateutil.parser import parse
from ast import literal_eval
from textwrap import wrap

import UML
from UML.exceptions import ArgumentException

"""
    Handle logging of creating and testing learners.
    Currently stores data in a SQLite database file and generates
    a human readable log by querying the tables within the database.
    There is a hierarchical structure to the log, allowing the user
    to specify the level of detail in the log:

    Current Hierarchy
    Level ?: Data creation and preprocessing logs
    Level 1: Outputs basic information about the run (timestamp, run number,
             learner name, train and test object details) and boolean values
             for the availability of additional information
    Level 2: Parameter, metric, and timer data if available
    Level 3: Cross validation
    Level 4: Epoch data
"""

class UmlLogger(object):
    def __init__(self, logLocation, logName):
        fullLogDesignator = os.path.join(logLocation, logName)
        self.logLocation = logLocation
        self.logName = logName
        self.logFileName = fullLogDesignator + ".mr"
        self.runNumber = None
        self.isAvailable = False
        self.suspended = False


    def setup(self, newFileName=None):
        """
            Try to open the file that will be used for logging.  If newFileName
            is present, will reset the log file to use the new file name and Attempt
            to open it.  Otherwise, will use the file name provided when this was
            instantiated.  If successfully opens the file, set isAvailable to true.
        """
        if newFileName is not None and isinstance(newFileName, (str, six.text_type)):
            self.logFileName = newFileName

        dirPath = os.path.dirname(self.logFileName)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        self.connection = sqlite3.connect(self.logFileName)
        statement = """
        CREATE TABLE IF NOT EXISTS logger (
        entry INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        runNumber INTEGER,
        logType TEXT,
        logDict TEXT);
        """
        self.cursor = self.connection.cursor()
        self.cursor.execute(statement)
        self.connection.commit()

        statement = "SELECT MAX(runNumber) FROM logger;"
        self.cursor.execute(statement)
        lastRun = self.cursor.fetchone()[0] #fetchone returns a tuple
        if lastRun is not None:
            self.runNumber = lastRun + 1
        else:
            self.runNumber = 0

        self.isAvailable = True


    def cleanup(self):
        # only need to call if we have previously called setup
        if self.isAvailable:
            self.connection.close()
            self.isAvailable = False


    def insertIntoLog(self, logType, logDict):
        """ Inserts the timestamp, runNumber, log logType and the log"""
        if not self.isAvailable:
            self.setup(self.logFileName)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        runNumber = self.runNumber
        logDict = str(logDict)
        statement = "INSERT INTO logger (timestamp,runNumber,logType,logDict) VALUES (?,?,?,?)"
        self.cursor.execute(statement, (timestamp, runNumber, logType, logDict))
        self.connection.commit()

    def extractFromLog(self, query, values=None):
        if not self.isAvailable:
            self.setup()
        if values is None:
            self.cursor.execute(query)
        else:
            self.cursor.execute(query, values)
        ret = self.cursor.fetchall()
        return ret

    ###################
    ### LOG ENTRIES ###
    ###################

    def logLoad(self, returnType, numPoints, numFeatures, name=None, path=None):
        """
        Log information about the loading of a data set
        """
        #TODO only log if name or path is present?
        logType = "load"
        logDict = {}
        logDict["numPoints"] = numPoints
        logDict["numFeatures"] = numFeatures
        logDict["name"] = name
        logDict["path"] = path

        self.insertIntoLog(logType, logDict)

    # def logData(self): #TODO
    #     """
    #     Send pertinent information about a data object that has been loaded/created to the log file
    #     """
    #     logType = "load"
    #     logDict = {}
    #     self.insertIntoLog(logDict)

    def logPrep(self, umlFunction, arguments):
        """
        Log information about a data preparation step performed
        """
        logType = "prep"
        logDict = {}
        logDict["function"] = umlFunction
        logDict["arguments"] = arguments

        self.insertIntoLog(logType, logDict)

    def logRun(self, umlFunction, trainData, trainLabels, testData, testLabels,
                               learnerFunction, arguments, metrics, timer,
                               extraInfo=None, numFolds=None):
        """
        Log information about each run
        """
        logType = "run"
        logDict = {}
        logDict["function"] = umlFunction
        if isinstance(learnerFunction, (str, six.text_type)):
            functionCall = learnerFunction
        else:
            #we get the source code of the function as a list of strings and glue them together
            funcLines = inspect.getsourcelines(learnerFunction)
            funcString = ""
            for i in range(len(funcLines) - 1):
                funcString += str(funcLines[i])
            if funcLines is None:
                funcLines = "N/A"
            functionCall = funcString
        logDict["learner"] = functionCall
        # check for integers or strings passed for Y values. TODO Should be done somewhere else?
        if isinstance(trainLabels, (six.string_types, int, numpy.int64)):
            trainData = trainData.copy()
            trainLabels = trainData.extractFeatures(trainLabels)
        if isinstance(testLabels, (six.string_types, int, numpy.int64)):
            testData = trainData.copy()
            testLabels = trainData.extractFeatures(testLabels)
        if trainData is not None:
            logDict["trainData"] = trainData.name
            logDict["trainDataPoints"] = trainData.points
            logDict["trainDataFeatures"] = trainData.features
        if trainLabels is not None:
            logDict["trainLabels"] = trainLabels.name
            logDict["trainLabelsPoints"] = trainLabels.points
            logDict["trainLabelsFeatures"] = trainLabels.features
        if testData is not None:
            logDict["testData"] = testData.name
            logDict["testDataPoints"] = testData.points
            logDict["testDataFeatures"] = testData.features
        if testLabels is not None:
            logDict["testLabels"] = testLabels.name
            logDict["testLabelsPoints"] = testLabels.points
            logDict["testLabelsFeatures"] = testLabels.features

        if arguments is not None and arguments != {}:
            logDict['arguments'] = arguments

        if metrics is not None and metrics is not {}:
            logDict["metrics"] = metrics

        if timer is not None and timer.cumulativeTimes is not {}:
            logDict["timer"] = sum(timer.cumulativeTimes.values())

        if extraInfo is not None and extraInfo is not {}:
            logDict["extraInfo"] = extraInfo

        self.insertIntoLog(logType, logDict)

    def logCrossValidation(self, trainData, trainLabels, learnerName, metric, performance,
                           timer, learnerArgs, folds=None):
        """
        Log the results of cross validation
        """
        logType = "cv"
        logDict = {}
        logDict["learner"] = learnerName
        logDict["learnerArgs"] = learnerArgs
        logDict["folds"] = folds
        logDict["metric"] = metric.__name__
        logDict["performance"] = performance

        self.insertIntoLog(logType, logDict)

    ###################
    ### LOG OUTPUT ###
    ##################

    def showLog(self, levelOfDetail, leastRunsAgo, mostRunsAgo, startDate,
                endDate, saveToFileName, maximumEntries, searchForText):
        """ Implementation of showLog function for UML"""
        if not self.isAvailable:
            self.setup()

        query, values = _showLogQueryAndValues(leastRunsAgo, mostRunsAgo, startDate,
                                               endDate, maximumEntries, searchForText)

        runLogs = self.extractFromLog(query, values)

        if maximumEntries is not None:
            # sorted descending by sqlite to get most recent entries
            # need to reverse to return to chronological order
            runLogs = runLogs[::-1]

        logOutput = _showLogOutputString(runLogs, levelOfDetail)

        if saveToFileName is not None:
            # TODO append if already exists?
            filePath = os.path.join(self.logLocation, saveToFileName)
            with open(filePath, mode='w') as f:
                f.write(logOutput)
        else:
            print(logOutput)


###################
### LOG HELPERS ###
###################

def useLogCheck(useLog):
    # if logger is suspended do not log and do not unsuspend
    if UML.logger.active.suspended:
        toLog = False
        unsuspend = False
        return toLog, unsuspend
    # if logger NOT suspended log based on useLog and unsuspend
    if useLog is None:
        useLog = UML.settings.get("logger", "enabledByDefault")
        useLog = True if useLog.lower() == 'true' else False
    toLog = useLog
    unsuspend = True
    UML.logger.active.suspended = True
    return useLog, unsuspend

def _showLogQueryAndValues(leastRunsAgo, mostRunsAgo, startDate,
                           endDate, maximumEntries, searchForText):
    selectQuery = "SELECT timestamp, runNumber, logType, logDict FROM logger"
    whereQueryList = []
    includedValues = []
    if leastRunsAgo is not None:
        # difference between the next runNumber and leastRunsAgo (final run value)
        whereQueryList.append("runNumber <= ((SELECT MAX(runNumber) FROM logger) - ? + 1)")
        includedValues.append(leastRunsAgo)
    if mostRunsAgo is not None:
        # difference between the next runNumber and mostRunsAgo (starting run value)
        whereQueryList.append("runNumber >= ((SELECT MAX(runNumber) FROM logger) - ? + 1)")
        includedValues.append(mostRunsAgo)
    if startDate is not None:
        whereQueryList.append("timestamp >= ?")
        includedValues.append(parse(startDate))
    if endDate is not None:
        whereQueryList.append("timestamp <= ?")
        includedValues.append(parse(endDate))
    if searchForText is not None:
        # add % to search for text anywhere within string
        searchForText = "%" + searchForText + "%"
        whereQueryList.append("(logType LIKE ? or logDict LIKE ?)")
        includedValues.append(searchForText)
        includedValues.append(searchForText)

    if whereQueryList != []:
        whereQuery = " and ".join(whereQueryList)
        fullQuery = selectQuery + " WHERE " + whereQuery
    else:
        fullQuery = selectQuery

    if maximumEntries is not None:
        fullQuery += " ORDER BY entry DESC "
        fullQuery += "LIMIT ?"
        includedValues.append(maximumEntries)
    fullQuery += ";"
    includedValues = tuple(includedValues)

    return fullQuery, includedValues

def _showLogOutputString(listOfLogs, levelOfDetail):
    fullLog = "{0:^80}\n".format("UML LOGS")
    fullLog += "." * 80
    previousLogRunNumber = None
    for log in listOfLogs:
        timestamp = log[0]
        runNumber = log[1]
        logType = log[2]
        logString = log[3]
        logDict = literal_eval(logString)

        if runNumber != previousLogRunNumber:
            fullLog += "\n"
            logString = "RUN {0}".format(runNumber)
            fullLog += ".{0:^78}.".format(logString)
            fullLog += "\n"
            fullLog += "." * 80
            previousLogRunNumber = runNumber

        if logType == 'load':
            fullLog += _buildLoadLogString(timestamp, logDict)
            fullLog += '.' * 80
        elif logType == 'data':
            pass
            # TODO
            # fullLog += "\n"
            # fullLog +=  _buildDataLogString(timestamp, logDict)
            # fullLog += '.' * 80
        elif logType == 'prep':
            if levelOfDetail > 1:
                fullLog +=  _buildPrepLogString(timestamp, logDict)
                fullLog += '.' * 80
        elif logType == 'run':
            if levelOfDetail > 1:
                fullLog += _buildRunLogString(timestamp, logDict)
                fullLog += '.' * 80
        elif logType == 'cv':
            if levelOfDetail > 2:
                fullLog += _buildCVLogString(timestamp, logDict)
                fullLog += '.' * 80
        else:
            if levelOfDetail > 3:
                pass
                # TODO
                # fullLog += "\n"
                # fullLog += _buildMultiClassLogString(timestamp, logDict)
                # fullLog += '.' * 80
    return fullLog

def _buildRunLogString(timestamp, log):
    """ """
    # header data
    fullLog = _logHeader(timestamp)
    timer = log.get("timer", False)
    if timer:
        fullLog += "Completed in {0:.3f} seconds\n".format(log['timer'])
    fullLog += "\n"
    fullLog += 'UML.{0}("{1}")\n'.format(log['function'], log["learner"])

    # train and test data
    fullLog += _formatRunLine("Data", "# points", "# features")
    if log.get("trainData", False):
        fullLog += _formatRunLine("trainX", log["trainDataPoints"], log["trainDataFeatures"])
    if log.get("trainLabels", False):
        fullLog += _formatRunLine("trainY", log["trainLabelsPoints"], log["trainLabelsFeatures"])
    if log.get("testData", False):
        fullLog += _formatRunLine("testX", log["testDataPoints"], log["testDataFeatures"])
    if log.get("testLabels", False):
        fullLog += _formatRunLine("testY", log["testLabelsPoints"], log["testLabelsFeatures"])
    # parameter data
    if log.get("arguments", False):
        fullLog += "\n"
        argString = "Arguments: "
        argString += _dictToKeywordString(log["arguments"])
        for string in wrap(argString, 80, subsequent_indent=" "*19):
            fullLog += string
            fullLog += "\n"
    # metric data
    if log.get("metrics", False):
        fullLog += "\n"
        fullLog += "Metrics: "
        fullLog += _dictToKeywordString(log["metrics"])
        fullLog += "\n"
    # extraInfo
    if log.get("extraInfo", False):
        fullLog += "\n"
        fullLog += "Extra Info: "
        fullLog += _dictToKeywordString(log["extraInfo"])
        fullLog += "\n"

    return fullLog

def _buildLoadLogString(timestamp, log):
    fullLog = _logHeader(timestamp)
    dataCol = "Data Loaded"
    if log['path'] is not None:
        fullLog += _formatRunLine(dataCol, "path", log["path"])
        dataCol = ""
    if log['name'] is not None:
        fullLog += _formatRunLine(dataCol, "name", log["name"])
        dataCol = ""
    fullLog += _formatRunLine(dataCol, "# of points", log["numPoints"])
    fullLog += _formatRunLine("", "# of features", log["numFeatures"])
    return fullLog

def _buildPrepLogString(timestamp, log):
    fullLog = _logHeader(timestamp)
    fullLog += "UML.{0}\n".format(log["function"])
    if log['arguments'] != {}:
        argString = "Arguments: "
        argString += _dictToKeywordString(log["arguments"])
        for string in wrap(argString, 80, subsequent_indent=" "*19):
            fullLog += string
            fullLog += "\n"
    return fullLog

def _buildCVLogString(timestamp, log):
    fullLog = _logHeader(timestamp)
    fullLog += "Cross Validating for {0}\n\n".format(log["learner"])
    # TODO when is learnerArgs returning an empty list?
    if isinstance(log["learnerArgs"], dict):
        fullLog += "Variable Arguments: "
        fullLog += _dictToKeywordString(log["learnerArgs"])
        fullLog += "\n\n"
    folds = log["folds"]
    metric = log["metric"]
    fullLog += "{0}-folding using {1} optimizing for min values\n\n".format(folds, metric)
    fullLog += _formatRunLine("Result", "Arguments")
    for arguments, result in log["performance"]:
        argString = _dictToKeywordString(arguments)
        fullLog += "{0:<20.3f}{1:20s}".format(result, argString)
        fullLog += "\n"
    return fullLog

def _dictToKeywordString(dictionary):
    kvStrings = []
    for key, value in dictionary.items():
        string = "{0}={1}".format(key,value)
        kvStrings.append(string)
    return ", ".join(kvStrings)

def _formatRunLine(*args):
    """ Formats equally spaced values for each column"""
    args = list(map(str, args))
    lineLog = ("{:20s}" * len(args)).format(*args) #TODO works below python2.7?
    lineLog += "\n"
    return lineLog

def _logHeader(timestamp):
    """ Formats the top line of each log entry"""
    lineLog = "\n"
    lineLog += "{0:>80}\n".format(timestamp)
    return lineLog

#######################
### Initialization  ###
#######################

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
            UML.logger.active = UmlLogger(newLocation, currName)
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
            UML.logger.active = UmlLogger(currLoc, newName)

        UML.settings.hook("logger", "name", cleanThenReInit_Name)

    try:
        loggingEnabled = UML.settings.get("logger", "enabledByDefault")
    except:
        loggingEnabled = 'True'
        UML.settings.set("logger", "enabledByDefault", loggingEnabled)
        UML.settings.saveChanges("logger", "enabledByDefault")

    try:
        runTests = UML.settings.get("logger", "_runTestsActive")
    except:
        runTests = 'False'
        UML.settings.set("logger", "_runTestsActive", runTests)
        UML.settings.saveChanges("logger", "_runTestsActive")

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
    UML.logger.active = UmlLogger(location, name)
