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
# TODO
"""
    Handle logging of creating and testing learners.
    Currently stores data in a SQLite database file and generates
    a human readable log by querying the table within the database.
    There is a hierarchical structure to the log, allowing the user
    to specify the level of detail in the log:

    Hierarchy
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
        self.cursor = self.connection.cursor()
        statement = """
        CREATE TABLE IF NOT EXISTS logger (
        entry INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        runNumber INTEGER,
        logType TEXT,
        logInfo TEXT);
        """
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


    def insertIntoLog(self, logType, logInfo):
        """ Inserts timestamp, runNumber, logType in their respective columns of the
            sqlite table. A string of the python dictionary containing any unstructured
            information for the log entry is stored in the final column, logInfo.
        """
        if not self.isAvailable:
            self.setup(self.logFileName)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        runNumber = self.runNumber
        logInfo = str(logInfo)
        statement = "INSERT INTO logger (timestamp,runNumber,logType,logInfo) VALUES (?,?,?,?)"
        self.cursor.execute(statement, (timestamp, runNumber, logType, logInfo))
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
        logInfo = {}
        logInfo["returnType"] = returnType
        logInfo["numPoints"] = numPoints
        logInfo["numFeatures"] = numFeatures
        logInfo["name"] = name
        logInfo["path"] = path

        self.insertIntoLog(logType, logInfo)

    def logData(self, summary):
        """
        Send pertinent information about a data object that has been loaded/created to the log file
        """
        logType = "data"
        logInfo = {"summary": summary}
        self.insertIntoLog(logType, logInfo)

    def logPrep(self, umlFunction, arguments):
        """
        Log information about a data preparation step performed
        """
        logType = "prep"
        logInfo = {}
        logInfo["function"] = umlFunction
        logInfo["arguments"] = arguments

        self.insertIntoLog(logType, logInfo)

    def logRun(self, umlFunction, trainData, trainLabels, testData, testLabels,
                               learnerFunction, arguments, metrics, timer,
                               extraInfo=None, numFolds=None):
        """
        Log information about each run
        """
        logType = "run"
        logInfo = {}
        logInfo["function"] = umlFunction
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
        logInfo["learner"] = functionCall
        # check for integers or strings passed for Y values. TODO Should be done somewhere else?
        if isinstance(trainLabels, (six.string_types, int, numpy.int64)):
            trainData = trainData.copy()
            trainLabels = trainData.extractFeatures(trainLabels)
        if isinstance(testLabels, (six.string_types, int, numpy.int64)):
            testData = trainData.copy()
            testLabels = trainData.extractFeatures(testLabels)
        if trainData is not None:
            logInfo["trainData"] = trainData.name
            logInfo["trainDataPoints"] = trainData.points
            logInfo["trainDataFeatures"] = trainData.features
        if trainLabels is not None:
            logInfo["trainLabels"] = trainLabels.name
            logInfo["trainLabelsPoints"] = trainLabels.points
            logInfo["trainLabelsFeatures"] = trainLabels.features
        if testData is not None:
            logInfo["testData"] = testData.name
            logInfo["testDataPoints"] = testData.points
            logInfo["testDataFeatures"] = testData.features
        if testLabels is not None:
            logInfo["testLabels"] = testLabels.name
            logInfo["testLabelsPoints"] = testLabels.points
            logInfo["testLabelsFeatures"] = testLabels.features

        if arguments is not None and arguments != {}:
            logInfo['arguments'] = arguments

        if metrics is not None and metrics is not {}:
            logInfo["metrics"] = metrics

        if timer is not None and timer.cumulativeTimes is not {}:
            logInfo["timer"] = sum(timer.cumulativeTimes.values())

        if extraInfo is not None and extraInfo is not {}:
            logInfo["extraInfo"] = extraInfo

        self.insertIntoLog(logType, logInfo)

    def logCrossValidation(self, trainData, trainLabels, learnerName, metric, performance,
                           timer, learnerArgs, folds=None):
        """
        Log the results of cross validation
        """
        logType = "cv"
        logInfo = {}
        logInfo["learner"] = learnerName
        logInfo["learnerArgs"] = learnerArgs
        logInfo["folds"] = folds
        logInfo["metric"] = metric.__name__
        logInfo["performance"] = performance

        self.insertIntoLog(logType, logInfo)

    ###################
    ### LOG OUTPUT ###
    ##################

    def showLog(self, levelOfDetail=2, leastRunsAgo=0, mostRunsAgo=2, startDate=None, endDate=None,
                saveToFileName=None, maximumEntries=100, searchForText=None):
        """
        showLog parses the active logfile based on the arguments passed and prints a
        human readable interpretation of the log file.

        ARGUMENTS:
        levelOfDetail:  The (int) value for the level of detail from 1, the least detail,
                        to 4 (most detail)
            **Level 1: Data loading and preprocessing
            *Level 2: Outputs basic information about the run (timestamp, run number,
                     learner name, train and test object details) and parameter, metric,
                     and timer data if available
            **Level 3: TODO (CrossValidation and Multiclass)
            **Level 4: TODO

        leastRunsAgo:   The (int) value for the least number of runs since the most recent
                        run to include in the log. Defaults to 0

        mostRunsAgo:    The (int) value for the least number of runs since the most recent
                        run to include in the log. Defaults to 2

        startDate:      A string or datetime object of the date to start adding runs to the log.
                        Acceptable formats:
                          "YYYY-MM-DD"
                          "YYYY-MM-DD HH:MM"
                          "YYYY-MM-DD HH:MM:SS"

        endDate:        A string of the date to stop adding runs to the log.
                        See startDate for formatting.

        saveToFileName: The name of the file where the human readable log will be saved. File will be
                        Default is None, showLog will print to standard out

        maximumEntries: Maximum number of entries to allow before stopping the log

        searchForText:  string (or regular expression TODO) to search for in the log runs
        """

        if not self.isAvailable:
            self.setup()

        query, values = _showLogQueryAndValues(leastRunsAgo, mostRunsAgo, startDate,
                                               endDate, maximumEntries, searchForText)
        runLogs = self.extractFromLog(query, values)
        if maximumEntries is not None:
            # sorted descending in sqlite to get most recent entries
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
    selectQuery = "SELECT timestamp, runNumber, logType, logInfo FROM logger"
    whereQueryList = []
    includedValues = []
    if leastRunsAgo is not None:
        # difference between the next runNumber and leastRunsAgo (final run value)
        whereQueryList.append("runNumber <= ((SELECT MAX(runNumber) FROM logger) - ?)")
        includedValues.append(leastRunsAgo)
    if mostRunsAgo is not None:
        # difference between the next runNumber and mostRunsAgo (starting run value)
        whereQueryList.append("runNumber > ((SELECT MAX(runNumber) FROM logger) - ?)")
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
        whereQueryList.append("(logType LIKE ? or logInfo LIKE ?)")
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
        logInfo = literal_eval(logString)

        if runNumber != previousLogRunNumber:
            fullLog += "\n"
            logString = "RUN {0}".format(runNumber)
            fullLog += ".{0:^78}.".format(logString)
            fullLog += "\n"
            fullLog += "." * 80
            previousLogRunNumber = runNumber

        if logType == 'load':
            fullLog += _buildLoadLogString(timestamp, logInfo)
            fullLog += '.' * 80
        elif logType == 'data':
            fullLog +=  _buildDataLogString(timestamp, logInfo)
            fullLog += '.' * 80
        elif logType == 'prep':
            if levelOfDetail > 1:
                fullLog +=  _buildPrepLogString(timestamp, logInfo)
                fullLog += '.' * 80
        elif logType == 'run':
            if levelOfDetail > 1:
                fullLog += _buildRunLogString(timestamp, logInfo)
                fullLog += '.' * 80
        elif logType == 'cv':
            if levelOfDetail > 2:
                fullLog += _buildCVLogString(timestamp, logInfo)
                fullLog += '.' * 80
        else:
            if levelOfDetail > 3:
                pass
                # TODO
                # fullLog += "\n"
                # fullLog += _buildMultiClassLogString(timestamp, logInfo)
                # fullLog += '.' * 80
    return fullLog

def _buildRunLogString(timestamp, log):
    """ """
    # header data
    timer = log.get("timer", "")
    if timer:
        timer = "Completed in {0:.3f} seconds".format(log['timer'])
    fullLog = _logHeader(timer, timestamp)
    fullLog += '\nUML.{0}("{1}")\n'.format(log['function'], log["learner"])
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
    dataCol = "{0} Loaded".format(log["returnType"])
    fullLog = _logHeader(dataCol,timestamp)
    if log['path'] is not None:
        fullLog += _formatRunLine("path", log["path"])
        dataCol = ""
    if log['name'] is not None:
        fullLog += _formatRunLine("name", log["name"])
        dataCol = ""
    fullLog += _formatRunLine("# of points", log["numPoints"])
    fullLog += _formatRunLine("# of features", log["numFeatures"])
    return fullLog

def _buildPrepLogString(timestamp, log):
    function = "UML.{0}".format(log["function"])
    fullLog = _logHeader(function, timestamp)
    if log['arguments'] != {}:
        argString = "Arguments: "
        argString += _dictToKeywordString(log["arguments"])
        for string in wrap(argString, 80, subsequent_indent=" "*19):
            fullLog += string
            fullLog += "\n"
    return fullLog

def _buildDataLogString(timestamp, log):
    fullLog = _logHeader("Summary Report", timestamp)
    fullLog += log["summary"]
    return fullLog

def _buildCVLogString(timestamp, log):
    crossVal = "Cross Validating for {0}".format(log["learner"])
    fullLog = _logHeader(crossVal, timestamp)
    fullLog += "\n"
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

def _logHeader(left, right):
    """ Formats the top line of each log entry"""
    lineLog = "\n"
    lineLog += "{0:40}{1:>40}\n".format(left, right)
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
