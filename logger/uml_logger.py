from __future__ import absolute_import
from __future__ import print_function
import os
import time
import six
import inspect
import numpy
import re
import sqlite3
from datetime import datetime
from dateutil.parser import parse
from ast import literal_eval
from textwrap import wrap
from functools import wraps

import UML
from UML.exceptions import ArgumentException
from .stopwatch import Stopwatch

"""
    Handle logging of creating and testing learners.
    Currently stores data in a SQLite database file and generates
    a human readable log by querying the table within the database.
    There is a hierarchical structure to the log, this limits the to
    entries with only the specified level of detail.

    Hierarchy
    Level 1: Data creation and preprocessing logs
    Level 2: Outputs basic information about the run, including timestamp,
             run number, learner name, train and test object details, parameter,
             metric and timer data if available
    Level 3: Cross validation
"""

def logCapture(function):
    """UML function wrapper to determine whether the provided function will be added to log
        """
    def wrapper(*args, **kwargs):
        funcName = function.__name__
        args = args
        kwargs = kwargs
        a, v, k, d = inspect.getargspec(function)
        argNames = a
        defaults = d
        logger = UML.logger.active
        logger.counter += 1
        timer = Stopwatch()
        timer.start("timer")
        try:
            ret = function(*args, **kwargs)
            logger.counter -= 1
        except Exception as e:
            logger.counter = 0
            raise e
        finally:
            timer.stop("timer")
        if logger.counter == 0:
            if funcName == "crossValidateBackend":
                useLog, deepLog = getUseLog(argNames, args, kwargs)
                if useLog and deepLog:
                    logger.insertIntoLog(logger.logType, logger.logInfo)
            elif "useLog" in argNames and hasattr(UML.data.base.Base, funcName):
                # special cases logging is handled in base.py
                specialCases = ["dropFeaturesContainingType", "_normalizeGeneric",
                                "featureReport", "summaryReport"]
                if funcName not in specialCases:
                    argDict = buildArgDict(argNames, defaults, args, kwargs)
                    logger.logPrep(funcName, args[0].getTypeString(), argDict)
                logger.insertIntoLog(logger.logType, logger.logInfo)
            elif "useLog" in argNames:
                useLog, _ = getUseLog(argNames, args, kwargs)
                if useLog:
                    logger.logInfo["timer"] = sum(timer.cumulativeTimes.values())
                    logger.insertIntoLog(logger.logType, logger.logInfo)
        elif funcName == "crossValidateBackend":
            useLog, deepLog = getUseLog(argNames, args, kwargs)
            if useLog and deepLog:
                logger.insertIntoLog(logger.logType, logger.logInfo)
        return ret
    return wrapper


def getUseLog(argNames, args, kwargs):
    try:
        useLogIndex = argNames.index("useLog")
        useLog = args[useLogIndex]
    except IndexError:
        useLog = kwargs.get("useLog", None)
    if useLog is None:
        useLog = UML.settings.get("logger", "enabledByDefault")
        useLog = True if useLog.lower() == 'true' else False
    deepLog = UML.settings.get("logger", "enableCrossValidationDeepLogging")
    deepLog = True if deepLog.lower() == 'true' else False
    return useLog, deepLog


# def logged(toWrap):
#     """
#     Suspends the logger when the user calls a UML function, preventing any internal calls to
#     UML functions from being logged.unsuspends the logger if an error is raised. This allows
#     the logger to continue logging in the case that the user is capturing/ignoring Exceptions
#     """
#     @wraps(toWrap)
#     def wrapper(*args, **kwargs):
#         logger = UML.logger.active
#         if logger.counter == 0 or toWrap.__name__ == "crossValidateBackend":
#             a, v, k, d = inspect.getargspec(toWrap)
#             print(toWrap.__name__)
#             print(a)
#             try:
#                 useLogIndex = a.index("useLog")
#                 useLog = args[useLogIndex]
#             except IndexError:
#                 useLog = kwargs.get("useLog", None)
#             if useLog is None:
#                 useLog = UML.settings.get("logger", "enabledByDefault")
#                 useLog = True if useLog.lower() == 'true' else False
#             deepLog = UML.settings.get("logger", "enableCrossValidationDeepLogging")
#             deepLog = True if deepLog.lower() == 'true' else False
#         else:
#             useLog = False
#         logger.counter += 1
#         try:
#             if useLog:
#                 timer = Stopwatch()
#                 timer.start("timer")
#             ret = toWrap(*args,**kwargs)
#             logger.counter -= 1
#             if useLog:
#                 timer.stop("timer")
#             if toWrap.__name__ == "crossValidateBackend":
#                 if useLog and deepLog:
#                     logger.logInfo["timer"] = sum(timer.cumulativeTimes.values())
#                     logger.insertIntoLog(logger.logType, logger.logInfo)
#             elif useLog and logger.counter == 0:
#                 logger.logInfo["timer"] = sum(timer.cumulativeTimes.values())
#                 logger.insertIntoLog(logger.logType, logger.logInfo)
#         except Exception as e:
#             logger.counter = 0
#             raise e
#         return ret
#     return wrapper


class UmlLogger(object):
    def __init__(self, logLocation, logName):
        fullLogDesignator = os.path.join(logLocation, logName)
        self.logLocation = logLocation
        self.logName = logName
        self.logFileName = fullLogDesignator + ".mr"
        self.runNumber = None
        self.isAvailable = False
        self.counter = 0
        self.logType = None
        self.logInfo = {}


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
        def regexp(y, x, search=re.search):
            return 1 if search(y, x) else 0
        self.connection.create_function('regexp', 2, regexp)
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
        """TODO"""
        # only need to call if we have previously called setup
        if self.isAvailable:
            self.connection.close()
            self.isAvailable = False


    def insertIntoLog(self, logType, logInfo):
        """
        Inserts timestamp, runNumber, logType in their respective columns of the
        sqlite table. A string of the python dictionary containing any unstructured
        information for the log entry is stored in the final column, logInfo.
        """
        if not self.isAvailable:
            self.setup(self.logFileName)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        runNumber = self.runNumber
        logInfo = str(logInfo)
        statement = "INSERT INTO logger (timestamp,runNumber,logType,logInfo) VALUES (?,?,?,?);"
        self.cursor.execute(statement, (timestamp, runNumber, logType, logInfo))
        self.connection.commit()


    def extractFromLog(self, query, values=None):
        """TODO"""
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

        logType = "load"
        logInfo = {}
        logInfo["returnType"] = returnType
        logInfo["numPoints"] = numPoints
        logInfo["numFeatures"] = numFeatures
        logInfo["name"] = name
        logInfo["path"] = path

        self.logType = logType
        self.logInfo = logInfo

    def logData(self, reportType, reportInfo):
        """
        Send pertinent information about a data object that has been loaded/created to the log file
        """
        logType = "data"
        logInfo = {}
        logInfo["reportType"] = reportType
        logInfo["reportInfo"] = reportInfo

        self.logType = logType
        self.logInfo = logInfo

    def logPrep(self, umlFunction, dataObject, arguments):
        """
        Log information about a data preparation step performed
        """
        logType = "prep"
        logInfo = {}
        logInfo["function"] = umlFunction
        logInfo["object"] = dataObject
        logInfo["arguments"] = arguments

        self.logType = logType
        self.logInfo = logInfo

    def logRun(self, umlFunction, trainData, trainLabels, testData, testLabels,
               learnerFunction, arguments, metrics, extraInfo=None, numFolds=None):
        """
        Log information about each run
        """

        logType = "run"
        logInfo = {}
        logInfo["function"] = umlFunction
        if isinstance(learnerFunction, (str, six.text_type)):
            functionCall = learnerFunction
        else:
            #TODO test this
            #we get the source code of the function as a list of strings and glue them together
            funcLines = inspect.getsourcelines(learnerFunction)
            funcString = ""
            for i in range(len(funcLines) - 1):
                funcString += str(funcLines[i])
            if funcLines is None:
                funcLines = "N/A"
            functionCall = funcString
        logInfo["learner"] = functionCall
        # check for integers or strings passed for Y values, convert if necessary
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

        if extraInfo is not None and extraInfo is not {}:
            logInfo["extraInfo"] = extraInfo

        self.logType = logType
        self.logInfo = logInfo

    def logCrossValidation(self, trainData, trainLabels, learnerName, metric, performance,
                           learnerArgs, folds=None):
        """
        Log the results of cross validation
        """
        logType = "crossVal"
        logInfo = {}
        logInfo["learner"] = learnerName
        logInfo["learnerArgs"] = learnerArgs
        logInfo["folds"] = folds
        logInfo["metric"] = metric.__name__
        logInfo["performance"] = performance

        self.logType = logType
        self.logInfo = logInfo


    ###################
    ### LOG OUTPUT ###
    ##################

    def _showLogImplementation(self, levelOfDetail, leastRunsAgo, mostRunsAgo, startDate, endDate,
                maximumEntries, searchForText, saveToFileName, append):
        """
        implementation of UML.showLog
        """

        if not self.isAvailable:
            self.setup()

        query, values = _showLogQueryAndValues(leastRunsAgo, mostRunsAgo, startDate,
                                               endDate, maximumEntries, searchForText)
        runLogs = self.extractFromLog(query, values)

        logOutput = _showLogOutputString(runLogs, levelOfDetail)

        if saveToFileName is not None:
            filePath = os.path.join(self.logLocation, saveToFileName)
            if append:
                with open(filePath, mode='a') as f:
                    f.write("\n")
                    f.write(logOutput)
            else:
                with open(filePath, mode='w') as f:
                    f.write(logOutput)
        else:
            print(logOutput)

###################
### LOG HELPERS ###
###################

def extractFunctionString(function):
    """Extracts function name or lambda function if passed a function,
       Otherwise returns a string"""
    try:
        functionName = function.__name__
        if functionName != "<lambda>":
            return functionName
        else:
            return lambdaFunctionString(function)
    except AttributeError:
        return str(function)

def lambdaFunctionString(function):
    """Returns a string of a lambda function"""
    sourceLine = inspect.getsourcelines(function)[0][0]
    line = re.findall(r'lambda.*',sourceLine)[0]
    lambdaString = ""
    afterColon = False
    openParenthesis = 1
    for letter in line:
        if letter == "(":
            openParenthesis += 1
        elif letter == ")":
            openParenthesis -= 1
        elif letter == ":":
            afterColon = True
        elif letter == "," and afterColon:
            return lambdaString
        if openParenthesis == 0:
            return lambdaString
        else:
            lambdaString += letter
    return lambdaString

def buildArgDict(argNames, defaults, args, kwargs):
    argNames.remove("self")
    args = args[1:]
    nameArgMap = {}
    for name, arg in zip(argNames,args):
        if str(arg).startswith("<") and str(arg).endswith(">"):
            nameArgMap[name] = extractFunctionString(arg)
        else:
            nameArgMap[name] = str(arg)
    startDefaults = len(argNames) - len(defaults)
    defaultArgs = argNames[startDefaults:]
    defaultDict = {}
    for name, value in zip(defaultArgs, defaults):
        if name != "useLog":
            defaultDict[name] = value

    argDict = {}
    for name in nameArgMap:
        if name not in defaultDict:
            argDict[name] = nameArgMap[name]
        elif name in defaultDict and defaultDict[name] != nameArgMap[name]:
            argDict[name] = nameArgMap[name]
    for name in kwargs:
        if name in defaultDict and defaultDict[name] != kwargs[name]:
            argDict[name] = kwargs[name]
    return argDict

def _showLogQueryAndValues(leastRunsAgo, mostRunsAgo, startDate,
                           endDate, maximumEntries, searchForText):
    """TODO"""
    selectQuery = "SELECT timestamp, runNumber, logType, logInfo FROM (SELECT * FROM logger"
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
        # convert to regex and get pattern
        searchForText = re.compile(searchForText).pattern
        whereQueryList.append("(logType REGEXP ? or logInfo REGEXP ?)")
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
    fullQuery += ") ORDER BY entry ASC;"
    includedValues = tuple(includedValues)

    return fullQuery, includedValues

def _showLogOutputString(listOfLogs, levelOfDetail):
    """TODO"""
    fullLog = "{0:^80}\n".format("UML LOGS")
    fullLog += "." * 80
    previousLogRunNumber = None
    for log in listOfLogs:
        timestamp = log[0]
        runNumber = log[1]
        logType = log[2]
        logString = log[3]
        try:
            logInfo = literal_eval(logString)
        except (ValueError, SyntaxError):
            # logString is a string (cannot eval)
            logInfo = logString
        if runNumber != previousLogRunNumber:
            fullLog += "\n"
            logString = "RUN {0}".format(runNumber)
            fullLog += ".{0:^78}.".format(logString)
            fullLog += "\n"
            fullLog += "." * 80
            previousLogRunNumber = runNumber
        try:
            if logType not in ["load", "data", "prep", "run", "crossVal"]:
                fullLog += _buildDefaultLogString(timestamp, logType, logInfo)
                fullLog += '.' * 80
            elif logType == 'load':
                fullLog += _buildLoadLogString(timestamp, logInfo)
                fullLog += '.' * 80
            elif logType == 'data':
                fullLog +=  _buildDataLogString(timestamp, logInfo)
                fullLog += '.' * 80
            elif logType == 'prep' and levelOfDetail > 1:
                fullLog +=  _buildPrepLogString(timestamp, logInfo)
                fullLog += '.' * 80
            elif logType == 'run' and levelOfDetail > 1:
                fullLog += _buildRunLogString(timestamp, logInfo)
                fullLog += '.' * 80
            elif logType == 'crossVal' and levelOfDetail > 2:
                fullLog += _buildCVLogString(timestamp, logInfo)
                fullLog += '.' * 80
        except (TypeError, KeyError): #TODO test
            # handles any user logs with a UML logType that cannot be processed by UML logger
            fullLog += _buildDefaultLogString(timestamp, logType, logInfo)
            fullLog += '.' * 80
    return fullLog

def _buildLoadLogString(timestamp, log):
    """TODO"""
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
    """TODO"""
    function = "{0}.{1}".format(log["object"], log["function"])
    fullLog = _logHeader(function, timestamp)
    if log['arguments'] != {}:
        argString = "Arguments: "
        argString += _dictToKeywordString(log["arguments"])
        for string in wrap(argString, 80, subsequent_indent=" "*19):
            fullLog += string
            fullLog += "\n"
    return fullLog

def _buildDataLogString(timestamp, log):
    """TODO"""
    reportName = log["reportType"].capitalize() + " Report"
    fullLog = _logHeader(reportName, timestamp)
    fullLog += "\n"
    fullLog += log["reportInfo"]
    return fullLog

def _buildRunLogString(timestamp, log):
    """TODO"""
    # header data
    timer = log.get("timer", "")
    if timer:
        timer = "Completed in {0:.3f} seconds".format(log['timer'])
    fullLog = _logHeader(timer, timestamp)
    fullLog += '\n{0}("{1}")\n'.format(log['function'], log["learner"])
    # train and test data
    fullLog += _formatRunLine("Data", "# points", "# features")
    if log.get("trainData", False):
        if log["trainData"].startswith("OBJECT_#"):
            fullLog += _formatRunLine("trainX", log["trainDataPoints"], log["trainDataFeatures"])
        else:
            fullLog += _formatRunLine(log["trainData"], log["trainDataPoints"], log["trainDataFeatures"])
    if log.get("trainLabels", False):
        if log["trainLabels"].startswith("OBJECT_#"):
            fullLog += _formatRunLine("trainY", log["trainLabelsPoints"], log["trainLabelsFeatures"])
        else:
            fullLog += _formatRunLine(log["trainLabels"], log["trainLabelsPoints"], log["trainLabelsFeatures"])
    if log.get("testData", False):
        if log["testData"].startswith("OBJECT_#"):
            fullLog += _formatRunLine("testX", log["testDataPoints"], log["testDataFeatures"])
        else:
            fullLog += _formatRunLine(log["testData"], log["testDataPoints"], log["testDataFeatures"])
    if log.get("testLabels", False):
        if log["testLabels"].startswith("OBJECT_#"):
            fullLog += _formatRunLine("testY", log["testLabelsPoints"], log["testLabelsFeatures"])
        else:
            fullLog += _formatRunLine(log["testLabels"], log["testLabelsPoints"], log["testLabelsFeatures"])
    fullLog += "\n"
    # parameter data
    if log.get("arguments", False):
        argString = "Arguments: "
        argString += _dictToKeywordString(log["arguments"])
        for string in wrap(argString, 80, subsequent_indent=" "*19):
            fullLog += string
            fullLog += "\n"
    # metric data
    if log.get("metrics", False):
        fullLog += "Metrics: "
        fullLog += _dictToKeywordString(log["metrics"])
        fullLog += "\n"
    # extraInfo
    if log.get("extraInfo", False):
        fullLog += "Extra Info: "
        fullLog += _dictToKeywordString(log["extraInfo"])
        fullLog += "\n"

    return fullLog

def _buildCVLogString(timestamp, log):
    """TODO"""
    crossVal = "Cross Validating for {0}".format(log["learner"])
    fullLog = _logHeader(crossVal, timestamp)
    fullLog += "\n"
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

def _buildDefaultLogString(timestamp, logType, log):
    """TODO"""
    fullLog = _logHeader(logType, timestamp)
    if isinstance(log, six.string_types):
        for string in wrap(log, 80):
            fullLog += string
            fullLog += "\n"
    elif isinstance(log, list):
        listString = _formatRunLine(log)
        for string in wrap(listString, 80):
            fullLog += string
            fullLog += "\n"
    else:
        dictString = _dictToKeywordString(log)
        for string in wrap(dictString, 80):
            fullLog += string
            fullLog += "\n"
    return fullLog

def _dictToKeywordString(dictionary):
    """TODO"""
    kvStrings = []
    for key, value in dictionary.items():
        string = "{0}={1}".format(key,value)
        kvStrings.append(string)
    return ", ".join(kvStrings)

def _formatRunLine(*args):
    """ Formats equally spaced values for each column"""
    args = list(map(str, args))
    lineLog = ""
    for arg in args:
        whitespace = 20 - len(arg)
        lineLog += arg + " " * whitespace
    lineLog += "\n"
    return lineLog

def _logHeader(left, right):
    """ Formats the top line of each log entry"""
    lineLog = "\n"
    lineLog += "{0:60}{1:>20}\n".format(left, right)
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
        deepCV = UML.settings.get("logger", 'enableCrossValidationDeepLogging')
    except:
        deepCV = 'False'
        UML.settings.set("logger", 'enableCrossValidationDeepLogging', deepCV)
        UML.settings.saveChanges("logger", 'enableCrossValidationDeepLogging')

    UML.logger.active = UmlLogger(location, name)
