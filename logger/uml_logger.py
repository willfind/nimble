"""
Handle logging of creating and testing learners.

Currently stores data in a SQLite database file and generates
a human readable log by querying the table within the database.
There is a hierarchical structure to the log, this limits the to
entries with only the specified level of detail.

Also included are the wrappers to ensure that logged functions are
logged only when necessary.

Hierarchy
Level 1: Data creation and preprocessing logs
Level 2: Outputs basic information about the run, including timestamp,
         run number, learner name, train and test object details,
         parameter, metric and timer data if available
Level 3: Cross validation
"""
from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import time
import inspect
import re
import sqlite3
from ast import literal_eval
from textwrap import wrap
from functools import wraps
from dateutil.parser import parse

import six
from six import reraise
import numpy

import UML
from .stopwatch import Stopwatch

class UmlLogger(object):
    """
    Handle logging of UML functions and generating the log output.

    Parameters
    ----------
    logLocation : str
        The path to the directory containing the log file
    logName : str
        The name of the log file. The suffix '.mr' will be added to the
        name, to indicate this is the 'machine-readable' version of the
        logged information.
    """
    def __init__(self, logLocation, logName):
        fullLogDesignator = os.path.join(logLocation, logName)
        self.logLocation = logLocation
        self.logName = logName
        self.logFileName = fullLogDesignator + ".mr"
        self.runNumber = None
        self.connection = None
        self.cursor = None
        self.isAvailable = False
        self.position = 0
        self.logType = None
        self.logInfo = {}


    def setup(self, newFileName=None):
        """
        Open or create the log file.

        Try to open the file that will be used for logging.  If
        ``newFileName`` is present, will reset the log file to use the
        new file name and attempt to open it.  Otherwise, will use the
        file name provided when this was instantiated.  If successfully
        opens the file, set isAvailable to true.

        Parameters
        ----------
        newFileName : str
            The name new logging file.
        """
        if (newFileName is not None
                and isinstance(newFileName, (str, six.text_type))):
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
        """
        Closes the connection to the logger database, if it is open.
        """
        # only need to call if we have previously called setup
        if self.isAvailable:
            self.connection.close()
            self.isAvailable = False


    def log(self, logType, logInfo):
        """
        Add information into the log file.

        Inserts timestamp, runNumber, ``logType`` in their respective
        columns of the sqlite table. A string of ``logInfo`` will be
        stored in the final column.

        Parameters
        ----------
        logType : str
            The type of information being added to the log. The values
            'load', 'prep', 'run', 'data', and 'crossVal', generate a
            custom output of the ``logInfo`` when printing the log.  Any
            other log type will print a string of ``logInfo`` without
            any additional formatting.
        logInfo : dict, list, str
            All types for this value will be converted and added to the
            log. If provided a dictionary; ``logType`` 'load', 'prep',
            'run', 'data', and 'crossVal' generate a custom output when
            printing the log.
        """
        if not self.isAvailable:
            self.setup(self.logFileName)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        runNum = self.runNumber
        logInfo = str(logInfo)
        statement = "INSERT INTO logger (timestamp,runNumber,logType,logInfo) "
        statement += "VALUES (?,?,?,?);"
        self.cursor.execute(statement, (timestamp, runNum, logType, logInfo))
        self.connection.commit()


    def extractFromLog(self, query, values=None):
        """
        Fetch data from log file.

        Return a list of tuples for values matching a SQLite query
        statement.

        Parameters
        ----------
        query : str
            a SQLite query statement.
        values : tuple
            values to use in place of the "?" placeholders used in the
            query.
        """
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

    def logLoad(self, returnType, numPoints, numFeatures, name=None,
                path=None):
        """
        Log information about the loading of a data object.

        Parameters
        ----------
        returnType : str
            The type of the loaded object.
        numPoints : int
            The number of points in the loaded object.
        numFeatures : int
            The number of features in the loaded object.
        name : str
            The name of the loaded object.
        path : str
            The path to the data in the loaded object.
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
        Log an object's information reports.

        Add data generated from calls to featureReport and
        summaryReport to the log.

        reportType : str
            'feature' or 'summary' based on the type of report.
        reportInfo : str
            The information generated during the call to the report
            function.
        """
        logType = "data"
        logInfo = {}
        logInfo["reportType"] = reportType
        logInfo["reportInfo"] = reportInfo

        self.logType = logType
        self.logInfo = logInfo

    def logPrep(self, umlFunction, dataObject, arguments):
        """
        Log information about a data preparation step performed.

        Add information regarding the data preprocessing functions
        called on an object.

        Parameters
        ----------
        umlFunction : str
            The name of the umlFunction called.
        dataObject : str
            The class of the calling object.
        arguments : dict
            A mapping of the argument name to the argument's value.
        """
        logType = "prep"
        logInfo = {}
        logInfo["function"] = umlFunction
        logInfo["object"] = dataObject
        logInfo["arguments"] = arguments

        self.logType = logType
        self.logInfo = logInfo

    def logRun(self, umlFunction, trainData, trainLabels, testData, testLabels,
               learnerFunction, arguments, metrics, extraInfo=None,
               numFolds=None):
        """
        Log information about each run.

        Add information related to each run on training and testing
        data.

        Parameters
        ----------
        umlFunction : str
            The name of the umlFunction called.
        trainData : UML data object
            The object containing the training data.
        trainLabels : UML data object, int
            The object or feature in ``trainData`` containing the
            training labels.
        testData : UML data object
            The object containing the testing data.
        testLabels : UML data object, int
            The object or feature in ``testData`` containing the
            training labels.
        learnerFunction : str
            The name of learner function.
        arguments : dict
            The arguments passed to the learner.
        metrics : dict
            The results of the testing on a run.
        extraInfo
            Any extra information to add to the log. Typically provides
            the best parameters from cross validation.
        numFolds : int
            The number of folds if k-fold cross validation utilized.
        """

        logType = "run"
        logInfo = {}
        logInfo["function"] = umlFunction
        if isinstance(learnerFunction, (str, six.text_type)):
            functionCall = learnerFunction
        else:
            #TODO test this
            #we get the source code of the function as a list of strings and
            # glue them together
            funcLines = inspect.getsourcelines(learnerFunction)
            funcString = ""
            for i in range(len(funcLines) - 1):
                funcString += str(funcLines[i])
            if funcLines is None:
                funcLines = "N/A"
            functionCall = funcString
        logInfo["learner"] = functionCall
        # integers or strings passed for Y values, convert if necessary
        if isinstance(trainLabels, (six.string_types, int, numpy.int64)):
            trainData = trainData.copy()
            trainLabels = trainData.features.extract(trainLabels)
        if isinstance(testLabels, (six.string_types, int, numpy.int64)):
            testData = trainData.copy()
            testLabels = trainData.features.extract(testLabels)
        if trainData is not None:
            logInfo["trainData"] = trainData.name
            logInfo["trainDataPoints"] = len(trainData.points)
            logInfo["trainDataFeatures"] = len(trainData.features)
        if trainLabels is not None:
            logInfo["trainLabels"] = trainLabels.name
            logInfo["trainLabelsPoints"] = len(trainLabels.points)
            logInfo["trainLabelsFeatures"] = len(trainLabels.features)
        if testData is not None:
            logInfo["testData"] = testData.name
            logInfo["testDataPoints"] = len(testData.points)
            logInfo["testDataFeatures"] = len(testData.features)
        if testLabels is not None:
            logInfo["testLabels"] = testLabels.name
            logInfo["testLabelsPoints"] = len(testLabels.points)
            logInfo["testLabelsFeatures"] = len(testLabels.features)

        if arguments is not None and arguments != {}:
            logInfo['arguments'] = arguments

        if metrics is not None and metrics != {}:
            logInfo["metrics"] = metrics

        if extraInfo is not None and extraInfo != {}:
            logInfo["extraInfo"] = extraInfo

        self.logType = logType
        self.logInfo = logInfo

    def logCrossValidation(self, trainData, trainLabels, learnerFunction,
                           arguments, metric, performance, folds=None):
        """
        Log the results of cross validation.

        Parameters
        ----------
        trainData : UML data object
            The object containing the training data.
        trainLabels : UML data object, int
            The object or feature in ``trainData`` containing the
            training labels.
        learnerFunction : str
            The name of the learner function.
        arguments : dict
            The arguments passed to the learner.
        metric : function
            The results of the testing on a run.
        performance : list
            A list of the performance results of each permutation.
        folds : int
            The number of folds.
        """
        logType = "crossVal"
        logInfo = {}
        logInfo["learner"] = learnerFunction
        logInfo["learnerArgs"] = arguments
        logInfo["folds"] = folds
        logInfo["metric"] = metric.__name__
        logInfo["performance"] = performance

        self.logType = logType
        self.logInfo = logInfo


    ###################
    ### LOG OUTPUT ###
    ##################

    def showLog(self, levelOfDetail, leastRunsAgo, mostRunsAgo, startDate,
                endDate, maximumEntries, searchForText, regex,
                saveToFileName, append):
        """
        Output data from the logger.

        Parse the log based on the arguments passed and print a human
        readable interpretation of the log file.

        Parameters
        ----------
        levelOfDetail:  int
            The value for the level of detail from 1, the least detail,
            to 3 (most detail). Default is 2.
            * Level 1 - Data loading, data preparation and preprocessing,
              custom user logs.
            * Level 2 - Outputs basic information about each run.
              Includes timestamp, run number, learner name, train and
              test object details, parameter, metric, and timer data if
              available.
            * Level 3 - Include cross-validation data.
        leastRunsAgo : int
            The least number of runs since the most recent run to
            include in the log. Default is 0.
        mostRunsAgo : int
            The most number of runs since the most recent run to
            include in the log. Default is 2.
        startDate :  str, datetime
            A string or datetime object of the date to begin adding runs
            to the log.
            Acceptable formats:
            * "YYYY-MM-DD"
            * "YYYY-MM-DD HH:MM"
            * "YYYY-MM-DD HH:MM:SS"
        endDate : str, datetime
            A string or datetime object of the date to stop adding runs
            to the log.
            See ``startDate`` for formatting.
        maximumEntries : int
            Maximum number of entries to allow before stopping the log.
            None will allow all entries provided from the query. Default
            is 100.
        searchForText :  str, regex
            Search for in each log entry. Default is None.
        saveToFileName : str
            The name of a file where the human readable log will be
            saved. Default is None, showLog will print to standard out.
        append : bool
            Append logs to the file in saveToFileName instead of
            overwriting file. Default is False.
        """

        if not self.isAvailable:
            self.setup()

        query, values = _showLogQueryAndValues(leastRunsAgo, mostRunsAgo,
                                               startDate, endDate,
                                               maximumEntries, searchForText,
                                               regex)
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

def logCapture(function):
    """
    Wrapper to add data to log when function is called by the user.

    UML function wrapper for handling logging of top-level UML
    functions. Ensures that only calls made by user are logged, ignoring
    internal calls to logged functions. Performs the timing of each
    operation inserts the necessary information into the log.
    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        logger = UML.logger.active
        try:
            logger.position += 1
            timer = Stopwatch()
            timer.start("timer")
            ret = function(*args, **kwargs)
        except Exception:
            einfo = sys.exc_info()
            reraise(*einfo)
        finally:
            logger.position -= 1
            timer.stop("timer")
        if function.__name__ == 'crossValidateBackend':
            enableDeep = "enableCrossValidationDeepLogging"
            deepLog = UML.settings.get("logger", enableDeep)
            deepLog = True if deepLog.lower() == 'true' else False
            if deepLog:
                logger.log(logger.logType, logger.logInfo)
        elif logger.position == 0:
            logger.logInfo["timer"] = sum(timer.cumulativeTimes.values())
            logger.log(logger.logType, logger.logInfo)
        return ret
    return wrapper

def directCall(function):
    """
    Wrapper to signal a function has been called by the user.

    UML function wrapper used to ensure that internal calls to other
    logged functions are not logged when the called function can be
    logged but is not currently being logged.
    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        logger = UML.logger.active
        try:
            logger.position += 1
            ret = function(*args, **kwargs)
        except Exception:
            einfo = sys.exc_info()
            reraise(*einfo)
        finally:
            logger.position -= 1
        return ret
    return wrapper

def enableLogging(useLog):
    """
    Access useLog value from configuration, if not explictly defined.
    """
    if useLog is None:
        useLog = UML.settings.get("logger", "enabledByDefault")
        useLog = True if useLog.lower() == 'true' else False
    return useLog

def _showLogQueryAndValues(leastRunsAgo, mostRunsAgo, startDate,
                           endDate, maximumEntries, searchForText, regex):
    """
    Constructs the query string and stores the variables based on the
    arguments passed to the showLog function.
    """
    selectQuery = "SELECT timestamp, runNumber, logType, logInfo "
    selectQuery += "FROM (SELECT * FROM logger"
    whereQueryList = []
    includedValues = []
    if leastRunsAgo is not None:
        # final run value
        # difference between the next runNumber and leastRunsAgo
        where = "runNumber <= ((SELECT MAX(runNumber) FROM logger) - ?)"
        whereQueryList.append(where)
        includedValues.append(leastRunsAgo)
    if mostRunsAgo is not None:
        # starting run value
        # difference between the next runNumber and mostRunsAgo
        where = "runNumber > ((SELECT MAX(runNumber) FROM logger) - ?)"
        whereQueryList.append(where)
        includedValues.append(mostRunsAgo)
    if startDate is not None:
        whereQueryList.append("timestamp >= ?")
        includedValues.append(parse(startDate))
    if endDate is not None:
        whereQueryList.append("timestamp <= ?")
        includedValues.append(parse(endDate))
    if searchForText is not None:
        # convert to regex and get pattern
        if regex:
            searchForText = re.compile(searchForText).pattern
            whereQueryList.append("(logType REGEXP ? or logInfo REGEXP ?)")
            includedValues.append(searchForText)
            includedValues.append(searchForText)
        else:
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
    fullQuery += ") ORDER BY entry ASC;"
    includedValues = tuple(includedValues)

    return fullQuery, includedValues

def _showLogOutputString(listOfLogs, levelOfDetail):
    """
    Formats the string that will be output for calls to the showLog
    function.
    """
    fullLog = "{0:^79}\n".format("UML LOGS")
    fullLog += "." * 79
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
            fullLog += ".{0:^77}.".format(logString)
            fullLog += "\n"
            fullLog += "." * 79
            previousLogRunNumber = runNumber
        try:
            if logType not in ["load", "data", "prep", "run", "crossVal"]:
                fullLog += _buildDefaultLogString(timestamp, logType, logInfo)
                fullLog += '.' * 79
            elif logType == 'load':
                fullLog += _buildLoadLogString(timestamp, logInfo)
                fullLog += '.' * 79
            elif logType == 'data':
                fullLog += _buildDataLogString(timestamp, logInfo)
                fullLog += '.' * 79
            elif logType == 'prep' and levelOfDetail > 1:
                fullLog += _buildPrepLogString(timestamp, logInfo)
                fullLog += '.' * 79
            elif logType == 'run' and levelOfDetail > 1:
                fullLog += _buildRunLogString(timestamp, logInfo)
                fullLog += '.' * 79
            elif logType == 'crossVal' and levelOfDetail > 2:
                fullLog += _buildCVLogString(timestamp, logInfo)
                fullLog += '.' * 79
        except (TypeError, KeyError):
            fullLog += _buildDefaultLogString(timestamp, logType, logInfo)
            fullLog += '.' * 79
    return fullLog

def _buildLoadLogString(timestamp, log):
    """
    Constructs the string that will be output for load logTypes.
    """
    dataCol = "{0} Loaded".format(log["returnType"])
    fullLog = _logHeader(dataCol, timestamp)
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
    """
    Constructs the string that will be output for prep logTypes.
    """
    function = "{0}.{1}".format(log["object"], log["function"])
    fullLog = _logHeader(function, timestamp)
    if log['arguments'] != {}:
        argString = "Arguments: "
        argString += _dictToKeywordString(log["arguments"])
        for string in wrap(argString, 79, subsequent_indent=" "*11):
            fullLog += string
            fullLog += "\n"
    return fullLog

def _buildDataLogString(timestamp, log):
    """
    Constructs the string that will be output for data logTypes.
    """
    reportName = log["reportType"].capitalize() + " Report"
    fullLog = _logHeader(reportName, timestamp)
    fullLog += "\n"
    fullLog += log["reportInfo"]
    return fullLog

def _buildRunLogString(timestamp, log):
    """
    Constructs the string that will be output for run logTypes.
    """
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
            fullLog += _formatRunLine("trainX", log["trainDataPoints"],
                                      log["trainDataFeatures"])
        else:
            fullLog += _formatRunLine(log["trainData"], log["trainDataPoints"],
                                      log["trainDataFeatures"])
    if log.get("trainLabels", False):
        if log["trainLabels"].startswith("OBJECT_#"):
            fullLog += _formatRunLine("trainY", log["trainLabelsPoints"],
                                      log["trainLabelsFeatures"])
        else:
            fullLog += _formatRunLine(log["trainLabels"],
                                      log["trainLabelsPoints"],
                                      log["trainLabelsFeatures"])
    if log.get("testData", False):
        if log["testData"].startswith("OBJECT_#"):
            fullLog += _formatRunLine("testX", log["testDataPoints"],
                                      log["testDataFeatures"])
        else:
            fullLog += _formatRunLine(log["testData"], log["testDataPoints"],
                                      log["testDataFeatures"])
    if log.get("testLabels", False):
        if log["testLabels"].startswith("OBJECT_#"):
            fullLog += _formatRunLine("testY", log["testLabelsPoints"],
                                      log["testLabelsFeatures"])
        else:
            fullLog += _formatRunLine(log["testLabels"],
                                      log["testLabelsPoints"],
                                      log["testLabelsFeatures"])
    fullLog += "\n"
    # parameter data
    if log.get("arguments", False):
        argString = "Arguments: "
        argString += _dictToKeywordString(log["arguments"])
        for string in wrap(argString, 79, subsequent_indent=" "*11):
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
    """
    Constructs the string that will be output for crossVal logTypes.
    """
    crossVal = "Cross Validating for {0}".format(log["learner"])
    fullLog = _logHeader(crossVal, timestamp)
    fullLog += "\n"
    if isinstance(log["learnerArgs"], dict):
        fullLog += "Variable Arguments: "
        fullLog += _dictToKeywordString(log["learnerArgs"])
        fullLog += "\n\n"
    folds = log["folds"]
    metric = log["metric"]
    fullLog += "{0}-folding using {1} ".format(folds, metric)
    fullLog += "optimizing for min values\n\n"
    fullLog += _formatRunLine("Result", "Arguments")
    for arguments, result in log["performance"]:
        argString = _dictToKeywordString(arguments)
        fullLog += "{0:<20.3f}{1:20s}".format(result, argString)
        fullLog += "\n"
    return fullLog

def _buildDefaultLogString(timestamp, logType, log):
    """
    Constructs the string that will be output for any unrecognized
    logTypes. Formatting varies based on string, list and dictionary
    types passed as the log.
    """
    fullLog = _logHeader(logType, timestamp)
    if isinstance(log, six.string_types):
        for string in wrap(log, 79):
            fullLog += string
            fullLog += "\n"
    elif isinstance(log, list):
        listString = _formatRunLine(log)
        for string in wrap(listString, 79):
            fullLog += string
            fullLog += "\n"
    else:
        dictString = _dictToKeywordString(log)
        for string in wrap(dictString, 79):
            fullLog += string
            fullLog += "\n"
    return fullLog

def _dictToKeywordString(dictionary):
    """
    Formats dictionaries to be more human-readable.
    """
    kvStrings = []
    for key, value in dictionary.items():
        string = "{0}={1}".format(key, value)
        kvStrings.append(string)
    return ", ".join(kvStrings)

def _formatRunLine(*args):
    """
    Formats equally spaced values for each column.
    """
    args = list(map(str, args))
    nextLine = args.copy()
    lineLog = ""
    equalSpace = int(79 / len(args))
    for arg in args:
        whitespace = equalSpace - len(arg)
        if len(arg) < equalSpace:
            lineLog += arg + " " * whitespace
        else:
            lineLog += arg[:equalSpace - 4] + "... "
    lineLog += "\n"

    return lineLog

def _logHeader(left, right):
    """
    Formats the first line of each log entry.
    """
    lineLog = "\n"
    lineLog += "{0:60}{1:>19}\n".format(left, right)
    return lineLog

#######################
### Initialization  ###
#######################

def initLoggerAndLogConfig():
    """
    Sets up or reads configuration options associated with logging, and
    initializes the currently active logger object using those options.
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
