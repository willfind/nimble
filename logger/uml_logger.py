from __future__ import absolute_import
import os
import time
import six
import sqlite3

import UML
from UML.exceptions import ArgumentException


"""
    Handle logging of creating and testing learners.  Currently
    creates two versions of a log for each run:  one that is human-readable,
    and one that is machine-readable (csv).
    Should report, for each run:
    Size of input data
    # of features  (columns)
    # of points (rows)
    # points used for training
    # points used for testing

    Name of package (mlpy, scikit learn, etc.)
    Name of learner
    parameters of learner
    performance metric(s)
    results of performance metric
    Any additional information
"""


def initSQLTables(connection, cursor):
    initLevel1Table = """
    CREATE TABLE IF NOT EXISTS runs (
    timestamp date,
    runNumber int PRIMARY KEY,
    functionCall text,
    trainDataName text,
    trainDataPath text,
    numTrainPoints int,
    numTrainFeatures int,
    testDataName text,
    testDataPath text,
    numTestPoints int,
    numTestFeatures int,
    customParams int,
    errorMetrics int,
    crossValidation int,
    epochData int,
    timer real);"""
    cursor.execute(initLevel1Table)
    connection.commit()

    initLevel2Table = """
    CREATE TABLE IF NOT EXISTS parameters (
    runNumber int,
    paramName text,
    paramValue text,
    paramID int PRIMARY KEY,
    FOREIGN KEY (runNumber) REFERENCES runs(runNumber));"""
    cursor.execute(initLevel2Table)
    connection.commit()

    initLevel3Table = """
    CREATE TABLE IF NOT EXISTS metrics (
    runNumber int,
    metricName text,
    metricValue text,
    metricID int PRIMARY KEY,
    FOREIGN KEY (runNumber) REFERENCES runs(runNumber));"""
    cursor.execute(initLevel3Table)
    connection.commit()

    initLevel4Table = """
    CREATE TABLE IF NOT EXISTS crossValidation (
    runNumber int,
    foldNumber int,
    foldScore real,
    cvID int PRIMARY KEY,
    FOREIGN KEY (runNumber) REFERENCES runs(runNumber));"""
    cursor.execute(initLevel4Table)
    connection.commit()

    initLevel5Table = """
    CREATE TABLE IF NOT EXISTS epochs (
    runNumber int,
    epochNumber int,
    epochLoss real,
    epochTime real,
    epochID int PRIMARY KEY,
    FOREIGN KEY (runNumber) REFERENCES runs(runNumber));"""
    cursor.execute(initLevel5Table)
    connection.commit()


class UmlLogger(object):
    def __init__(self, logLocation, logName):
        fullLogDesignator = os.path.join(logLocation, logName)
        self.logFileName = fullLogDesignator + ".mr"
        self.isAvailable = self.setup(self.logFileName)

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
        self.connect = sqlite3.connect(self.logFileName)
        self.cursor = self.connect.cursor()
        initSQLTables(self.connect, self.cursor) # check

        return True

    def cleanup(self):
        # only need to call if we have previously called setup
        if self.isAvailable:
            self.connect.close()

    def logStatement(self, statement, values):
        """
            Generic function to write a message to this object's log file.  Does
            no formatting; just writes whatever is in 'message' to the file.  Attempt
            to open the log file if it has not yet been opened.  By default, adds a
            new line to each message sent, but if addNewLine is false, will not put
            the message on a new line.
        """

        #if the log file hasn't been created, we try to create it now
        if self.isAvailable:
            pass
        else:
            self.setup()

        self.cursor.execute(statement, values)
        self.connect.commit()

    # def logRun(self, trainData, trainLabels, testData, testLabels, function, metrics,
    #            predictions, performance, timer, extraInfo=None, numFolds=None):
    #     """
    #     Send pertinent information about a cycle of train a learner and test its performance
    #     to the log file
    #     """
    #     self._logRun_implementation(trainData, trainLabels, testData, testLabels,
    #                                 function, metrics, predictions, performance, timer, extraInfo, numFolds)

    def logRun(self, trainData, trainLabels, testData, testLabels,
                               function, metrics, predictions, performance, timer,
                               extraInfo=None, numFolds=None):
        """
            Write one (data + classifer + error metrics) combination to a log file
            in machine readable format.  Should include as much information as possible,
            to allow someone to reproduce the test.  Information included:
            # of training data points
            # of testing data points
            # of features in training data
            # of features in testing data
            Function defining the classifer (learnerName, parameters, etc.)
            Error metrics computed based on predictions of classifier: name/function and numerical
            result)
            Any additional information, definedy by user, passed as 'extraInfo'

            Format is key:value,key:value,...,key:value
        """

        timestamp = (time.strftime('%Y-%m-%d %H:%M:%S'))

        try:
            runNumber = _getNextRunNumber(self.cursor)
        except AttributeError:
            runNumber = 0
        # get function called in string format
        if isinstance(function, (str, six.text_type)):
            functionCall = function
        else:
            #we get the source code of the function as a list of strings and glue them together
            funcLines = inspect.getsourcelines(function)
            funcString = ""
            for i in range(len(funcLines) - 1):
                funcString += str(funcLines[i])
            if funcLines is None:
                funcLines = "N/A"
            functionCall = funcString

        trainDataName = None
        trainDataPath = None
        testDataName = None
        testDataPath = None

        #log info about training data, if present
        if trainData is not None:
            #If present, add name and path of source files for train and test data
            if trainData.name is not None:
                trainDataName = trainData.name
            if trainData.path is not None:
                trainDataPath = trainData.path
            numTrainPoints = trainData.points
            numTrainFeatures = trainData.features

        numTestPoints = 0
        numTestFeatures = 0
        #log info about testing data, if present
        if testData is not None:
            if testData.name is not None:
                testDataName = testData.name
            if testData.path is not None:
                testDataPath = testData.path
            numTestPoints = testData.points
            numTestFeatures = testData.features

        # SQLite does not have a boolean type; using 0 or 1
        if extraInfo is not None:
            #TODO insert into parameters table
            customParams = 1
        else:
            customParams = 0

        if numFolds is not None:
            # TODO insert into crossValidation table
            cross_validation = 1
        else:
            crossValidation = 0

        if metrics is not None:
            # TODO insert into metrics table
            errorMetrics = 1
        else:
            errorMetrics = 0

        # TODO add epoch information for neural nets; maybe by checking function?
        # if epochData is not None:
        #    insert into epochs table
        #    epochData = 1
        epochData = 0

        timer_time = 0
        for eachTime in timer.cumulativeTimes:
            timer_time += timer.cumulativeTimes[eachTime]

        logLine = ("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);")
        values = (timestamp, runNumber, functionCall, trainDataName, trainDataPath,
                  numTrainPoints, numTrainFeatures, testDataName, testDataPath,
                  numTestPoints, numTestFeatures, customParams, errorMetrics,
                  crossValidation, epochData, timer_time)

        self.logStatement(logLine, values)



    def logData(self, baseDataObject):
        """
        Send pertinent information about a data object that has been loaded/created to the log file
        """
        self._logData_implementation(baseDataObject)


    def logLoad(self, dataFileName, baseDataType=None, name=None):
        """
        Send pertinent information about the loading of some data set to the log file
        """
        if dataFileName is None and baseDataType is None and name is None:
            raise ArgumentException("logLoad requires at least one non-None argument")
        else:
            self._logLoad_implementation(dataFileName, baseDataType, name)

    def logCrossValidation(self, trainData, trainLabels, learnerName, metric, performance,
                           timer, learnerArgs, folds=None):
        """
        Send information about selection of a set of parameters using cross validation
        """
        self._logCrossValidation_implemention(trainData, trainLabels, learnerName, metric,
                                              performance, timer, learnerArgs, folds)

def addTableValue(value, dataType):
    if dataType == 'int' or dataType == 'real':
        logLine = "{}, ".format(value)
        return logLine
    else:
        logLine = "'{}', ".format(value)
        return logLine


def _getNextRunNumber(cursor):
    query = "SELECT runNumber FROM runs ORDER BY runNumber DESC LIMIT 1"
    cursor.execute(query)
    try:
        lastRun = cursor.fetchone()[0] #returns tuple
    except TypeError:
        lastRun = -1
    nextRun = lastRun + 1

    return nextRun

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
    UML.logger.active = UmlLogger(location, name)
