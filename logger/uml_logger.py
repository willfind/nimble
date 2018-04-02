from __future__ import absolute_import
from __future__ import print_function
import os
import time
import six
import sqlite3

import UML
from UML.exceptions import ArgumentException

#TODO edit description
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

    def insertIntoLog(self, table, values):
        """
            Generic function to write a sql insert into to this object's log file.
            statement is sqlite code with ? as placeholder for variables
            values are a tuple of variable valuables
        """

        #if the log file hasn't been created, we try to create it now
        if self.isAvailable:
            pass
        else:
            self.setup()

        placeholder_list = ["?" for value in values]
        placeholders = ",".join(placeholder_list)
        statement = "INSERT INTO {table} VALUES ({placeholders});"
        statement = statement.format(table=table, placeholders=placeholders)
        self.cursor.execute(statement, values)
        self.connect.commit()

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

        table = 'runs'

        timestamp = (time.strftime('%Y-%m-%d %H:%M:%S'))

        lastRun = self.getColumnMax(table, 'runNumber')
        runNumber = lastRun + 1

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
        numTrainPoints = None
        numTrainFeatures = None

        #log info about training data, if present
        if trainData is not None:
            #If present, add name and path of source files for train and test data
            if trainData.name is not None:
                trainDataName = trainData.name
            if trainData.path is not None:
                trainDataPath = trainData.path
            numTrainPoints = trainData.points
            numTrainFeatures = trainData.features

        numTestPoints = None
        numTestFeatures = None
        #log info about testing data, if present
        if testData is not None:
            if testData.name is not None:
                testDataName = testData.name
            if testData.path is not None:
                testDataPath = testData.path
            numTestPoints = testData.points
            numTestFeatures = testData.features

        # SQLite does not have a boolean type
        if extraInfo is not None and extraInfo is not {}:
            print(extraInfo)
            self.logRunDetailDicts('parameters', runNumber, extraInfo)
            customParameters = 'True'
        else:
            customParameters = 'False'

        if metrics is not None:
            #self.logRunDetails('metrics', runNumber, metrics)
            errorMetrics = 'True'
        else:
            errorMetrics = 'False'

        if numFolds is not None:
            # TODO insert into crossValidation table
            cross_validation = 'True'
        else:
            crossValidation = 'False'

        # TODO add epoch information for neural nets; maybe by checking function?
        # if epochData is not None:
        #    insert into epochs table
        #    epochData = 1
        epochData = 'False'

        if timer is not None and timer.cumulativeTimes is not {}:
            self.logRunDetailDicts('timers', runNumber, timer.cumulativeTimes)
            timed = 'True'
        else:
            timed = None
        # for eachTime in timer.cumulativeTimes:
        #     timer_time += timer.cumulativeTimes[eachTime]

        values = (timestamp, runNumber, functionCall, trainDataName, trainDataPath,
                  numTrainPoints, numTrainFeatures, testDataName, testDataPath,
                  numTestPoints, numTestFeatures, customParameters, errorMetrics, timed,
                  crossValidation, epochData)

        self.insertIntoLog(table, values)


    def logRunDetailDicts(self, table, runNumber, details):
        """ logs details (parameters, metrics, and timer) from each run """
        column = table + "ID"
        uniqueID = self.getColumnMax(table, column)
        for detailName, detailValue in six.iteritems(details):
            uniqueID += 1
            values = (runNumber, detailName, detailValue, uniqueID)
            self.insertIntoLog(table, values)

    def _showLogImplementation(self, levelOfDetail, leastRunsAgo, mostRunsAgo, startDate,
                               endDate, saveToFileName, maximumEntries, searchForText):
        """ Implementation of showLog function for UML"""
        lastRun = self.getColumnMax('runs', 'runNumber')
        startRun = lastRun - mostRunsAgo
        endRun = lastRun - leastRunsAgo
        if levelOfDetail == 1:
            fullLog = '*' * 35 + " RUN LOGS " + '*' * 35
            fullLog += "\n"
            for runNumber in range(startRun, endRun + 1):
                fullLog += "\n"
                fullLog += self.buildLevel1String(runNumber)
                fullLog += '*' * 80
                fullLog += "\n"
            print(fullLog)


    def buildLevel1String(self, runNumber, maximumEntries=100, searchForText=None):
        """ """
        query = "SELECT * FROM runs WHERE runNumber = ?;"
        values = tuple((runNumber,))
        self.cursor.execute(query, values)
        rows = self.cursor.fetchall()
        if rows == []:
            fullLog = "No Results Found"
        else:
            columnNames = [descripton[0] for descripton in self.cursor.description]
            for row in rows:
                # add timestamp, runNumber and learnerName to first row
                # convert all values to strings for concatenation and printing
                row = map(str, row)
                fullLog = columnNames[0]+ ': ' + row[0] + '\n'
                fullLog += columnNames[1]+ ': ' + row[1] + '\n'
                fullLog += columnNames[2]+ ': ' + row[2] + '\n\n'
                # add data about training set, if present
                trainDataColNames = [columnNames[3], columnNames[4], columnNames[5], columnNames[6]]
                trainDataRowValues = [row[3], row[4], row[5], row[6]]
                fullLog += formatRunLine(trainDataColNames, trainDataRowValues)
                # add data about testing set, if present
                testDataColNames = [columnNames[7], columnNames[8], columnNames[9], columnNames[10]]
                testDataRowValues = [row[7], row[8], row[9], row[10]]
                fullLog += formatRunLine(testDataColNames, testDataRowValues)
                # add information about other available data tables
                otherTablesColNames = [columnNames[11], columnNames[12], columnNames[13],
                                          columnNames[14], columnNames[15]]
                otherTablesRowValues = [row[11], row[12], row[13], row[14], row[15]]
                fullLog += formatRunLine(otherTablesColNames, otherTablesRowValues)

        return fullLog

    def getColumnMax(self, table, column):
        """ Returns the maximum number in the given column for the specified table """
        query = "SELECT {column} FROM {table} ORDER BY {column} DESC LIMIT 1"
        query = query.format(column=column, table=table)
        self.cursor.execute(query)
        try:
            lastNumber = self.cursor.fetchone()[0]
        except TypeError:
            lastNumber = -1

        return lastNumber

    # def logData(self, baseDataObject):
    #     """
    #     Send pertinent information about a data object that has been loaded/created to the log file
    #     """
    #     self._logData_implementation(baseDataObject)
    #
    #
    # def logLoad(self, dataFileName, baseDataType=None, name=None):
    #     """
    #     Send pertinent information about the loading of some data set to the log file
    #     """
    #     if dataFileName is None and baseDataType is None and name is None:
    #         raise ArgumentException("logLoad requires at least one non-None argument")
    #     else:
    #         self._logLoad_implementation(dataFileName, baseDataType, name)
    #
    # def logCrossValidation(self, trainData, trainLabels, learnerName, metric, performance,
    #                        timer, learnerArgs, folds=None):
    #     """
    #     Send information about selection of a set of parameters using cross validation
    #     """
    #     self._logCrossValidation_implemention(trainData, trainLabels, learnerName, metric,
    #                                           performance, timer, learnerArgs, folds)


########################
#   Helper Functions   #
########################

def formatRunLine(columnNames, rowValues):
    """ Formats """
    columnNames, rowValues = removeItemsWithoutData(columnNames, rowValues)
    if columnNames == []:
        return ""
    lineLog = ("{:20s}" * len(columnNames)).format(*columnNames)
    lineLog += "\n"
    lineLog += ("{:20s}" * len(rowValues)).format(*rowValues)
    lineLog += "\n\n"

    return lineLog


def removeItemsWithoutData(columnNames, rowValues):
    """ Prevents the Log from displaying columns that do not have a data"""
    keepIndexes = []
    for index, item in enumerate(rowValues):
        if item !=  "False":
            keepIndexes.append(index)
    keepColumnName = []
    keepRowValue = []
    for index in keepIndexes:
        keepColumnName.append(columnNames[index])
        keepRowValue.append(rowValues[index])
    return keepColumnName, keepRowValue


#########################
#    Initialization     #
#########################

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


def initSQLTables(connection, cursor):
    """ creates data and level1-level5 sql tables for logger if the table is not already in the database"""

    initDataTable = """
    CREATE TABLE IF NOT EXISTS data (
    dataID int,
    runNumber int,
    objectName text,
    numPoints int,
    numFeatures int);
    """
    cursor.execute(initDataTable)
    connection.commit()

    initRunsTable = """
    CREATE TABLE IF NOT EXISTS runs (
    timestamp date,
    runNumber int PRIMARY KEY,
    learnerName text,
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
    timed int,
    crossValidation int,
    epochData int);"""
    cursor.execute(initRunsTable)
    connection.commit()

    initParametersTable = """
    CREATE TABLE IF NOT EXISTS parameters (
    runNumber int,
    paramName text,
    paramValue text,
    parametersID int PRIMARY KEY,
    FOREIGN KEY (runNumber) REFERENCES runs(runNumber));"""
    cursor.execute(initParametersTable)
    connection.commit()

    initMetricsTable = """
    CREATE TABLE IF NOT EXISTS metrics (
    runNumber int,
    metricName text,
    metricValue text,
    metricsID int PRIMARY KEY,
    FOREIGN KEY (runNumber) REFERENCES runs(runNumber));"""
    cursor.execute(initMetricsTable)
    connection.commit()

    initTimersTable = """
    CREATE TABLE IF NOT EXISTS timers (
    runNumber int,
    timedProcess text,
    ProcessingTime text,
    timersID int PRIMARY KEY,
    FOREIGN KEY (runNumber) REFERENCES runs(runNumber));"""
    cursor.execute(initTimersTable)
    connection.commit()

    initCVTable = """
    CREATE TABLE IF NOT EXISTS cv (
    runNumber int,
    foldNumber int,
    foldScore real,
    cvID int PRIMARY KEY,
    FOREIGN KEY (runNumber) REFERENCES runs(runNumber));"""
    cursor.execute(initCVTable)
    connection.commit()

    initEpochsTable = """
    CREATE TABLE IF NOT EXISTS epochs (
    runNumber int,
    epochNumber int,
    epochLoss real,
    epochTime real,
    epochsID int PRIMARY KEY,
    FOREIGN KEY (runNumber) REFERENCES runs(runNumber));"""
    cursor.execute(initEpochsTable)
    connection.commit()
