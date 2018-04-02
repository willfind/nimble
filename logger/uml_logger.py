from __future__ import absolute_import
from __future__ import print_function
import os
import time
import six
import sqlite3

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
        self.logLocation = logLocation
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

    def insertIntoLog(self, tableName, rowValues):
        """
            Generic function to write a sql insert into to this object's log file.
            statement is sqlite code with ? as placeholder for variables
            values must be a tuple of variable valuables
        """

        placeholder_list = ["?" for value in rowValues]
        placeholders = ",".join(placeholder_list)
        statement = "INSERT INTO {table} VALUES ({placeholders});"
        statement = statement.format(table=tableName, placeholders=placeholders)
        self.cursor.execute(statement, rowValues)
        self.connect.commit()

    def logRun(self, trainData, trainLabels, testData, testLabels,
                               function, metrics, predictions, performance, timer,
                               extraInfo=None, numFolds=None):
        """
            logRun directs the data from the arguments and additional data derived from the
            arguments to the appropriate SQL tables and generates a sequential run number
            and inserts the values into each table.

            |------------------------------------------------------------------------------|
            | Table        | Columns                                                       |
            |--------------|---------------------------------------------------------------|
            | runs         | timestamp, runNumber, learnerName,                            |
            |              | trainDataName, trainDataPath, numTrainPoint, numTrainFeatures,|
            |              | testDataName, testDataPath, numTestPoints, numTestFeatures,   |
            |              | customParams, errorMetrics, timer, crossValidation, epochs    |
            |--------------|---------------------------------------------------------------|
            | parameters   | runNumber, paramName, paramValue, parametersID                |
            |--------------|---------------------------------------------------------------|
            | metrics      | runNumber, metricName, metricValue, metricsID                 |
            |--------------|---------------------------------------------------------------|
            | timers       | runNumber, timedProcess, processingTime, timersID             |
            |--------------|---------------------------------------------------------------|
            | cv           | runNumber, foldNumber, fold_score, cvID                       |
            |--------------|---------------------------------------------------------------|
            | epochs       | runNumber, epochNumber, epochLoss, epochTime, epochsID        |
            |--------------|---------------------------------------------------------------|
        """

        timestamp = (time.strftime('%Y-%m-%d %H:%M:%S'))

        runNumber = self.getNextID('runs', 'runNumber')

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
            self._logDictionary(extraInfo, 'parameters', runNumber)
            customParameters = 'True'
        else:
            customParameters = 'False'

        if metrics is not None:
            self.logMetrics(runNumber, metrics, performance)
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
            self._logDictionary(timer.cumulativeTimes, 'timers', runNumber)
            timed = 'True'
        else:
            timed = None

        table = 'runs'
        values = (timestamp, runNumber, functionCall, trainDataName, trainDataPath,
                  numTrainPoints, numTrainFeatures, testDataName, testDataPath,
                  numTestPoints, numTestFeatures, customParameters, errorMetrics, timed,
                  crossValidation, epochData)

        self.insertIntoLog(table, values)

    def logParameters(self, runNumber, parameters):
        """ Logs parameter names and values used in run into the parameters table"""
        table = 'parameters'
        self._logDictionary(parameters, table, runNumber)


    def logTimers(self, runNumber, timers):
        """ Logs timer values used in run in the timers table"""
        table = 'timers'
        self._logDictionary(parameters, table, runNumber)


    def logMetrics(self, runNumber, metrics, performance):
        """ Logs metric names and values into the metrics table"""
        nextID = self.getNextID('metrics', 'metricsID')
        for metric, perf in zip(metrics, performance):
            metricName = metric.__name__
            values = (runNumber, metricName, perf, nextID)
            self.insertIntoLog('metrics', values)
            nextID += 1


    def _logDictionary(self, dictionary, table, runNumber):
        """ Logs dictionary type arguments into the specified table
        with the designated runNumber"""
        column = table + "ID"
        nextID = self.getNextID(table, column)
        for key, value in six.iteritems(dictionary):
            values = (runNumber, key, value, nextID)
            self.insertIntoLog(table, values)
            nextID += 1


    def _showLogImplementation(self, levelOfDetail, leastRunsAgo, mostRunsAgo, startDate,
                               endDate, saveToFileName, maximumEntries, searchForText):
        """ Implementation of showLog function for UML"""
        nextRun = self.getNextID('runs', 'runNumber')
        startRun = nextRun - mostRunsAgo
        endRun = nextRun - leastRunsAgo
        runNumbers = range(startRun, endRun)

        if startDate is not None and endDate is not None:
            query = "SELECT runNumber FROM runs WHERE timestamp >= ? and timestamp <= ?"
            values = (startDate, endDate)
            self.cursor.execute(query, values)
            fetchRows = self.cursor.fetchall()
            # fetchall returns a list of tuples [(0,), (1,), ...]
            runNumbers = []
            for value in fetchRows:
                runNumbers.append(value[0])


        if levelOfDetail == 1:
            fullLog = '*' * 35 + " RUN LOGS " + '*' * 35
            fullLog += "\n"
            for runNumber in runNumbers:
                fullLog += "\n"
                fullLog += self.buildLevel1String(runNumber)
                fullLog += '*' * 80
                fullLog += "\n"
            if saveToFileName is not None:
                filePath = os.path.join(self.logLocation, saveToFileName)
                with open(filePath, mode='w') as f:
                    f.write(fullLog)
            else:
                print(fullLog)


    def buildLevel1String(self, runNumber, maximumEntries=100, searchForText=None):
        """ Extracts and formats information from the 'runs' table for printable output """
        query = "SELECT * FROM runs WHERE runNumber = ?;"
        values = tuple((runNumber,))
        self.cursor.execute(query, values)
        fetchRows = self.cursor.fetchall()
        if fetchRows == []:
            fullLog = "No Results Found"
            fullLog += "\n"
        else:
            columnNames = [descripton[0] for descripton in self.cursor.description]
            for row in fetchRows:
                # convert all values to strings for concatenation and printing
                row = map(str, row)
                # add timestamp, runNumber and learnerName
                fullLog = columnNames[0]+ ': ' + row[0] + '\n'
                fullLog += columnNames[1]+ ': ' + row[1] + '\n'
                fullLog += columnNames[2]+ ': ' + row[2] + '\n\n'
                # add training set data, if present
                trainDataColNames = [columnNames[3], columnNames[4], columnNames[5], columnNames[6]]
                trainDataRowValues = [row[3], row[4], row[5], row[6]]
                fullLog += _formatRunLine(trainDataColNames, trainDataRowValues)
                # add testing set data, if present
                testDataColNames = [columnNames[7], columnNames[8], columnNames[9], columnNames[10]]
                testDataRowValues = [row[7], row[8], row[9], row[10]]
                fullLog += _formatRunLine(testDataColNames, testDataRowValues)
                # add information about other available data tables
                otherTablesColNames = [columnNames[11], columnNames[12], columnNames[13],
                                          columnNames[14], columnNames[15]]
                otherTablesRowValues = [row[11], row[12], row[13], row[14], row[15]]
                fullLog += _formatRunLine(otherTablesColNames, otherTablesRowValues)

        return fullLog


    def buildLevel2String(self, runNumber, maximumEntries=100, searchForText=None):
        pass


    def getNextID(self, table, column):
        """ Returns the maximum number in the given column for the specified table """
        query = "SELECT {column} FROM {table} ORDER BY {column} DESC LIMIT 1"
        query = query.format(column=column, table=table)
        self.cursor.execute(query)
        try:
            lastNumber = self.cursor.fetchone()
            # fetchone returns a tuple (0,)
            nextNumber = lastNumber[0] + 1
        except TypeError:
            nextNumber = 0

        return nextNumber

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


#######################
### Generic Helpers ###
#######################

def _formatRunLine(columnNames, rowValues):
    """ Formats """
    columnNames, rowValues = _removeItemsWithoutData(columnNames, rowValues)
    if columnNames == []:
        return ""
    lineLog = ("{:20s}" * len(columnNames)).format(*columnNames)
    lineLog += "\n"
    lineLog += ("{:20s}" * len(rowValues)).format(*rowValues)
    lineLog += "\n\n"

    return lineLog


def _removeItemsWithoutData(columnNames, rowValues):
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
    ProcessingTime real,
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
