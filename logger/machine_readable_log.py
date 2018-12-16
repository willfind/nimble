from __future__ import absolute_import
import time
import inspect
import re
import types

import UML
from .uml_logger import UmlLogger

from UML.exceptions import ArgumentException
import six
from six.moves import range
from six.moves import zip


class MachineReadableLogger(UmlLogger):
    """
    Class that handles various UML-specific logging tasks, printing in a more machine-friendly
    format than the human-friendly version of logging done by HumanReadableLogger.  Logs created
    by a MachineReadableLogger format should be easy to read and parse into a dictionary-type
    data object, with one logged object per line and each element of a logged object in the
    form name:content
    """

    def __init__(self, logFileName=None):
        UmlLogger.__init__(self, logFileName)

    def _logLoad_implementation(self, dataFileName, baseDataType=None, name=None):
        """
        Send information about the loading of a data file into a UML.data.Base object to the log.
        Includes: name of the data file, type of UML.data.Base object it was loaded into, and the
        name of the object, if any.
        """
        logLine = "{LOAD}::"
        if dataFileName is not None:
            if isinstance(dataFileName, str):
                logLine += "fileName:" + dataFileName
            else:
                raise ArgumentException("dataFileName must be a string")

        if baseDataType is not None:
            if isinstance(baseDataType, str):
                if logLine[-1] != ':':
                    logLine += ",dataObjectType:" + baseDataType
                logLine += "dataObjectType:" + baseDataType
            else:
                raise ArgumentException("baseDataType must be a string")

        if name is not None:
            if isinstance(name, str):
                if logLine[-1] != ':':
                    logLine += ",dataObjectType:" + baseDataType
                logLine += "dataObjectName:" + name
            else:
                raise ArgumentException("name must be a string")

        self.logMessage(logLine)

    def _logData_implementation(self, baseDataObject):
        """
        Send information about a UML.data.Base object to the machine-readable log.  Includes
        name of the object, if present, the type of the object, and the dimensions of the data
        in the object.  The human readable version of this function includes much more data
        about the object and the data it contains.
        """
        if baseDataObject is None:
            return

        logLine = "{DATA}::"

        #boolean to track whether logLine has had info appended to it
        firstInfoAppended = False

        #add the data object's name, if it has one
        if baseDataObject.name is not None:
            logLine += "name:" + str(baseDataObject.name)
            firstInfo = True

        if firstInfoAppended:
            logLine += "," + "type:" + str(baseDataObject.getTypeString())
        else:
            logLine += "type:" + str(baseDataObject.getTypeString())

        if firstInfoAppended:
            logLine += "," + "dimensions:(" + str(len(baseDataObject.points)) + ", " + str(
                len(baseDataObject.features)) + ")"

        self.logMessage(logLine)


    def _logRun_implementation(self, trainData, trainLabels, testData, testLabels,
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
        #Create a string to be logged (as one line), and add dimensions and the function
        logLine = "{RUN}::"
        logLine += createMRLineElement("timestamp", time.strftime('%Y-%m-%d %H:%M:%S'))

        #log info about training data, if present
        if trainData is not None:
            #If present, add name and path of source files for train and test data
            if trainData.name is not None:
                logLine += createMRLineElement("trainDataName", trainData.name)
            if trainData.path is not None:
                logLine += createMRLineElement("trainDataPath", trainData.path)
            logLine += createMRLineElement("numTrainDataPoints", len(trainData.points))
            logLine += createMRLineElement("numTrainDataFeatures", len(trainData.features))

        #log info about testing data, if present
        if testData is not None:
            if testData.name is not None:
                logLine += createMRLineElement("testDataName", testData.name)
            if testData.path is not None:
                logLine += createMRLineElement("testDataPath", testData.path)
            logLine += createMRLineElement("numTestDataPoints", len(testData.points))
            logLine += createMRLineElement("numTestDataFeatures", len(testData.features))

        #add numFolds if it is present - generally when testData is not present, as numFolds
        #implies k-fold cross validation, in which case there is no test set
        if numFolds is not None:
            logLine += createMRLineElement("numFolds", numFolds)

        #add timing info
        #logLine += createMRLineElement("runTime", "{0:.2f}".format(runTime))
        if timer is not None and len(timer.cumulativeTimes) > 0:
            for label in timer.cumulativeTimes.keys():
                duration = timer.calcRunTime(label)
                logLine += createMRLineElement(label, "{0:.2f}".format(duration))

        if isinstance(function, (str, six.text_type)):
            logLine += createMRLineElement("function", function)
        else:
            #we get the source code of the function as a list of strings and glue them together
            funcLines = inspect.getsourcelines(function)
            funcString = ""
            for i in range(len(funcLines) - 1):
                funcString += str(funcLines[i])
            if funcLines is None:
                funcLines = "N/A"
            logLine += createMRLineElement("function", funcString)

        #add any extraInfo to the log string
        if extraInfo is not None:
            for key, value in six.iteritems(extraInfo):
                if isinstance(key, (str, int, float, bool)) and isinstance(value, (str, int, float, bool)):
                    logLine += createMRLineElement(key, value)
                elif isinstance(value, types.FunctionType):
                    logLine += createMRLineElement(key, value.__name__)
                elif isinstance(value, UML.data.Base):
                    logLine += createMRLineElement(key, str(len(value.points)) + ", " + str(len(value.features)) + ")")
                else:
                    logLine += createMRLineElement(key, value)

        if metrics is not None:
            for metric, result in zip(metrics, performance):
                if isinstance(metric, (str, six.text_type)):
                    logLine += createMRLineElement(str(metric), result)
                else:
                    metricLines = inspect.getsourcelines(metric)
                    metricString = ""
                    for i in range(len(metricLines) - 1):
                        metricString += str(metricLines[i])
                    if metricLines is None:
                        metricLines = "N/A"
                    logLine += createMRLineElement(metricString, result)

        if logLine[-1] == ',':
            logLine = logLine[:-1]

        self.logMessage(logLine)


    def _logCrossValidation_implemention(self, trainData, trainLabels, learnerName,
                                         metric, performance, timer, learnerArgs, folds=None):
        pass


def parseLog(pathToLogFile):
    """
        Provided with a path to a log file containing some lines representing machine-readable
        logs of runs, read all such lines and parse them into a list of hashes.  Each hash
        represents one run, and contains the logged information about that run.  Assumes that
        each line in the log file that starts with {RUN} represents a run; any line that doesn't
        start with {RUN} is ignored.
    """
    logFile = open(pathToLogFile, 'r')
    rawRuns = logFile.readlines()
    parsedRuns = []
    for rawRun in rawRuns:
        if rawRun.startswith("{RUN}::"):
            rawRun = rawRun.replace("{RUN}::", "", 1)
            run = parseLoggedRun(rawRun)
            parsedRuns.append(run)

    return parsedRuns


def parseLoggedRun(loggedRun):
    """
        Convert one line of a log file - which represents output information of one run - into
        a dictionary containing the same information, keyed by standard labels (see
        MachineReadableRunLog.logRun() for examples of labels that are used).
    """
    runDict = {}
    #split each logged Run wherever there is a comma not preceded by a backslash
    elementsRaw = re.split(r"([^\\]|^)(\\\\)*,", loggedRun)
    elements = []
    ticker = 0
    for i in range(len(elementsRaw)):
        if i == len(elementsRaw) - 1:
            element = elementsRaw[i]
            if element is not None:
                elements.append(element)
        elif ticker == 0:
            if elementsRaw[i] is not None:
                element = elementsRaw[i]
        elif ticker == 1:
            if elementsRaw[i] is not None:
                element += elementsRaw[i]
        else:
            if elementsRaw[i] is not None:
                element += elementsRaw[i]
            ticker = -1
            elements.append(element)
        #increment ticker
        ticker += 1

    for element in elements:
        parts = re.split(r"([^\\]|^)(\\\\)*:", element)
        #re.split should produce a list w/3 or 4 elements, as it will split a string
        #into the portion before the regular expression, each parenthesized group in the
        #re, and the portion after the re.  Which will be either 3 or 4 parts.
        if len(parts) != 3 and len(parts) != 4:
            raise Exception("Badly formed log element in machine readable log")

        #We need to glue together the results of re.split(), as it will split the key portion
        #of the pair into three parts
        key = parts[0]
        if parts[1] is not None:
            key += parts[1]
        if parts[2] is not None:
            key += parts[2]

        value = parts[3]

        #remove leading or trailing whitespace characters
        value = value.strip()

        #trim one (and only one) leading/trailing quote character
        if value is not None and len(value) > 1:
            if value[0] == "'" or value[0] == '"':
                value = value[1:]
            if value[-1] == "'" or value[-1] == '"':
                value = value[0:-1]

        unSanitizedValue = unSanitizeStringFromLog(value)
        runDict[key] = unSanitizedValue

    return runDict


def createMRLineElement(key, value, addComma=True):
    """
        Combine the key and value into a string suitable for a machine-readable log.
        Returns string of the format 'key:"value",' though the double-quotes around
        value are not included if value is a number.  The trailing comma is added by
        default but will be ommitted if addComma is False.
    """
    if isinstance(value, six.string_types):
        processedValue = "\"" + sanitizeStringForLog(value) + "\""
    else:
        processedValue = str(value)
    result = key + ":" + processedValue
    if addComma:
        result += ","

    return result


def unSanitizeStringFromLog(sanitizedString):
    """
        Replace escaped versions of characters within sanitizedString with the original,
        unescaped version.  Mirror opposite of sanitizeStringForLog: where sanitize
        replaces newLines with '\n', unSanitize replaces '\n' with a newline.
    """
    if len(sanitizedString) < 2:
        return sanitizedString

    unsanitizedString = sanitizedString

    if re.search(r"[^\\](\\\\)*\\\\$", unsanitizedString) != None:
        unsanitizedString = unsanitizedString[:-1]

    done = False
    while not done:
        if re.search(r"([^\\]|^)((\\\\)*)(\\n)", unsanitizedString) is not None:
            unsanitizedString = re.sub(r"([^\\]|^)((\\\\)*)(\\)(n)", '\g<1>\g<2>\n', unsanitizedString, 1)
        else:
            done = True

    done = False
    while not done:
        if re.search(r"([^\\]|^)((\\\\)*)(\\r)", unsanitizedString) is not None:
            unsanitizedString = re.sub(r"([^\\]|^)((\\\\)*)(\\)(r)", '\g<1>\g<2>\r', unsanitizedString, 1)
        else:
            done = True

    done = False
    while not done:
        if re.search(r"([^\\]|^)((\\\\)*)(\\)[\":,]", unsanitizedString) is not None:
            unsanitizedString = re.sub(r"([^\\]|^)((\\\\)*)(\\)([\":,])", '\g<1>\g<2>\g<5>', unsanitizedString, 1)
        else:
            done = True

    return unsanitizedString


def sanitizeStringForLog(rawString):
    """
        Escape all characters in rawString that may interfere with the machine-readable
        logging format: double-quotes, commas, colons, carriage returns, line-feeds, and
        backslashes.  Takes a string as argument; returns that string with an additional
        backslash in front of any of the above-mentioned special characters.
    """

    if rawString is None or rawString == "":
        return ""

    sanitizedString = rawString
    #escape trailing backslash, if it is not already escaped
    if re.search(r"[^\\](\\\\)*\\$", sanitizedString) != None:
        sanitizedString = sanitizedString + '\\'

    #add preceding backslash to all special characters
    sanitizedString = sanitizedString.replace("\"", "\\\"")
    sanitizedString = sanitizedString.replace("\n", "\\n")
    sanitizedString = sanitizedString.replace("\r", "\\r")
    sanitizedString = sanitizedString.replace(":", "\:")
    sanitizedString = sanitizedString.replace(",", "\,")

    return sanitizedString
