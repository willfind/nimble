from __future__ import absolute_import

import os
import shutil
import time
import ast
from nose import with_setup

import UML
from UML.helpers import generateClassificationData
from UML.calculate import rootMeanSquareError as RMSE

"""
Unit tests for functionality of the UMLLogger

"""
def setup_func():
    location = UML.settings.get("logger", "location")
    name = UML.settings.get("logger", "name")
    pathToFile = os.path.join(location, name + ".mr")

    if os.path.exists(location):
        shutil.rmtree(location)

    UML.settings.set("logger", "enabledByDefault", "True")
    UML.logger.uml_logger.initLoggerAndLogConfig()

def teardown_func():
    UML.logger.active.cleanup()
    UML.settings.set("logger", "enabledByDefault", "False")

def singleValueQueries(*queries):
    out = []
    for query in queries:
        valueList = UML.logger.active.extractFromLog(query)
        singleValue = valueList[0][0]
        out.append(singleValue)
    return out

@with_setup(setup_func, teardown_func)
def testLogDirectoryAndFileSetup():
    """assert a new directory and log file are created with first attempt to log"""
    location = UML.settings.get("logger", "location")
    name = UML.settings.get("logger", "name")
    pathToFile = os.path.join(location, name + ".mr")
    X = UML.createData("Matrix", [])

    assert os.path.exists(location)
    assert os.path.exists(pathToFile)

#############
### INPUT ###
#############

@with_setup(setup_func, teardown_func)
def testTopLevelInputFunction():
    """assert the UML.log function correctly inserts data into the log"""
    logType = "insertAndExtract"
    logInfo = {"test": "testInsertAndExtract"}
    UML.log(logType, logInfo)
    # select all columns from the last entry into the logger
    query = "SELECT * FROM logger ORDER BY entry DESC LIMIT 1"
    lastLog = UML.logger.active.extractFromLog(query)
    lastLog = lastLog[0]

    assert lastLog[0] == 1
    assert lastLog[2] == 0
    assert lastLog[3] == logType
    assert lastLog[4] == str(logInfo)

@with_setup(setup_func, teardown_func)
def testNewRunNumberEachSetup():
    """assert that a new, sequential runNumber is generated each time the log file is reopened"""
    for run in range(5):
        logType = "newRunNumber"
        logInfo = {"test": "testNewRunNumberEachSetup"}
        UML.logger.active.insertIntoLog(logType, logInfo)
        # cleanup will require setup before the next log entry
        UML.logger.active.cleanup()
    query = "SELECT runNumber FROM logger"
    lastLogs = UML.logger.active.extractFromLog(query)

    for entry, log in enumerate(lastLogs):
        assert log[0] == entry

@with_setup(setup_func, teardown_func)
def testTopLevelFunctionsUseLog():
    """assert that each call to a top level function with a useLog argument generates a log entry"""
    lengthQuery = "SELECT COUNT(entry) FROM logger"
    infoQuery = "SELECT logInfo FROM logger ORDER BY entry DESC LIMIT 1"
    lengthExpected = 0
    lengthLog = UML.logger.active.extractFromLog(lengthQuery)[0][0] # returns list of tuples i.e. [(0,)]
    # ensure starting table has no values
    assert lengthLog == lengthExpected

    # data
    trainX = [[1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1],
              [1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1]]
    trainY = [[0], [1], [2], [0], [1], [2], [0], [1], [2], [0], [1], [2]]
    testX = [[1,0,0], [0,1,0], [0,0,1], [1,1,0]]
    testY = [[0], [1], [2], [1]]

    # createData
    trainXObj = UML.createData("Matrix", trainX)
    lengthExpected += 1
    trainYObj = UML.createData("Matrix", trainY)
    lengthExpected += 1
    testXObj = UML.createData("Matrix", testX)
    lengthExpected += 1
    testYObj = UML.createData("Matrix", testY)
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert testYObj.getTypeString() in logInfo

    #normalizeData
    # copy to avoid modifying original data
    trainXNormalize = trainXObj.copy()
    lengthExpected += 1
    testXNormalize = testXObj.copy()
    lengthExpected += 1
    UML.normalizeData('mlpy.PCA', trainXNormalize, testX=testXNormalize, arguments={'k': 1})
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'normalizeData'" in logInfo

    #train
    trainedLearner = UML.train("sciKitLearn.SVC", trainXObj, trainYObj)
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'train'" in logInfo

    #trainAndApply
    predictions = UML.trainAndApply("sciKitLearn.SVC", trainXObj, trainYObj, testXObj)
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'trainAndApply'" in logInfo

    #trainAndTest
    results = UML.trainAndTest("sciKitLearn.SVC", trainXObj, trainYObj, testXObj, testYObj,
                                 performanceFunction=RMSE)
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'trainAndTest'" in logInfo

    #trainAndTestOnTrainingData
    results = UML.trainAndTestOnTrainingData("sciKitLearn.SVC", trainXObj, trainYObj,
                                             performanceFunction=RMSE)
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'trainAndTestOnTrainingData'" in logInfo

@with_setup(setup_func, teardown_func)
def testBaseObjectFunctionsUseLog():
    lengthQuery = "SELECT COUNT(entry) FROM logger"
    infoQuery = "SELECT logInfo FROM logger ORDER BY entry DESC LIMIT 1"
    lengthExpected = 0
    lengthLog = UML.logger.active.extractFromLog(lengthQuery)[0][0] # returns list of tuples i.e. [(0,)]
    # ensure starting table has no values
    assert lengthLog == lengthExpected

    data = [["a", 1], ["a", 1], ["a", 1], ["a", 1], ["a", 1], ["a", 1],
            ["b", 2], ["b", 2], ["b", 2], ["b", 2], ["b", 2], ["b", 2],
            ["c", 3], ["c", 3], ["c", 3], ["c", 3], ["c", 3], ["c", 3]]

    # dropFeaturesContainingType; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    dataObj.dropFeaturesContainingType(str)
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'dropFeaturesContainingType'" in logInfo

    # replaceFeatureWithBinaryFeatures; createData not logged

    dataObj = UML.createData("Matrix", data, useLog=False)
    dataObj.replaceFeatureWithBinaryFeatures(0)
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'replaceFeatureWithBinaryFeatures'" in logInfo

    # transformFeatureToIntegers; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    dataObj.transformFeatureToIntegers(0)
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'transformFeatureToIntegers'" in logInfo

    # extractPointsByCoinToss; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    extracted = dataObj.extractPointsByCoinToss(0.5)
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'extractPointsByCoinToss'" in logInfo

    # calculateForEachPoint; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    calculated = dataObj.calculateForEachPoint(lambda x: [point for point in x])
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'calculateForEachPoint'" in logInfo

    # calculateForEachFeature; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    calculated = dataObj.calculateForEachFeature(lambda x: [point for point in x], features=0)
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'calculateForEachFeature'" in logInfo

    # #TODO mapReducePoints; createData not logged
    # dataObj = UML.createData("Matrix", data, useLog=False)
    # def mapper(x):
    #     pass
    # def reducer(mapping):
    #     pass
    # calculated = dataObj.mapReducePoints(mapper,reducer)
    # lengthExpected += 1
    #
    # logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    # assert logLength == lengthExpected
    # assert "'function': 'mapReducePoints'" in logInfo

    # groupByFeature; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    calculated = dataObj.groupByFeature(by=0)
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'groupByFeature'" in logInfo

    # calculateForEachElement; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    calculated = dataObj.calculateForEachElement(lambda x: len(x), features=0)
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'calculateForEachElement'" in logInfo

    # calculateForEachPoint; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    calculated = dataObj.calculateForEachPoint(lambda x: len(x))
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'calculateForEachPoint'" in logInfo

    # calculateForEachFeature; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    calculated = dataObj.calculateForEachFeature(lambda x: len(x), features=0)
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'calculateForEachFeature'" in logInfo

    # shufflePoints; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    dataObj.shufflePoints()
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'shufflePoints'" in logInfo

    # shuffleFeatures; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    dataObj.shuffleFeatures()
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'shuffleFeatures'" in logInfo

    # trainAndTestSets; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    train, test = dataObj.trainAndTestSets(testFraction=0.5)
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'trainAndTestSets'" in logInfo

    # normalizePoints; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    dataObj.normalizePoints(subtract=0, divide=1)
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'normalizePoints'" in logInfo

    # normalizeFeatures; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    dataObj.normalizeFeatures(subtract=0, divide=1)
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'normalizeFeatures'" in logInfo

    # sortPoints; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    dataObj.sortPoints(sortBy=dataObj.getFeatureName(0))
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'sortPoints'" in logInfo

    # sortFeatures; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    dataObj.sortFeatures(sortBy=dataObj.getFeatureName(0))
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'sortFeatures'" in logInfo

    # extractPoints; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    extracted = dataObj.extractPoints(toExtract=0)
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'extractPoints'" in logInfo

    # extractFeatures; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    extracted = dataObj.extractFeatures(toExtract=0)
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'extractFeatures'" in logInfo

    # sortFeatures; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    dataObj.sortFeatures(sortBy=dataObj.getFeatureName(0))
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'function': 'sortFeatures'" in logInfo

    # featureReport; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    fReport = dataObj[:,1].featureReport()
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'reportType': 'feature'" in logInfo

    # summaryReport; createData not logged
    dataObj = UML.createData("Matrix", data, useLog=False)
    sReport = dataObj.summaryReport()
    lengthExpected += 1

    logLength, logInfo = singleValueQueries(lengthQuery, infoQuery)
    assert logLength == lengthExpected
    assert "'reportType': 'summary'" in logInfo

# @with_setup(setup_func, teardown_func)
# def testUniversalInterfaceUseLog():
#     pass
#
# @with_setup(setup_func, teardown_func)
# def testHelpersUseLog():
#     pass
#
# @with_setup(setup_func, teardown_func)
# def testLogEntriesByType():
#     #logRun test nonstring learnerType
#     pass
#
# @with_setup(setup_func, teardown_func)
# def testLogEntriesByType():
#     pass
#
# @with_setup(setup_func, teardown_func)
# def testHandmadeLogEntriesInput():
#     pass
#
# ##############
# ### OUTPUT ###
# ##############
#
# @with_setup(setup_func, teardown_func)
# def testShowLogSearchFilters():
#     #runNumber, date, text, maxEntries, all permutations
#     pass
#
# @with_setup(setup_func, teardown_func)
# def testShowLogToStdOut():
#     pass
#
# @with_setup(setup_func, teardown_func)
# def testShowLogToFile():
#     #append and overwrite
#     pass
#
# @with_setup(setup_func, teardown_func)
# def testHandmadeLogEntriesOutput():
#     pass
#
#



