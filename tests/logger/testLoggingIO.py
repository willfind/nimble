"""
Unit tests for functionality of the SessionLogger
"""

import os
import shutil
import sys
import sqlite3
import tempfile
import re
import functools
from io import StringIO
import inspect

import numpy as np

import nimble
from nimble.calculate import rootMeanSquareError as RMSE
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.exceptions import InvalidArgumentType
from tests.helpers import raises, patch
from tests.helpers import getDataConstructors

#####################
# Helpers for tests #
#####################

def emptyLogSafetyWrapper(testFunc):
    @functools.wraps(testFunc)
    def wrapped():
        removeLogFile()
        try:
            testFunc()
        finally:
            removeLogFile()
    return wrapped

def prepopulatedLogSafetyWrapper(testFunc):
    @functools.wraps(testFunc)
    def wrapped():
        removeLogFile()
        # change settings and input dummy data into log
        nimble.settings.set('logger', 'enabledByDefault', 'True')
        nimble.settings.set('logger', 'enableDeepLogging', 'True')
        # create an example log file
        variables = ["x1", "x2", "x3", "label"]
        data1 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2],
                 [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
                 [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3],
                 [1, 0, 0, 3], [0, 1, 0, 1], [0, 0, 1, 2]]
        data2 = [[1, 0, 0, 1],
                 [0, 1, 0, 2],
                 [0, 0, 1, 3]]
        # add data to log
        for i in range(5):
            # randomSeed
            nimble.random.setSeed(1)
            # load
            trainObj = nimble.data(source=data1, featureNames=variables)
            testObj = nimble.data(source=data2, featureNames=variables)
            # data
            report = trainObj.report()
            # prep
            trainYObj = trainObj.features.extract(3)
            testYObj = testObj.features.extract(3)
            # run and crossVal
            results = nimble.trainAndTest(
                'nimble.KNNClassifier', RMSE, trainX=trainObj,
                trainY=trainYObj, testX=testObj, testY=testYObj,
                arguments={"k": nimble.Tune([3, 5])})
        # edit log sessionNumbers and timestamps
        location = nimble.settings.get("logger", "location")
        name = nimble.settings.get("logger", "name")
        pathToFile = os.path.join(location, name + ".db")
        conn = sqlite3.connect(pathToFile)
        c = conn.cursor()
        c.execute("UPDATE logger SET timestamp = '2018-03-22 12:00:00' WHERE entry <= 18")
        conn.commit()
        c.execute("UPDATE logger SET sessionNumber = 1, timestamp = '2018-03-23 12:00:00' WHERE entry > 18 AND entry <= 36")
        conn.commit()
        c.execute("UPDATE logger SET sessionNumber = 2, timestamp = '2018-03-23 18:00:00' WHERE entry > 36 AND entry <= 54")
        conn.commit()
        c.execute("UPDATE logger SET sessionNumber = 3, timestamp = '2018-03-25 12:00:00' WHERE entry > 54 AND entry <= 72")
        conn.commit()
        c.execute("UPDATE logger SET sessionNumber = 4, timestamp = '2018-04-24 12:00:00' WHERE entry > 72")
        conn.commit()

        try:
            testFunc()
        finally:
            removeLogFile()
    return wrapped


def removeLogFile():
    nimble.core.logger.active.cleanup()
    location = nimble.settings.get("logger", "location")
    name = nimble.settings.get("logger", "name")
    pathToFile = os.path.join(location, name + ".db")
    if os.path.exists(pathToFile):
        os.remove(pathToFile)

def getLastLogData():
    query = "SELECT logInfo FROM logger ORDER BY entry DESC LIMIT 1"
    valueList = nimble.core.logger.active.extractFromLog(query)
    lastLog = valueList[0][0]
    return lastLog

#############
### SETUP ###
#############

@emptyLogSafetyWrapper
def testLogDirectoryAndFileSetup():
    """assert a new directory and log file are created with first attempt to log"""
    newDirectory = os.path.join(nimble.nimblePath, "notCreatedDirectory")
    nimble.settings.set("logger", "location", newDirectory)
    nimble.settings.set("logger", "name", 'notCreatedFile')
    pathToFile = os.path.join(newDirectory, "notCreatedFile.db")
    assert not os.path.exists(newDirectory)
    assert not os.path.exists(pathToFile)

    X = nimble.data([], useLog=True)

    assert os.path.exists(newDirectory)
    assert os.path.exists(pathToFile)
    shutil.rmtree(newDirectory)

#############
### INPUT ###
#############

@emptyLogSafetyWrapper
def testTopLevelInputFunction():
    """assert the nimble.log function correctly inserts data into the log"""
    header = "input"
    logInfo = {"test": "testInput"}
    nimble.log(header, logInfo)
    # select all columns from the last entry into the logger
    query = "SELECT * FROM logger"
    lastLog = nimble.core.logger.active.extractFromLog(query)
    lastLog = lastLog[0]

    assert lastLog[0] == 1
    assert lastLog[2] == 0
    assert lastLog[3] == "User - " + header
    assert lastLog[4] == str(logInfo)

@emptyLogSafetyWrapper
def testNewSessionNumberEachSetup():
    """assert that a new, sequential sessionNumber is generated each time the log file is reopened"""
    nimble.settings.set('logger', 'enabledByDefault', 'True')

    data = [[],[]]
    for session in range(5):
        nimble.data(data)
        # cleanup will require setup before the next log entry
        nimble.core.logger.active.cleanup()
    query = "SELECT sessionNumber FROM logger"
    lastLogs = nimble.core.logger.active.extractFromLog(query)

    for entry, log in enumerate(lastLogs):
        assert log[0] == entry

@emptyLogSafetyWrapper
def testLoadTypeFunctionsUseLog():
    """tests that nimble.data is being logged"""
    nimble.settings.set('logger', 'enabledByDefault', 'True')
    # data
    trainX = [[1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1],
              [1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1]]
    trainY = [[0], [1], [2], [0], [1], [2], [0], [1], [2], [0], [1], [2]]
    testX = [[1,0,0], [0,1,0], [0,0,1], [1,1,0]]
    testY = [[0], [1], [2], [1]]

    # nimble.data
    trainXObj = nimble.data(trainX, returnType='Matrix')
    logInfo = getLastLogData()
    assert f"'identifier': '{trainXObj.logID}'" in logInfo
    assert "'numPoints': 12" in logInfo
    assert "'numFeatures': 3" in logInfo

    trainYObj = nimble.data(trainY, returnType='List')
    logInfo = getLastLogData()
    assert f"'identifier': '{trainYObj.logID}'" in logInfo
    assert "'numPoints': 12" in logInfo
    assert "'numFeatures': 1" in logInfo

    testXObj = nimble.data(testX, returnType='Sparse')
    logInfo = getLastLogData()
    assert f"'identifier': '{testXObj.logID}'" in logInfo
    assert "'numPoints': 4" in logInfo
    assert "'numFeatures': 3" in logInfo

    testYObj = nimble.data(testY, returnType='DataFrame')
    logInfo = getLastLogData()
    assert f"'identifier': '{testYObj.logID}'" in logInfo
    assert "'numPoints': 4" in logInfo
    assert "'numFeatures': 1" in logInfo

    with tempfile.NamedTemporaryFile(suffix=".pickle") as tmpFile:
        trainXObj.save(tmpFile.name)
        load = nimble.data(tmpFile.name)
    logInfo = getLastLogData()
    assert "'returnType': None" in logInfo
    assert "'numPoints': 12" in logInfo
    assert "'numFeatures': 3" in logInfo

    # the sparsity and seed are also stored for random data
    randomObj = nimble.random.data(5, 5, 0)
    logInfo = getLastLogData()
    assert f"'identifier': '{randomObj.logID}'" in logInfo
    assert "'sparsity': 0" in logInfo
    assert "seed" in logInfo

    # loadTrainedLearner
    tl = nimble.train('nimble.KNNClassifier', trainXObj, trainYObj, arguments={'k': 1})
    with tempfile.NamedTemporaryFile(suffix=".pickle") as tmpFile:
        tl.save(tmpFile.name)
        load = nimble.loadTrainedLearner(tmpFile.name)
    logInfo = getLastLogData()
    assert f"'identifier': '{tl.logID}'" in logInfo
    assert "'learnerName': 'KNNClassifier'" in logInfo
    # all keys and values in learnerArgs are stored as strings
    assert "'learnerArgs': {'k': 1}" in logInfo

@emptyLogSafetyWrapper
def test_setSeed():
    nimble.settings.set('logger', 'enabledByDefault', 'True')
    nimble.random.setSeed(1337)
    logInfo = getLastLogData()

    assert "'action': 'random.setSeed'" in logInfo
    assert "'seed': 1337" in logInfo

    with nimble.random.alternateControl(123):
        logInfo = getLastLogData()
        assert "'action': 'entered random.alternateControl'" in logInfo
        assert "'seed': 123" in logInfo
    logInfo = getLastLogData()
    assert "'action': 'exited random.alternateControl'" in logInfo
    assert "'seed': 123" in logInfo

@emptyLogSafetyWrapper
def testRunTypeFunctionsUseLog():
    """tests that top level and TrainedLearner functions are being logged"""
    nimble.settings.set('logger', 'enabledByDefault', 'True')
    # data
    trainX = [[1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1],
              [1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1]]
    trainY = [[0], [1], [2], [0], [1], [2], [0], [1], [2], [0], [1], [2]]
    testX = [[1,0,0], [0,1,0], [0,0,1], [1,1,0]]
    testY = [[0], [1], [2], [1]]

    trainXObj = nimble.data(trainX, useLog=False)
    trainYObj = nimble.data(trainY, useLog=False)
    testXObj = nimble.data(testX, useLog=False)
    testYObj = nimble.data(testY, useLog=False)

    timePattern = re.compile(r"'time': [0-9]+\.[0-9]+")
    randomSeedPattern = re.compile(r"'randomSeed': [0-9]+")

    # train
    tl = nimble.train("sciKitLearn.SVC", trainXObj, trainYObj)
    logInfo = getLastLogData()
    assert "'function': 'train'" in logInfo
    assert re.search(timePattern, logInfo)
    assert re.search(randomSeedPattern, logInfo)

    # trainAndApply
    predictions = nimble.trainAndApply("sciKitLearn.SVC", trainXObj, trainYObj,
                                       testXObj)
    logInfo = getLastLogData()
    assert "'function': 'trainAndApply'" in logInfo
    assert re.search(timePattern, logInfo)
    assert re.search(randomSeedPattern, logInfo)

    # trainAndTest
    performance = nimble.trainAndTest(
        "sciKitLearn.SVC", RMSE, trainXObj, trainYObj, testXObj, testYObj)
    logInfo = getLastLogData()
    assert "'function': 'trainAndTest'" in logInfo
    # ensure that metrics is storing performanceFunction and result
    assert "'metrics': {'rootMeanSquareError': 0.0}" in logInfo
    assert re.search(timePattern, logInfo)
    assert re.search(randomSeedPattern, logInfo)

    # normalizeData
    # copy to avoid modifying original data
    _ = nimble.normalizeData('sciKitLearn.PCA', trainXObj, testX=testXObj)
    logInfo = getLastLogData()
    assert "'function': 'normalizeData'" in logInfo
    assert re.search(timePattern, logInfo)
    assert re.search(randomSeedPattern, logInfo)

    # trainAndTestOnTrainingData
    results = nimble.trainAndTestOnTrainingData(
        "sciKitLearn.SVC", RMSE, trainXObj, trainYObj)
    logInfo = getLastLogData()
    assert "'function': 'trainAndTestOnTrainingData'" in logInfo
    # ensure that metrics is storing performanceFunction and result
    assert "'metrics': {'rootMeanSquareError': 0.0}" in logInfo
    assert re.search(timePattern, logInfo)
    assert re.search(randomSeedPattern, logInfo)

    # randomSeed for top level functions
    tl = nimble.train("sciKitLearn.SVC", trainXObj, trainYObj, randomSeed=123)
    logInfo = getLastLogData()
    assert "'randomSeed': 123" in logInfo

    pred = nimble.trainAndApply("sciKitLearn.SVC", trainXObj, trainYObj,
                                testXObj, randomSeed=123)
    logInfo = getLastLogData()
    assert "'randomSeed': 123" in logInfo

    res = nimble.trainAndTest("sciKitLearn.SVC", RMSE, trainXObj, trainYObj,
                              testXObj, testYObj, randomSeed=123)
    logInfo = getLastLogData()
    assert "'randomSeed': 123" in logInfo

    trainXNormalize = trainXObj.copy()
    testXNormalize = testXObj.copy()
    nimble.normalizeData('sciKitLearn.PCA', trainXNormalize, testX=testXNormalize,
                         randomSeed=123)
    logInfo = getLastLogData()
    assert "'randomSeed': 123" in logInfo

    results = nimble.trainAndTestOnTrainingData(
        "sciKitLearn.SVC", RMSE, trainXObj, trainYObj, randomSeed=123)
    logInfo = getLastLogData()
    assert "'randomSeed': 123" in logInfo

    # TrainedLearner.apply
    predictions = tl.apply(testXObj)
    logInfo = getLastLogData()
    assert f"'function': '{tl.logID}.apply'" in logInfo
    assert re.search(timePattern, logInfo)
    # randomSeed recorded during training
    assert re.search(randomSeedPattern, logInfo) is None

    # TrainedLearner.test
    performance = tl.test(RMSE, testXObj, testYObj)
    logInfo = getLastLogData()
    assert f"'function': '{tl.logID}.test'" in logInfo
    # ensure that metrics is storing performanceFunction and result
    assert "'metrics': {'rootMeanSquareError': 0.0}" in logInfo
    assert re.search(timePattern, logInfo)
    # randomSeed recorded during training
    assert re.search(randomSeedPattern, logInfo) is None

    nimble.settings.set('logger', 'enableDeepLogging', 'True')

def checkLogContents(funcName, objectID, arguments=None):
    lastLog = getLastLogData()
    expFunc = "'function': '{0}'".format(funcName)
    expID = "'identifier': '{0}'".format(objectID)
    assert expFunc in lastLog
    assert expID in lastLog

    if arguments:
        assert 'arguments' in lastLog
        for argName, argVal in arguments.items():
            expArgs1 = "'{0}': '{1}'".format(argName, argVal)
            # double quotations may wrap the second arg if it contains quotations
            expArgs2 = """'{0}': "{1}" """.format(argName, argVal).strip()
            assert expArgs1 in lastLog or expArgs2 in lastLog
    else:
        assert "'arguments': {}" in lastLog


@emptyLogSafetyWrapper
def testPrepTypeFunctionsUseLog():
    """Test that the functions in base using useLog are being logged"""
    nimble.settings.set('logger', 'enabledByDefault', 'True')

    data = [["a", 1, 1], ["a", 1, 1], ["a", 1, 1], ["a", 1, 1], ["a", 1, 1], ["a", 1, 1],
            ["b", 2, 2], ["b", 2, 2], ["b", 2, 2], ["b", 2, 2], ["b", 2, 2], ["b", 2, 2],
            ["c", 3, 3], ["c", 3, 3], ["c", 3, 3], ["c", 3, 3], ["c", 3, 3], ["c", 3, 3]]

    ########
    # Base #
    ########

    # replaceFeatureWithBinaryFeatures
    dataObj = nimble.data(data, useLog=False)
    dataObj.replaceFeatureWithBinaryFeatures(0)
    checkLogContents('replaceFeatureWithBinaryFeatures', dataObj.logID,
                     {'featureToReplace': 0})

    # transformFeatureToIntegers
    dataObj = nimble.data(data, returnType='List', useLog=False)
    dataObj.transformFeatureToIntegers(0)
    checkLogContents('transformFeatureToIntegers', dataObj.logID, {'featureToConvert': 0})

    # trainAndTestSets
    dataObj = nimble.data(data, name='trainAndTestData', useLog=False)
    train, test = dataObj.trainAndTestSets(testFraction=0.5)
    checkLogContents('trainAndTestSets', 'trainAndTestData', {'testFraction': 0.5})

    # groupByFeature
    dataObj = nimble.data(data, returnType='Sparse', useLog=False)
    calculated = dataObj.groupByFeature(by=0)
    checkLogContents('groupByFeature', dataObj.logID, {'by': 0})

    # transpose
    dataObj = nimble.data(data, returnType='List', useLog=False)
    dataObj.transpose()
    checkLogContents('transpose', dataObj.logID)

    # replaceRectangle
    dataObj = nimble.data(data, useLog=False)
    dataObj.replaceRectangle(1, 2, 0, 4, 0)
    checkLogContents('replaceRectangle', dataObj.logID,
        {'replaceWith': 1, 'pointStart': 2, 'pointEnd': 4, 'featureStart': 0,
         'featureEnd': 0})

    # flatten (point order)
    dataObj = nimble.data(data, returnType='Matrix', useLog=False)
    dataObj.flatten()

    checkLogContents('flatten', dataObj.logID)

    # unflatten; using flattened dataObj from above
    dataObj.unflatten((18, 3))
    checkLogContents('unflatten', dataObj.logID, {'dataDimensions': (18,3)})

    # flatten (feature order)
    dataObj = nimble.data(data, returnType='Sparse', useLog=False)
    dataObj.flatten(order='feature')
    checkLogContents('flatten', dataObj.logID, {'order': 'feature'})

    # unflatten; using flattened dataObj from above
    dataObj.unflatten((18, 3), order='feature')
    checkLogContents('unflatten', dataObj.logID, {'dataDimensions': (18,3),
                                                  'order': 'feature'})

    # merge
    dPtNames = ['p' + str(i) for i in range(18)]
    dFtNames = ['f0', 'f1', 'f2']
    dataObj = nimble.data(data, pointNames=dPtNames,
                          featureNames=dFtNames, useLog=False)
    mData = [[1, 4], [2, 5], [3, 6]]
    mPtNames = ['p0', 'p6', 'p12']
    mFtNames = ['f2', 'f3']
    mergeObj = nimble.data(mData, pointNames=mPtNames,
                           featureNames=mFtNames, useLog=False)
    dataObj.merge(mergeObj, point='intersection', feature='union')
    checkLogContents('merge', dataObj.logID, {"other": mergeObj.logID,
                                              "point": 'intersection'})

    # transformElements
    dataObj = nimble.data(data, useLog=False)
    dataCopy = dataObj.copy()
    calculated = dataCopy.transformElements(lambda x: x, features=0)
    checkLogContents('transformElements', dataCopy.logID,
                     {'toTransform': 'lambda x: x', 'features': 0})

    # calculateOnElements
    dataObj = nimble.data(data, useLog=False)
    calculated = dataObj.calculateOnElements(lambda x: len(x), features=0)
    checkLogContents('calculateOnElements', dataObj.logID,
                     {'toCalculate': "lambda x: len(x)", 'features': 0})

    # matchingElements
    dataObj = nimble.data(data, useLog=False)
    calculated = dataObj.matchingElements(lambda e: e > 2, features=[1, 2])
    checkLogContents('matchingElements', dataObj.logID,
                     {'toMatch': "lambda e: e > 2", 'features': [1, 2]})

    ###################
    # Points/Features #
    ###################

    def simpleMapper(vector):
        vID = vector[0]
        intList = []
        for i in range(1, len(vector)):
            intList.append(vector[i])
        ret = []
        for value in intList:
            ret.append((vID, value))
        return ret

    def simpleReducer(identifier, valuesList):
        total = 0
        for value in valuesList:
            total += value
        return (identifier, total)

    # points.mapReduce
    dataObj = nimble.data(data, returnType='Matrix', useLog=False)
    calculated = dataObj.points.mapReduce(simpleMapper,simpleReducer)
    checkLogContents('points.mapReduce', dataObj.logID, {"mapper": "simpleMapper",
                                                         "reducer": "simpleReducer"})

    # points.calculate
    dataObj = nimble.data(data, useLog=False)
    calculated = dataObj.points.calculate(lambda x: len(x))
    checkLogContents('points.calculate', dataObj.logID, {'function': "lambda x: len(x)"})

    # features.calculate
    dataObj = nimble.data(data, returnType='Matrix', useLog=False)
    calculated = dataObj.features.calculate(lambda x: len(x), features=0)
    checkLogContents('features.calculate', dataObj.logID, {'function': "lambda x: len(x)",
                                                           'features': 0})

    # points.permute
    dataObj = nimble.data(data, returnType='List', useLog=False)
    dataObj.points.permute()
    checkLogContents('points.permute', dataObj.logID)

    # features.permute
    dataObj = nimble.data(data, useLog=False)
    dataObj.features.permute()
    checkLogContents('features.permute', dataObj.logID)

    def noChange(vec):
        return vec

    # features.normalize
    dataObj = nimble.data(data, returnType='Matrix', useLog=False)
    dataObj.features.normalize(noChange)
    checkLogContents('features.normalize', dataObj.logID, {'function': 'noChange'})

    # points.sort
    dataObj = nimble.data(data, featureNames=['a', 'b', 'c'], useLog=False)
    dataObj.points.sort(by=dataObj.features.getName(0))
    checkLogContents('points.sort', dataObj.logID, {'by': dataObj.features.getName(0)})

    # features.sort
    dataObj = nimble.data(data, returnType='Matrix', useLog=False)
    dataObj.features.sort(by=nimble.match.allNumeric, reverse=True)
    checkLogContents('features.sort', dataObj.logID, {'by': 'allNumeric', 'reverse': True})

    # points.copy
    dataObj = nimble.data(data, returnType='Matrix', useLog=False)
    extracted = dataObj.points.copy(0)
    checkLogContents('points.copy', dataObj.logID, {'toCopy': 0})

    # features.copy
    dataObj = nimble.data(data, useLog=False)
    extracted = dataObj.features.copy(number=1)
    checkLogContents('features.copy', dataObj.logID, {'number': 1})

    # points.extract
    dataObj = nimble.data(data, useLog=False)
    extracted = dataObj.points.extract(toExtract=0)
    checkLogContents('points.extract', dataObj.logID, {'toExtract': 0})

    # features.extract
    dataObj = nimble.data(data, useLog=False)
    extracted = dataObj.features.extract(number=1)
    checkLogContents('features.extract', dataObj.logID, {'number': 1})

    # points.delete
    dataObj = nimble.data(data, returnType='Matrix', useLog=False,
                          pointNames=['p' + str(i) for i in range(18)])
    extracted = dataObj.points.delete(start='p0', end='p3')
    checkLogContents('points.delete', dataObj.logID, {'start': 'p0', 'end': 'p3'})

    # features.delete
    dataObj = nimble.data(data, useLog=False)
    extracted = dataObj.features.delete(number=2, randomize=True)
    checkLogContents('features.delete', dataObj.logID, {'number': 2, 'randomize': True})

    def retainer(vector):
        return True

    # points.retain
    dataObj = nimble.data(data, useLog=False)
    extracted = dataObj.points.retain(toRetain=retainer)
    checkLogContents('points.retain', dataObj.logID, {'toRetain': 'retainer'})

    # features.retain
    dataObj = nimble.data(data, returnType='Matrix', useLog=False)
    extracted = dataObj.features.retain(toRetain=lambda ft: True)
    checkLogContents('features.retain', dataObj.logID, {'toRetain': 'lambda ft: True'})

    # points.transform
    dataObj = nimble.data(data, returnType='Matrix', useLog=False)
    dataCopy = dataObj.copy()
    calculated = dataCopy.points.transform(lambda x: [val for val in x])
    checkLogContents('points.transform', dataCopy.logID,
                     {'function': 'lambda x: [val for val in x]'})

    # features.transform
    dataObj = nimble.data(data, returnType='Matrix', useLog=False)
    dataCopy = dataObj.copy()
    calculated = dataCopy.features.transform(lambda x: [val for val in x], features=0)
    checkLogContents('features.transform', dataCopy.logID,
                     {'function': 'lambda x: [val for val in x]', 'features': 0})

    # points.insert
    dataObj = nimble.data(data, useLog=False)
    insertData = [["d", 4, 4], ["d", 4, 4], ["d", 4, 4], ["d", 4, 4], ["d", 4, 4], ["d", 4, 4]]
    toInsert = nimble.data(insertData, useLog=False)
    dataObj.points.insert(0, toInsert)
    checkLogContents('points.insert', dataObj.logID, {'insertBefore': 0,
                                                      'toInsert': toInsert.logID})

    # features.insert
    dataObj = nimble.data(data, name='current', useLog=False)
    insertData = np.zeros((18,1))
    toInsert = nimble.data(insertData, name='insert', useLog=False)
    dataObj.features.insert(0, toInsert)
    checkLogContents('features.insert', "current", {'insertBefore': 0,
                                                    'toInsert': toInsert.name})

    # points.append
    dataObj = nimble.data(data, returnType='Matrix', useLog=False)
    appendData = [["d", 4, 4], ["d", 4, 4], ["d", 4, 4], ["d", 4, 4], ["d", 4, 4], ["d", 4, 4]]
    toAppend = nimble.data(appendData, useLog=False)
    dataObj.points.append(toAppend)
    checkLogContents('points.append', dataObj.logID, {'toAppend': toAppend.logID})

    # features.append
    dataObj = nimble.data(data, returnType='Matrix', useLog=False)
    appendData = np.zeros((18,1))
    toAppend = nimble.data(appendData, name='append', useLog=False)
    dataObj.features.append(toAppend)
    checkLogContents('features.append', dataObj.logID, {'toAppend': toAppend.name})

    # points.fillMatching
    dataObj = nimble.data(data, useLog=False)
    dataObj.points.fillMatching(0, nimble.match.nonNumeric)
    checkLogContents('points.fillMatching', dataObj.logID,
                     {'fillWith': 0, 'matchingElements': 'nonNumeric'})

    # features.fillMatching
    dataObj = nimble.data(data, returnType='Matrix', useLog=False)
    dataObj.features.fillMatching(nimble.fill.mean, 1, features=[1,2])
    checkLogContents('features.fillMatching', dataObj.logID,
                     {'fillWith': 'mean', 'matchingElements': 1})

    # features.splitByParsing
    toSplit = [[1, 'a0', 2], [1, 'a1', 2], [3, 'b0', 4], [5, 'c0', 6]]
    fNames = ['keep1', 'split', 'keep2']
    dataObj = nimble.data(toSplit, featureNames=fNames, returnType='List',
                          useLog=False)
    dataObj.features.splitByParsing('split', 1, ['str', 'int'])
    checkLogContents('features.splitByParsing', dataObj.logID,
                     {'feature': 'split', 'rule': 1, 'resultingNames': ['str', 'int']})

    # points.splitByCollapsingFeatures
    toSplit = [['NYC', 4, 5, 10], ['LA', 20, 21, 21], ['CHI', 0, 2, 7]]
    fNames = ['city', 'jan', 'feb', 'mar']
    dataObj = nimble.data(toSplit, featureNames=fNames, returnType='DataFrame',
                          useLog=False)
    dataObj.points.splitByCollapsingFeatures(['jan', 'feb', 'mar'],
                                              'month', 'temp')
    checkLogContents('points.splitByCollapsingFeatures', dataObj.logID,
                     {'featuresToCollapse': ['jan', 'feb', 'mar'],
                      'featureForNames': 'month', 'featureForValues': 'temp'})

    # points.combineByExpandingFeatures
    toCombine = [['Bolt', '100m', 9.81],
                 ['Bolt', '200m', 19.78],
                 ['Gatlin', '100m', 9.89],
                 ['de Grasse', '200m', 20.02],
                 ['de Grasse', '100m', 9.91]]
    fNames = ['athlete', 'dist', 'time']
    dataObj = nimble.data(toCombine, featureNames=fNames, useLog=False)
    dataObj.points.combineByExpandingFeatures('dist', 'time')
    checkLogContents('points.combineByExpandingFeatures', dataObj.logID,
                     {'featureWithFeatureNames': 'dist', 'featuresWithValues': 'time'})

    # points.setNames
    dataObj = nimble.data(data, returnType='Matrix', useLog=False)
    newPtNames = ['point' + str(i) for i in range(18)]
    dataObj.points.setNames(newPtNames)
    checkLogContents('points.setNames', dataObj.logID, {'assignments': newPtNames})
    dataObj.points.setNames(None)
    checkLogContents('points.setNames', dataObj.logID, {'assignments': None})

    # features.setNames
    dataObj = nimble.data(data, name='toSet', useLog=False)
    newFtNames = ['feature' + str(i) for i in range(3)]
    dataObj.features.setNames(newFtNames)
    checkLogContents('features.setNames', 'toSet', {'assignments': newFtNames})
    dataObj.features.setNames(None)
    checkLogContents('features.setNames', 'toSet', {'assignments': None})

@emptyLogSafetyWrapper
def testDataTypeFunctionsUseLog():
    """Test that the data type functions are being logged"""
    nimble.settings.set('logger', 'enabledByDefault', 'True')
    data = [["a", 1], ["a", 1], ["a", 1], ["a", 1], ["a", 1], ["a", 1],
            ["b", 2], ["b", 2], ["b", 2], ["b", 2], ["b", 2], ["b", 2],
            ["c", 3], ["c", 3], ["c", 3], ["c", 3], ["c", 3], ["c", 3]]

    # features.report
    dataObj = nimble.data(data, useLog=False)
    fReport = dataObj[:,1].features.report()

    logInfo = getLastLogData()
    assert "'reportType': 'feature'" in logInfo

    # report
    dataObj = nimble.data(data, useLog=False)
    sReport = dataObj.report()

    logInfo = getLastLogData()
    assert "'reportType': 'summary'" in logInfo

@emptyLogSafetyWrapper
def testHandmadeLogEntriesInput():
    typeQuery = "SELECT logType FROM logger ORDER BY entry DESC LIMIT 1"
    # custom string
    customString = "enter this string into the log"
    nimble.log("customString", customString)

    logType = nimble.core.logger.active.extractFromLog(typeQuery)[0][0]
    logInfo = getLastLogData()
    assert logType == "User - customString"
    assert customString in logInfo

    # custom list
    customList = ["this", "custom", "list", 1, 2, 3, {"list":"tested"}]
    nimble.log("customList", customList)

    logType = nimble.core.logger.active.extractFromLog(typeQuery)[0][0]
    logInfo = getLastLogData()
    assert logType == "User - customList"
    for value in customList:
        assert str(value) in logInfo

    # custom dict
    customDict = {"custom":"dict", "log":"testing", 1:2, 3:"four"}
    nimble.log("customDict", customDict)

    logType = nimble.core.logger.active.extractFromLog(typeQuery)[0][0]
    logInfo = getLastLogData()
    assert logType == "User - customDict"
    for key in customDict.keys():
        assert str(key) in logInfo
    for value in customDict.values():
        assert str(value) in logInfo

    # heading matches nimble logType
    nimble.log('run', "User log with heading that matches a logType")
    logType = nimble.core.logger.active.extractFromLog(typeQuery)[0][0]
    logInfo = getLastLogData()
    assert logType == "User - run"
    assert "User log with heading that matches a logType" in logInfo

def raisesOSError(*args, **kwargs):
    raise OSError

@emptyLogSafetyWrapper
@patch(inspect, 'getsourcelines', raisesOSError)
def testFailedLambdaStringConversion():
    nimble.settings.set('logger', 'enabledByDefault', 'True')

    data = [["a", 1, 1], ["a", 1, 1], ["a", 1, 1], ["a", 1, 1], ["a", 1, 1], ["a", 1, 1],
            ["b", 2, 2], ["b", 2, 2], ["b", 2, 2], ["b", 2, 2], ["b", 2, 2], ["b", 2, 2],
            ["c", 3, 3], ["c", 3, 3], ["c", 3, 3], ["c", 3, 3], ["c", 3, 3], ["c", 3, 3]]
    dataObj = nimble.data(data, useLog=False)
    calculated = dataObj.calculateOnElements(lambda x: len(x), features=0)
    checkLogContents('calculateOnElements', dataObj.logID,
                     {'toCalculate': "<lambda>", 'features': 0})

@emptyLogSafetyWrapper
def testLambdaStringConversionCommas():
    nimble.settings.set('logger', 'enabledByDefault', 'True')

    data = [["a", 1, 1], ["a", 1, 1], ["a", 1, 1], ["a", 1, 1], ["a", 1, 1], ["a", 1, 1],
            ["b", 2, 2], ["b", 2, 2], ["b", 2, 2], ["b", 2, 2], ["b", 2, 2], ["b", 2, 2],
            ["c", 3, 3], ["c", 3, 3], ["c", 3, 3], ["c", 3, 3], ["c", 3, 3], ["c", 3, 3]]
    for constructor in getDataConstructors():
        dataObj = constructor(data, useLog=False)
        calculated1 = dataObj.points.calculate(lambda x: [x[0], x[2]], points=0)
        checkLogContents('points.calculate', dataObj.logID,
                         {'function': "lambda x: [x[0], x[2]]", 'points': 0})
        calculated2 = dataObj.points.calculate(lambda x: (x[0], x[2]), points=6)
        checkLogContents('points.calculate', dataObj.logID,
                         {'function': "lambda x: (x[0], x[2])", 'points': 6})

@emptyLogSafetyWrapper
@raises(InvalidArgumentType)
def testLogUnacceptedlogType():
    nimble.log(["unacceptable"], "you can't do this")

@emptyLogSafetyWrapper
@raises(InvalidArgumentType)
def testLogUnacceptedlogInfo():
    dataObj = nimble.data([[1]], useLog=False)
    nimble.log("acceptable", dataObj)

@emptyLogSafetyWrapper
@raises(InvalidArgumentValue)
def testLogHeadingTooLong():
    heading = "#" * 51
    nimble.log(heading, 'foo')

##############
### OUTPUT ###
##############

@emptyLogSafetyWrapper
def testShowLogToFile():
    nimble.data([[1], [2], [3]], useLog=True)
    nimble.data([[4, 5], [6, 7], [8, 9]], useLog=True)
    # write to log
    location = nimble.settings.get("logger", "location")
    with tempfile.NamedTemporaryFile() as out:
        pathToFile = out.name
        nimble.showLog(saveToFileName=pathToFile)
        assert os.path.exists(pathToFile)

        originalSize = os.path.getsize(pathToFile)
        removeLogFile()

        # overwrite
        nimble.data([[1], [2], [3]], useLog=True)
        nimble.showLog(saveToFileName=pathToFile)
        overwriteSize = os.path.getsize(pathToFile)
        assert overwriteSize < originalSize
        removeLogFile()

        # append
        nimble.data([[4, 5], [6, 7], [8, 9]], useLog=True)
        nimble.showLog(saveToFileName=pathToFile, append=True)
        appendSize = os.path.getsize(pathToFile)
        # though the information is the same as the original, the appended
        # version will have the information divided under two separate
        # session headings
        assert appendSize > originalSize
        sessionHeadingCount = 0
        with open(pathToFile, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    assert "NIMBLE LOGS" in line
                # no additional log headers should be present when appending
                elif "NIMBLE LOGS" in line:
                    assert False # extra header in log
                if "SESSION" in line:
                    sessionHeadingCount += 1
        assert sessionHeadingCount == 2

@prepopulatedLogSafetyWrapper
def testShowLogToStdOut():
    saved_stdout = sys.stdout
    try:
        location = nimble.settings.get("logger", "location")
        name = "showLogTestFile.txt"
        pathToFile = os.path.join(location, name)
        # create showLog file with default arguments
        nimble.showLog(saveToFileName=pathToFile)

        # get content of file as a string
        with open(pathToFile) as log:
            lines = log.readlines()
        # check log header output
        assert "NIMBLE LOGS" in lines[0]
        # check session header output
        assert lines[1] == "." * 79 + "\n"
        assert "SESSION" in lines[2]
        assert lines[1] == "." * 79 + "\n"

        fileContent = "".join(lines)
        fileContent = fileContent.strip()

        # redirect stdout
        out = StringIO()
        sys.stdout = out

        # showLog to stdout with default arguments
        nimble.showLog()
        stdoutContent = out.getvalue().strip()

        assert stdoutContent == fileContent

    finally:
        sys.stdout = saved_stdout

@emptyLogSafetyWrapper
def testShowLogWithSubobject():
    class Int_(object):
        """
        For the purposes of nimble.KNNClassifier, behaves exactly as the
        integer needed for k , but for the log will cause a failure if
        not represented in arguments as a string.
        """
        def __init__(self, num):
            self.num = num

        def __index__(self):
            return self.num

        def __repr__(self):
            return 'Int_({})'.format(self.num)

    saved_stdout = sys.stdout
    try:
        trainX = [[1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1],
                  [1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1]]
        trainY = [[0], [1], [2], [0], [1], [2], [0], [1], [2], [0], [1], [2]]
        trainXObj = nimble.data(trainX)
        trainYObj = nimble.data(trainY)
        tl = nimble.train('nimble.KNNClassifier', trainXObj, trainYObj,
                          arguments={'k': Int_(1)}, useLog=True)
        # redirect stdout
        out = StringIO()
        sys.stdout = out

        # showLog to stdout with default arguments
        nimble.showLog()
        stdoutContent = out.getvalue().strip()

        assert "Arguments: k=Int_(1)" in stdoutContent

    finally:
        sys.stdout = saved_stdout

@prepopulatedLogSafetyWrapper
def testShowLogSearchFilters():
    """test the level of detail, sessionNumber, date, text, maxEntries search filters"""
    location = nimble.settings.get("logger", "location")
    name = "showLogTestFile.txt"
    pathToFile = os.path.join(location, name)
    nimble.showLog(levelOfDetail=3, leastSessionsAgo=0, mostSessionsAgo=5, maximumEntries=100, saveToFileName=pathToFile)
    fullShowLogSize = os.path.getsize(pathToFile)

    # level of detail
    nimble.showLog(levelOfDetail=3, saveToFileName=pathToFile)
    mostDetailedSize = os.path.getsize(pathToFile)

    nimble.showLog(levelOfDetail=2, saveToFileName=pathToFile)
    lessDetailedSize = os.path.getsize(pathToFile)
    assert lessDetailedSize < mostDetailedSize
    # logs above level 2 should not be present in formatted or default form
    with open(pathToFile) as f:
        output = f.read()
        assert 'runCV' not in output
        assert 'KFoldCrossValidation' not in output

    nimble.showLog(levelOfDetail=1, saveToFileName=pathToFile)
    leastDetailedSize = os.path.getsize(pathToFile)
    assert leastDetailedSize < lessDetailedSize
    # logs above level 1 should not be present in formatted or default form
    with open(pathToFile) as f:
        output = f.read()
        assert 'prep' not in output
        assert 'features.extract ' not in output
        assert 'runCV' not in output
        assert 'KFoldCrossValidation' not in output
        assert 'crossVal' not in output
        assert 'Cross Validating' not in output
        assert 'run' not in output
        assert 'trainAndTest' not in output

    # sessionNumber
    nimble.showLog(levelOfDetail=3, mostSessionsAgo=4, saveToFileName=pathToFile)
    fewerSessionsAgoSize = os.path.getsize(pathToFile)
    assert fewerSessionsAgoSize < fullShowLogSize

    nimble.showLog(levelOfDetail=3, leastSessionsAgo=1, mostSessionsAgo=5, saveToFileName=pathToFile)
    moreSessionsAgoSize = os.path.getsize(pathToFile)
    assert moreSessionsAgoSize < fullShowLogSize
    nimble.showLog(levelOfDetail=3, leastSessionsAgo=1, mostSessionsAgo=5)
    assert moreSessionsAgoSize == fewerSessionsAgoSize

    nimble.showLog(levelOfDetail=3, leastSessionsAgo=2, mostSessionsAgo=4, saveToFileName=pathToFile)
    sessionSelectionSize = os.path.getsize(pathToFile)
    assert sessionSelectionSize < moreSessionsAgoSize

    # startDate
    nimble.showLog(levelOfDetail=3, mostSessionsAgo=5, startDate="2018-03-23", saveToFileName=pathToFile)
    startLaterSize = os.path.getsize(pathToFile)
    assert startLaterSize < fullShowLogSize

    nimble.showLog(levelOfDetail=3, mostSessionsAgo=5, startDate="2018-04-24", saveToFileName=pathToFile)
    startLastSize = os.path.getsize(pathToFile)
    assert startLastSize < startLaterSize

    # endDate
    nimble.showLog(levelOfDetail=3, mostSessionsAgo=5, endDate="2018-03-25", saveToFileName=pathToFile)
    endEarlierSize = os.path.getsize(pathToFile)
    assert endEarlierSize < fullShowLogSize

    nimble.showLog(levelOfDetail=3, mostSessionsAgo=5, endDate="2018-03-22", saveToFileName=pathToFile)
    endEarliestSize = os.path.getsize(pathToFile)
    assert endEarliestSize < endEarlierSize

    # startDate and endDate
    nimble.showLog(levelOfDetail=3, mostSessionsAgo=5, startDate="2018-03-23", endDate="2018-03-25", saveToFileName=pathToFile)
    dateSelectionSize = os.path.getsize(pathToFile)
    assert dateSelectionSize < startLaterSize
    assert dateSelectionSize < endEarlierSize

    # startDate and endDate with time
    nimble.showLog(levelOfDetail=3, mostSessionsAgo=5, startDate="2018-03-23 11:00", endDate="2018-03-23 17:00:00", saveToFileName=pathToFile)
    timeSelectionSize = os.path.getsize(pathToFile)
    assert timeSelectionSize < dateSelectionSize

    #text
    nimble.showLog(levelOfDetail=3, mostSessionsAgo=1, searchForText=None, saveToFileName=pathToFile)
    oneSessionSize = os.path.getsize(pathToFile)

    nimble.showLog(levelOfDetail=3, mostSessionsAgo=1, searchForText="trainAndTest", saveToFileName=pathToFile)
    trainSearchSize = os.path.getsize(pathToFile)
    assert trainSearchSize < oneSessionSize

    nimble.showLog(levelOfDetail=3, mostSessionsAgo=1, searchForText="KNNClassifier", saveToFileName=pathToFile)
    loadSearchSize = os.path.getsize(pathToFile)
    assert loadSearchSize < oneSessionSize

    # regex
    nimble.showLog(levelOfDetail=3, mostSessionsAgo=1, searchForText="KNN.+fier", regex=True, saveToFileName=pathToFile)
    loadRegexSize = os.path.getsize(pathToFile)
    assert loadSearchSize == loadRegexSize

    # maximumEntries
    nimble.showLog(levelOfDetail=3, mostSessionsAgo=5, maximumEntries=36, saveToFileName=pathToFile)
    size36Entries = os.path.getsize(pathToFile)
    assert size36Entries < fullShowLogSize

    nimble.showLog(levelOfDetail=3, mostSessionsAgo=5, maximumEntries=35, saveToFileName=pathToFile)
    size35Entries = os.path.getsize(pathToFile)
    assert size35Entries < size36Entries

    nimble.showLog(levelOfDetail=3, mostSessionsAgo=5, maximumEntries=18, saveToFileName=pathToFile)
    maxEntriesOneSession = os.path.getsize(pathToFile)
    assert maxEntriesOneSession == oneSessionSize

    # showLog returns None, file still created with only header
    nimble.showLog(levelOfDetail=3, maximumEntries=1, saveToFileName=pathToFile)
    oneEntrySize = os.path.getsize(pathToFile)
    # pick startDate after final date in log
    nimble.showLog(levelOfDetail=3, startDate="2018-05-24", saveToFileName=pathToFile)
    noDataSize = os.path.getsize(pathToFile)
    assert noDataSize < oneEntrySize

@emptyLogSafetyWrapper
@raises(InvalidArgumentValue)
def testLevelOfDetailNotInRange():
    nimble.showLog(levelOfDetail=6)

@emptyLogSafetyWrapper
@raises(InvalidArgumentValueCombination)
def testStartGreaterThanEndDate():
    nimble.showLog(startDate="2018-03-24", endDate="2018-03-22")

@emptyLogSafetyWrapper
def testInvalidDateTimeFormats():
    # year invalid format
    with raises(InvalidArgumentValue):
        nimble.showLog(startDate="18-03-24")
    # month invalid format
    with raises(InvalidArgumentValue):
        nimble.showLog(startDate="2018-3-24")
    # day invalid format
    with raises(InvalidArgumentValue):
        nimble.showLog(startDate="2018-04-1")
    # date format ok but invalid date
    with raises(InvalidArgumentValue):
        nimble.showLog(startDate="2018-02-31")
    # hour invalid format
    with raises(InvalidArgumentValue):
        nimble.showLog(startDate="2018-03-24 1:00")
    # minute invalid format
    with raises(InvalidArgumentValue):
        nimble.showLog(startDate="2018-03-24 01:19.22")
    # second invalid
    with raises(InvalidArgumentValue):
        nimble.showLog(startDate="2018-03-24 01:19:0.2")

@emptyLogSafetyWrapper
@raises(InvalidArgumentValue)
def testLeastSessionsAgoNegative():
    nimble.showLog(leastSessionsAgo=-2)

@emptyLogSafetyWrapper
@raises(InvalidArgumentValueCombination)
def testMostSessionsLessThanLeastSessions():
    nimble.showLog(leastSessionsAgo=2, mostSessionsAgo=1)

@emptyLogSafetyWrapper
def testShowLogSuccessWithUserLog():
    """ Test user headings that match defined logTypes are successfully rendered """
    for lType in nimble.core.logger.active.logTypes:
        nimble.log(lType, "foo")
    # defined logTypes require specific input to render correctly, if these
    # headings are stored as a defined logType, the log would not render
    # nimble.log should prepend "User - " to the heading to avoid this conflict.
    nimble.showLog()
