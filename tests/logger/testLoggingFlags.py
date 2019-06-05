"""
Group of tests which checks that use controlled local and global
mechanisms for controlling logging are functioning as expected.
"""

from __future__ import absolute_import
import os
import tempfile

from nose.plugins.attrib import attr
import numpy

import nimble
from nimble.helpers import generateClassificationData
from nimble.calculate import fractionIncorrect
from nimble.configuration import configSafetyWrapper

learnerName = 'custom.KNNClassifier'

def logEntryCount(logger):
    entryCount = logger.extractFromLog("SELECT COUNT(entry) FROM logger;")
    return entryCount[0][0]

@configSafetyWrapper
def back_load(toCall, *args, **kwargs):
    logger = nimble.logger.active

    # count number of starting log entries
    nimble.settings.set('logger', 'enabledByDefault', 'True')

    start, end = loadAndCheck(toCall, True, *args)
    assert start + 1 == end

    start, end = loadAndCheck(toCall, None, *args)
    assert start + 1 == end

    start, end = loadAndCheck(toCall, False, *args)
    assert start == end

    nimble.settings.set('logger', 'enabledByDefault', 'False')

    start, end = loadAndCheck(toCall, True, *args)
    assert start + 1 == end

    start, end = loadAndCheck(toCall, None, *args)
    assert start == end

    start, end = loadAndCheck(toCall, False, *args)
    assert start == end

def loadAndCheck(toCall, useLog, *args):
    logger = nimble.logger.active
    # count number of starting log entries
    startCount = logEntryCount(logger)
    # call the function we're testing for log control
    toCall(*args, useLog=useLog)
    # make sure it has the expected effect on the count
    endCount = logEntryCount(logger)
    return (startCount, endCount)

def test_createData():
    back_load(nimble.createData, 'Matrix', [[1, 2, 3], [4, 5, 6]])

def test_createRandomData():
    back_load(nimble.createRandomData, 'Sparse', 5, 5, 0.99)

def test_loadData():
    obj = nimble.createData('Matrix', [[1, 2, 3], [4, 5, 6]], useLog=False)
    with tempfile.NamedTemporaryFile(suffix='.nimd') as tmpFile:
        obj.save(tmpFile.name)
        back_load(nimble.loadData, tmpFile.name)

def test_loadTrainedLearner():
    train = nimble.createData('Matrix', [[0, 0, 1], [0, 1, 0], [1, 0, 0]], useLog=False)
    test = nimble.createData('Matrix', [[3], [2], [1]], useLog=False)
    tl = nimble.train('custom.KNNClassifier', train, test)
    with tempfile.NamedTemporaryFile(suffix='.nimm') as tmpFile:
        tl.save(tmpFile.name)
        back_load(nimble.loadTrainedLearner, tmpFile.name)

def test_setRandomSeed():
    nimble.randomness.startAlternateControl()
    back_load(nimble.setRandomSeed, 1337)
    nimble.randomness.endAlternateControl()

# helper function which checks log status for runs
def runAndCheck(toCall, useLog):
    # generate data
    cData = generateClassificationData(2, 10, 2)
    ((trainX, trainY), (testX, testY)) = cData
    logger = nimble.logger.active
    # count number of starting log entries
    startCount = logEntryCount(logger)

    # call the function we're testing for log control
    toCall(trainX, trainY, testX, testY, useLog)
    # make sure it has the expected effect on the count
    endCount = logEntryCount(logger)

    return (startCount, endCount)

@configSafetyWrapper
def backend(toCall, validator):
    # for each combination of local and global, call and check

    nimble.settings.set('logger', 'enabledByDefault', 'True')

    (start, end) = validator(toCall, useLog=True)
    assert start + 1 == end

    (start, end) = validator(toCall, useLog=None)
    assert start + 1 == end

    (start, end) = validator(toCall, useLog=False)
    assert start == end

    nimble.settings.set('logger', 'enabledByDefault', 'False')

    (start, end) = validator(toCall, useLog=True)
    assert start + 1 == end

    (start, end) = validator(toCall, useLog=None)
    assert start == end

    (start, end) = validator(toCall, useLog=False)
    assert start == end

def test_train():
    def wrapped(trainX, trainY, testX, testY, useLog):
        return nimble.train(learnerName, trainX, trainY, useLog=useLog)

    backend(wrapped, runAndCheck)

def test_trainAndApply():
    def wrapped(trainX, trainY, testX, testY, useLog):
        return nimble.trainAndApply(learnerName, trainX, trainY, testX, useLog=useLog)

    backend(wrapped, runAndCheck)

def test_trainAndTest():
    def wrapped(trainX, trainY, testX, testY, useLog):
        return nimble.trainAndTest(
            learnerName, trainX, trainY, testX, testY,
            performanceFunction=fractionIncorrect, useLog=useLog)

    backend(wrapped, runAndCheck)

def test_trainAndTestOnTrainingData_trainError():
    def wrapped(trainX, trainY, testX, testY, useLog):
        return nimble.trainAndTestOnTrainingData(
            learnerName, trainX, trainY, performanceFunction=fractionIncorrect,
            crossValidationError=False, useLog=useLog)

    backend(wrapped, runAndCheck)

def test_normalizeData():
    def wrapped(trainX, trainY, testX, testY, useLog):
        return nimble.normalizeData('mlpy.PCA', trainX, testX=testX,
                                  arguments={'k': 1}, useLog=useLog)

    backend(wrapped, runAndCheck)

#def test_trainAndTestOvO():
#	def wrapped(trainX, trainY, testX, testY, useLog):
#		return nimble.helpers.trainAndTestOneVsOne(learnerName, trainX, trainY, testX, testY, performanceFunction=fractionIncorrect, useLog=useLog)

#	backend(wrapped, runAndCheck)

#def test_trainAndTestOvA():
#	def wrapped(trainX, trainY, testX, testY, useLog):
#		return nimble.helpers.trainAndTestOneVsAll(learnerName, trainX, trainY, testX, testY, performanceFunction=fractionIncorrect, useLog=useLog)

#	backend(wrapped, runAndCheck)

def test_TrainedLearner_apply():
    cData = generateClassificationData(2, 10, 2)
    ((trainX, trainY), (testX, testY)) = cData
    # get a trained learner
    tl = nimble.train(learnerName, trainX, trainY, useLog=False)

    def wrapped(trainX, trainY, testX, testY, useLog):
        return tl.apply(testX, useLog=useLog)

    backend(wrapped, runAndCheck)

def test_TrainedLearner_test():
    cData = generateClassificationData(2, 10, 2)
    ((trainX, trainY), (testX, testY)) = cData
    # get a trained learner
    tl = nimble.train(learnerName, trainX, trainY, useLog=False)

    def wrapped(trainX, trainY, testX, testY, useLog):
        return tl.test(testX, testY, performanceFunction=fractionIncorrect,
                       useLog=useLog)

    backend(wrapped, runAndCheck)

@configSafetyWrapper
def backendDeep(toCall, validator):
    if toCall.__name__.startswith("crossValidate"):
        expectedLogChangeTrue = 1
        expectedLogChangeFalse = 0
    elif toCall.__name__.startswith("train"):
        expectedLogChangeTrue = 2 # cross val entry and train entry
        expectedLogChangeFalse = 1 # only train entry
    else:
        msg = "The function name for this test is not recognized. "
        msg += "Functions using this backend must have the wrapped 'toCall' "
        msg += "function renamed to the tested function so it can be "
        msg += "determined how many log entries should be added"
        raise TypeError(msg)
    nimble.settings.set('logger', 'enabledByDefault', 'True')
    nimble.settings.set('logger', 'enableCrossValidationDeepLogging', 'True')

    # the deep logging flag is continget on global and local
    # control, so we confirm that in those instances where
    # logging should be disable, it is still disabled
    (startT1, endT1) = validator(toCall, useLog=True)
    (startT2, endT2) = validator(toCall, useLog=None)
    (startT3, endT3) = validator(toCall, useLog=False) # 0 logs added
    assert startT1 + expectedLogChangeTrue == endT1
    assert startT2 + expectedLogChangeTrue == endT2
    assert startT3 == endT3

    nimble.settings.set('logger', 'enableCrossValidationDeepLogging', 'False')

    (startF1, endF1) = validator(toCall, useLog=True)
    (startF2, endF2) = validator(toCall, useLog=None)
    (startF3, endF3) = validator(toCall, useLog=False) # 0 logs added
    assert startF1 + expectedLogChangeFalse == endF1
    assert startF2 + expectedLogChangeFalse == endF2
    assert startF3 == endF3

    # next we compare the differences between the calls when
    # the deep flag is different
    assert (endT1 - startT1) - 1 == (endF1 - startF1)
    assert (endT2 - startT2) - 1 == (endF2 - startF2)

    nimble.settings.set('logger', 'enabledByDefault', 'False')
    nimble.settings.set('logger', 'enableCrossValidationDeepLogging', 'True')

    # the deep logging flag is contingent on global and local
    # control, so we confirm that logging is called or
    # not appropriately
    (startT1, endT1) = validator(toCall, useLog=True) # 2 logs added
    (startT2, endT2) = validator(toCall, useLog=None) # 0 logs added
    assert startT2 == endT2
    (startT3, endT3) = validator(toCall, useLog=False) # 0 logs added
    assert startT3 == endT3

    nimble.settings.set('logger', 'enableCrossValidationDeepLogging', 'False')

    (startF1, endF1) = validator(toCall, useLog=True) # 1 logs added
    (startF2, endF2) = validator(toCall, useLog=None) # 0 logs added
    assert startF2 == endF2
    (startF3, endF3) = validator(toCall, useLog=False) # 0 logs added
    assert startF3 == endF3

    # next we compare the differences between the calls when
    # the deep flag is different
    assert (endT1 - startT1) - 1 == (endF1 - startF1)

def test_Deep_crossValidate():
    def wrapped(trainX, trainY, testX, testY, useLog):
        return nimble.crossValidate(learnerName, trainX, trainY,
                                 performanceFunction=fractionIncorrect,
                                 useLog=useLog)
    wrapped.__name__ = 'crossValidate'
    backendDeep(wrapped, runAndCheck)

def test_Deep_train():
    def wrapped(trainX, trainY, testX, testY, useLog):
        k = nimble.CV([2, 3])  # we are not calling CV directly, we need to trigger it
        return nimble.train(learnerName, trainX, trainY,
                         performanceFunction=fractionIncorrect, useLog=useLog,
                         k=k)
    wrapped.__name__ = 'train'
    backendDeep(wrapped, runAndCheck)

def test_Deep_trainAndApply():
    def wrapped(trainX, trainY, testX, testY, useLog):
        k = nimble.CV([2, 3])  # we are not calling CV directly, we need to trigger it
        return nimble.trainAndApply(learnerName, trainX, trainY, testX,
                                 performanceFunction=fractionIncorrect,
                                 useLog=useLog, k=k)
    wrapped.__name__ = 'trainAndApply'
    backendDeep(wrapped, runAndCheck)

def test_Deep_trainAndTest():
    def wrapped(trainX, trainY, testX, testY, useLog):
        k = nimble.CV([2, 3])  # we are not calling CV directly, we need to trigger it
        return nimble.trainAndTest(learnerName, trainX, trainY, testX, testY,
                                performanceFunction=fractionIncorrect,
                                useLog=useLog, k=k)
    wrapped.__name__ = 'trainAndTest'
    backendDeep(wrapped, runAndCheck)

def test_Deep_trainAndTestOnTrainingData_CVError():
    def wrapped(trainX, trainY, testX, testY, useLog):
        return nimble.trainAndTestOnTrainingData(
            learnerName, trainX, trainY, performanceFunction=fractionIncorrect,
            crossValidationError=True, useLog=useLog)
    wrapped.__name__ = 'trainAndTestOnTrainingData'
    backendDeep(wrapped, runAndCheck)

def prepAndCheck(toCall, useLog):
    data = [["a", 1, 1], ["a", 1, 1], ["a", 1, 1], ["a", 1, 1], ["a", 1, 1], ["a", 1, 1],
            ["b", 2, 2], ["b", 2, 2], ["b", 2, 2], ["b", 2, 2], ["b", 2, 2], ["b", 2, 2],
            ["c", 3, 3], ["c", 3, 3], ["c", 3, 3], ["c", 3, 3], ["c", 3, 3], ["c", 3, 3]]
    pNames = ['p' + str(i) for i in range(18)]
    fNames = ['f0', 'f1', 'f2']
    # createData not logged
    dataObj = nimble.createData("Matrix", data, pointNames=pNames,
                             featureNames=fNames, useLog=False)

    logger = nimble.logger.active
    # count number of starting log entries
    startCount = logEntryCount(logger)

    toCall(dataObj, useLog)
    # make sure it has the expected effect on the count
    endCount = logEntryCount(logger)

    return (startCount, endCount)

########
# Base #
########

def test_replaceFeatureWithBinaryFeatures():
    def wrapped(obj, useLog):
        return obj.replaceFeatureWithBinaryFeatures(0, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_transformFeatureToIntegers():
    def wrapped(obj, useLog):
        return obj.transformFeatureToIntegers(0, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_trainAndTestSets():
    def wrapped(obj, useLog):
        return obj.trainAndTestSets(testFraction=0.5, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_groupByFeature():
    def wrapped(obj, useLog):
        return obj.groupByFeature(by=0, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_referenceDataFrom():
    def wrapped(obj, useLog):
        obj.referenceDataFrom(obj, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_transpose():
    def wrapped(obj, useLog):
        obj.transpose(useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_fillWith():
    def wrapped(obj, useLog):
        obj.fillWith(1, 2, 0, 4, 0, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_fillUsingAllData():
    def simpleFiller(obj, match):
        return nimble.createData('Matrix', numpy.zeros((18, 3)), useLog=False)
    def wrapped(obj, useLog):
        obj.fillUsingAllData('a', fill=simpleFiller, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_featureReport():
    def wrapped(obj, useLog):
        obj[:, 1].featureReport(useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_summaryReport():
    def wrapped(obj, useLog):
        obj.summaryReport(useLog=useLog)

    backend(wrapped, prepAndCheck)

@configSafetyWrapper
def flattenUnflattenBackend(toCall, validator):
    # for each combination of local and global, call and check

    nimble.settings.set('logger', 'enabledByDefault', 'True')

    (start, end) = validator(toCall, useLog=True)
    assert start + 2 == end

    (start, end) = validator(toCall, useLog=None)
    assert start + 2 == end

    (start, end) = validator(toCall, useLog=False)
    assert start == end

    nimble.settings.set('logger', 'enabledByDefault', 'False')

    (start, end) = validator(toCall, useLog=True)
    assert start + 2 == end

    (start, end) = validator(toCall, useLog=None)
    assert start == end

    (start, end) = validator(toCall, useLog=False)
    assert start == end

def test_flattenUnflatten_pointAxis():
    def wrapped_Flatten_UnFlatten(obj, useLog):
        obj.flattenToOnePoint(useLog=useLog)
        obj.unflattenFromOnePoint(18, useLog=useLog)

    flattenUnflattenBackend(wrapped_Flatten_UnFlatten, prepAndCheck)

def test_flattenUnflatten_featureAxis():
    def wrapped_Flatten_UnFlatten(obj, useLog):
        obj.flattenToOneFeature(useLog=useLog)
        obj.unflattenFromOneFeature(3, useLog=useLog)

    flattenUnflattenBackend(wrapped_Flatten_UnFlatten, prepAndCheck)

def test_merge():
    mData = [[1, 4], [2, 5], [3, 6]]
    mPtNames = ['p0', 'p6', 'p12']
    mFtNames = ['f2', 'f3']
    mergeObj = nimble.createData('Matrix', mData, pointNames=mPtNames,
                              featureNames=mFtNames)
    def wrapped(obj, useLog):
        obj.merge(mergeObj, point='intersection', feature='union', useLog=useLog)

    backend(wrapped, prepAndCheck)

############################
# Points/Features/Elements #
############################

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

def test_point_mapReduce():
    def wrapped(obj, useLog):
        return obj.points.mapReduce(simpleMapper, simpleReducer, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_features_mapReduce():
    def wrapped(obj, useLog):
        # transpose data to make use of same mapper and reducer
        obj.transpose(useLog=False)
        return obj.features.mapReduce(simpleMapper, simpleReducer, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_elements_calculate():
    def wrapped(obj, useLog):
        return obj.elements.calculate(lambda x: len(x), features=0, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_points_calculate():
    def wrapped(obj, useLog):
        return obj.points.calculate(lambda x: len(x), useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_features_calculate():
    def wrapped(obj, useLog):
        return obj.features.calculate(lambda x: len(x), features=0, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_points_shuffle():
    def wrapped(obj, useLog):
        return obj.points.shuffle(useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_features_shuffle():
    def wrapped(obj, useLog):
        return obj.features.shuffle(useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_points_normalize():
    def wrapped(obj, useLog):
        return obj.points.normalize(subtract=0, divide=1, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_features_normalize():
    def wrapped(obj, useLog):
        return obj.features.normalize(subtract=0, divide=1, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_points_sort():
    def wrapped(obj, useLog):
        return obj.points.sort(sortBy="f0", useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_features_sort():
    def wrapped(obj, useLog):
        return obj.features.sort(sortBy="p0", useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_points_extract():
    def wrapped(obj, useLog):
        return obj.points.extract(toExtract=0, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_features_extract():
    def wrapped(obj, useLog):
        return obj.features.extract(toExtract=0, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_points_delete():
    def wrapped(obj, useLog):
        return obj.points.delete(toDelete=0, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_features_delete():
    def wrapped(obj, useLog):
        return obj.features.delete(toDelete=0, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_points_retain():
    def wrapped(obj, useLog):
        return obj.points.retain(toRetain=0, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_features_retain():
    def wrapped(obj, useLog):
        return obj.features.retain(toRetain=0, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_points_copy():
    def wrapped(obj, useLog):
        return obj.points.copy(toCopy=0, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_features_copy():
    def wrapped(obj, useLog):
        return obj.features.copy(toCopy=0, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_points_fill():
    def wrapped(obj, useLog):
        return obj.points.fill(match=1, fill=11, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_features_fill():
    def wrapped(obj, useLog):
        return obj.features.fill(match=1, fill=11, useLog=useLog)

    backend(wrapped, prepAndCheck)


def test_points_transform():
    def wrapped(obj, useLog):
        return obj.points.transform(lambda x: [point for point in x], useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_features_transform():
    def wrapped(obj, useLog):
        return obj.features.transform(lambda x: [point for point in x], features=0, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_elements_transform():
    def wrapped(obj, useLog):
        return obj.elements.transform(lambda x: [point for point in x], features=0, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_points_add():
    def wrapped(obj, useLog):
        appendData = [["d", 4, 4], ["d", 4, 4], ["d", 4, 4], ["d", 4, 4], ["d", 4, 4], ["d", 4, 4]]
        toAppend = nimble.createData("Matrix", appendData, useLog=False)
        return obj.points.add(toAppend, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_features_add():
    def wrapped(obj, useLog):
        appendData = numpy.zeros((18,1))
        toAppend = nimble.createData("Matrix", appendData, useLog=False)
        return obj.features.add(toAppend, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_features_splitByParsing():
    def customParser(val):
        if val == 1:
            return ['a', 1]
        elif val == 2:
            return ['b', 2]
        else:
            return ['c', 3]
    def wrapped(obj, useLog):
        return obj.features.splitByParsing(1, customParser, ['str', 'int'], useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_points_splitByCollapsingFeatures():
    def wrapped(obj, useLog):
        return obj.points.splitByCollapsingFeatures(['f0', 'f1', 'f2'],
                                                    'featureNames', 'values',
                                                    useLog = useLog)
    backend(wrapped, prepAndCheck)

def test_points_combineByExpandingFeatures():
    def wrapped(obj, useLog):
        newData = [['Bolt', '100m', 9.81],
                   ['Bolt', '200m', 19.78],
                   ['Gatlin', '100m', 9.89],
                   ['de Grasse', '200m', 20.02],
                   ['de Grasse', '100m', 9.91]]
        fNames = ['athlete', 'dist', 'time']
        newObj = nimble.createData('Matrix', newData, featureNames=fNames, useLog=False)
        return newObj.points.combineByExpandingFeatures('dist', 'time', useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_points_setName():
    def wrapped(obj, useLog):
        return obj.points.setName(0, 'newPointName', useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_features_setName():
    def wrapped(obj, useLog):
        return obj.features.setName(0, 'newFeatureName', useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_points_setNames():
    def wrapped(obj, useLog):
        newNames = ['new_pt' + str(i) for i in range(18)]
        return obj.points.setNames(newNames, useLog=useLog)

    backend(wrapped, prepAndCheck)

def test_features_setNames():
    def wrapped(obj, useLog):
        newNames = ['new_ft' + str(i) for i in range(3)]
        return obj.features.setNames(newNames, useLog=useLog)

    backend(wrapped, prepAndCheck)
