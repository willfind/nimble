
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
Helper functions for any functions and objects defined in learn.py.

They are separated here so that that (most) top level user-facing
functions are contained in learn.py without the distraction of helpers.
"""

from functools import wraps
import numbers
import math

import numpy as np

import nimble
from nimble.exceptions import InvalidArgumentValue, InvalidArgumentType
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.exceptions import PackageException
from nimble.core.data import Base
from nimble.random import pythonRandom
from nimble.core.configuration import setInterfaceOptions


def findBestInterface(package):
    """
    Attempt to determine the interface.

    Takes the string name of a possible interface provided to some other
    function by a nimble user, and attempts to find the interface which
    best matches that name amoung those available. If it does not match
    any available interfaces, then an exception is thrown.
    """
    if package in nimble.core.interfaces.available:
        # canonical name provided and available
        return nimble.core.interfaces.available[package]
    for interface in nimble.core.interfaces.available.values():
        if interface.isAlias(package):
            return interface
    for interface in nimble.core.interfaces.predefined:
        if interface.isAlias(package):
            try:
                interfaceObj = interface()
            except Exception as e: # pylint: disable=broad-except
                if isinstance(e, PackageException):
                    raise # interface version does not meet requirements
                # interface is a predefined one, but instantiation failed
                return interface.provideInitExceptionInfo()
            # add successful instantiations to interfaces.available and
            # set options in config
            interfaceName = interfaceObj.getCanonicalName()
            nimble.core.interfaces.available[interfaceName] = interfaceObj
            setInterfaceOptions(interfaceObj, False)
            return interfaceObj
    # if package is not recognized, provide generic exception information
    msg = "package '" + package
    msg += "' was not associated with any of the available package interfaces"
    raise InvalidArgumentValue(msg)


def initAvailablePredefinedInterfaces():
    """
    Instantiate all available predefined interfaces for functions that
    require access to all available.
    """
    for interface in nimble.core.interfaces.predefined:
        canonicalName = interface.getCanonicalName()
        if canonicalName not in nimble.core.interfaces.available:
            try:
                interfaceObj = interface()
                interfaceName = interfaceObj.getCanonicalName()
                nimble.core.interfaces.available[interfaceName] = interfaceObj
                setInterfaceOptions(interfaceObj, False)
            except Exception: # pylint: disable=broad-except
                # if fails for any reason, it's not available
                pass


def _learnerQuery(name, queryType):
    """
    Takes a string of the form 'package.learnerName' and a string
    defining a queryType of either 'parameters' or 'defaults' then
    returns the results of either the package's
    getParameters(learnerName) function or the package's
    getDefaultValues(learnerName) function.
    """
    interface, learnerName = _unpackLearnerName(name)

    if queryType == "parameters":
        toCallName = 'getLearnerParameterNames'
    elif queryType == 'defaults':
        toCallName = 'getLearnerDefaultValues'
    else:
        raise InvalidArgumentValue("Unrecognized queryType: " + queryType)

    ret = getattr(interface, toCallName)(learnerName)

    if len(ret) == 1:
        return ret[0]
    return ret


def computeMetrics(dependentVar, knownData, predictedData,
                   performanceFunction):
    """
    Calculate the performance of the learner.

    Using the provided metric, compare the known data or labels to the
    predicted data or labels and calculate the performance of the
    learner which produced the predicted data.

    Parameters
    ----------
    dependentVar : indentifier, list, nimble Base object
        Indicate the feature names or indices in knownData containing
        the known labels, or a data object that contains the known
        labels.
    knownData : nimble Base object
        Data object containing the known labels of the training set, as
        well as the features of the training set. Can be None if
        'dependentVar' is an object containing the labels.
    predictedData : nimble Base object
        Data object containing predicted labels/data. Assumes that the
        predicted label (or labels) in the nth row of predictedLabels
        is associated with the same data point/instance as the label in
        the nth row of knownLabels.
    performanceFunction : function
        A python function that returns a single numeric value evaluating
        performance. The function must take either two or three args.
        In the two arg case, they must be two sets of data or labels to
        be compared. In the three arg case, the first two args are the
        same as in the two arg case, and the third arg must take the
        value of what is to be considered the negative label in this
        binary classification problem. See nimble.calculate for a number
        of builtin options.

    Returns
    -------
    Value
        Measurement of the performance of the learner that produced the
        given data.
    """
    if dependentVar is None or isinstance(dependentVar, Base):
        #The known Indicator argument already contains all known
        #labels, so we do not need to do any further processing
        knownLabels = dependentVar
    else:
        #known Indicator is a feature ID or group of IDs; we extract the
        # columns it indicates from knownValues
        knownLabels = knownData.features.copy(dependentVar, useLog=False)

    result = performanceFunction(knownLabels, predictedData)

    return result

# with class-based refactor:
# todo add scale control as paramater for generateClusteredPoints
#  - remember to scale noise term accordingly
def generateClusteredPoints(numClusters, numPointsPerCluster,
                            numFeaturesPerPoint, addFeatureNoise=True,
                            addLabelNoise=True, addLabelColumn=False,
                            returnType='Matrix'):
    """
    Function to generate Data object with arbitrary number of points,
    number of clusters, and number of features.

    The function returns the dataset in an object, 'labels' for each
    point in the dataset (noise optional), and the 'noiseless' labels
    for the points, which is the central value used to define the
    feature values for each point.

    generateClusteredPoints() outputs a dataset of the following format:
    the value of each entry is either 0 or (+/-)1+noise. The
    features of the generated object are partitioned so that every cluster
    is represented in at least one of the returned features, with the even
    numbered clusters present with a positive value and the odd numbered
    clusters present with a negative value.

    NOTE: if addFeatureNoise and addLabelNoise are false, then the
    'clusters' are actually all contain just repeated points, where each
    point in the cluster has the same features and the same labels.

    Returns
    -------
    tuple of nimble.Base objects:
    (pointsObj, labelsObj, noiselessLabelsObj)
    Examples
    --------
    >>> ret = generateClusteredPoints(numClusters=5, numPointsPerCluster=1,
    ...       numFeaturesPerPoint=6, addFeatureNoise=False,
    ...       addLabelNoise=False, addLabelColumn=False)
    >>> ret[0]
    <Matrix 5pt x 6ft
           0       1       2       3       4      5
       ┌─────────────────────────────────────────────
     0 │  0.000   0.000   0.000   0.000  0.000  0.000
     1 │ -1.000  -1.000   0.000   0.000  0.000  0.000
     2 │  0.000   0.000   2.000   2.000  0.000  0.000
     3 │  0.000   0.000  -3.000  -3.000  0.000  0.000
     4 │  0.000   0.000   0.000   0.000  4.000  4.000
    >

    """
    if numFeaturesPerPoint < numClusters/2:
        msg = "There must be at least as many features as possible cluster "
        msg += "labels divided by 2"
        raise InvalidArgumentValue(msg)

    pointsList = []
    labelsList = []
    clusterNoiselessLabelList = []

    def _noiseTerm():
        return pythonRandom.random() * 0.0001 - 0.00005

    for curr in range(numClusters):
        for _ in range(numPointsPerCluster):
            ftsPerPt = numFeaturesPerPoint
            ftsPerCluster = int(math.ceil(ftsPerPt / math.ceil(numClusters/2)))
            val = curr if curr % 2 == 0 else -curr
            featureVector = [val if curr//2 == x//ftsPerCluster else 0
                             for x in range(ftsPerPt)]

            if addFeatureNoise:
                featureVector = [_noiseTerm() + entry for entry
                                    in featureVector]

            if addLabelNoise:
                curLabel = _noiseTerm() + curr
            else:
                curLabel = curr

            if addLabelColumn:
                featureVector.append(curLabel)

            #append curLabel as a list to maintain dimensionality
            labelsList.append([curLabel])

            pointsList.append(featureVector)
            clusterNoiselessLabelList.append([float(curr)])

    pointsArray = np.array(pointsList, dtype=np.float64)
    labelsArray = np.array(labelsList, dtype=np.float64)
    clusterNoiselessLabelArray = np.array(clusterNoiselessLabelList,
                                          dtype=np.float64)
    # todo verify that your list of lists is valid initializer for all
    # datatypes, not just matrix
    # then convert
    # finally make matrix object out of the list of points w/ labels in last
    # column of each vector/entry:
    pointsObj = nimble.data(pointsArray, useLog=False)

    labelsObj = nimble.data(labelsArray, useLog=False)

    # todo change actuallavels to something like associatedClusterCentroid
    noiselessLabelsObj = nimble.data(clusterNoiselessLabelArray,
                                     useLog=False)

    # convert datatype if not matrix
    if returnType.lower() != 'matrix':
        pointsObj = pointsObj.copy(to=returnType)
        labelsObj = labelsObj.copy(to=returnType)
        noiselessLabelsObj = noiselessLabelsObj.copy(to=returnType)

    return (pointsObj, labelsObj, noiselessLabelsObj)


def sumAbsoluteDifference(dataOne, dataTwo):
    """
    Aggregates absolute difference between corresponding entries in base
    objects dataOne and dataTwo.

    Checks to see that the vectors (which must be base objects) are of
    the same shape, first. Next it iterates through the corresponding
    points in each vector/matrix and appends the absolute difference
    between corresponding points to a list.

    Finally, the function returns the sum of the absolute differences.
    """

    #compare shapes of data to make sure a comparison is sensible.
    if len(dataOne.features) != len(dataTwo.features):
        msg = "Can't calculate difference between corresponding entries in "
        msg += "dataOne and dataTwo, the underlying data has different "
        msg += "numbers of features."
        raise InvalidArgumentValueCombination(msg)
    if len(dataOne.points) != len(dataTwo.points):
        msg = "Can't calculate difference between corresponding entries in "
        msg += "dataOne and dataTwo, the underlying data has different "
        msg += "numbers of points."
        raise InvalidArgumentValueCombination(msg)

    numpyOne = dataOne.copy(to='numpyarray')
    numpyTwo = dataTwo.copy(to='numpyarray')

    differences = numpyOne - numpyTwo

    absoluteDifferences = np.abs(differences)

    sumAbsoluteDifferences = np.sum(absoluteDifferences)

    return sumAbsoluteDifferences


def _regressorDataset():
    """
    Generates clustered points, where the labels of the points
    within a single cluster are all very similar, but non-identical.
    """

    clusterCount = 3
    pointsPer = 10
    featuresPer = 5

    #add noise to both the features and the labels
    regressorTrainData, trainLabels, noiselessTrainLabels = (
        generateClusteredPoints(clusterCount, pointsPer, featuresPer,
                                addFeatureNoise=True, addLabelNoise=True,
                                addLabelColumn=False))
    regressorTestData, testLabels, noiselessTestLabels = (
        generateClusteredPoints(clusterCount, 1, featuresPer,
                                addFeatureNoise=True, addLabelNoise=True,
                                addLabelColumn=False))

    return ((regressorTrainData, trainLabels, noiselessTrainLabels),
            (regressorTestData, testLabels, noiselessTestLabels))

def _classifierDataset():
    """
    Generates clustered points, hwere the labels of the points
    within each cluster are all identical.
    """

    clusterCount = 3
    pointsPer = 10
    featuresPer = 5

    #add noise to the features only
    trainData, trainLabels, noiselessTrainLabels = (
        generateClusteredPoints(clusterCount, pointsPer, featuresPer,
                                addFeatureNoise=True, addLabelNoise=False,
                                addLabelColumn=False))
    testData, testLabels, noiselessTestLabels = (
        generateClusteredPoints(clusterCount, 1, featuresPer,
                                addFeatureNoise=True, addLabelNoise=False,
                                addLabelColumn=False))

    return ((trainData, trainLabels, noiselessTrainLabels),
            (testData, testLabels, noiselessTestLabels))


class LearnerInspector:
    """
    Class using heirustics to classify the 'type' of problem an
    algorithm is meant to work on.
    e.g. classification, regression, dimensionality reduction, etc.

    Use:
    A LearnerInspector object generates private datasets that are
    intentionally constructed to invite particular results when an
    algorithm is run on them. Once a user has a LearnerInspector object,
    she can call learnerType(algorithmName) and get the 'best guess'
    type for that algorithm.

    Note:
    If characterizing multiple algorithms, use the SAME LearnerInspector
    object, and call learnerType() once for each algorithm you are
    trying to classify.
    """

    def __init__(self):
        """
        Caches the regressor and classifier datasets, to speed up
        learnerType() calls for multiple learners.
        """
        # TODO why is it this value??? should see how it is used and revise
        self.NEAR_THRESHHOLD = .1
        self.EXACT_THRESHHOLD = .00000001

        #initialize datasets for tests
        self.regressorDataTrain, self.regressorDataTest = _regressorDataset()
        #todo use classifier
        self.classifierDataTrain, self.classifierDataTest = (
            _classifierDataset())

    def learnerType(self, learnerName):
        """
        Returns, as a string, the heuristically determined best guess
        for the type of problem the learnerName learner is designed to
        run on.
        Example output: 'classification', 'regression', 'other'
        """
        if not isinstance(learnerName, str):
            raise InvalidArgumentType("learnerName must be a string")
        return self._classifyAlgorithmDecisionTree(learnerName)

    # todo pull from each 'trail' function to find out what possible results
    # it can have then make sure that you've covered all possible combinations
    def _classifyAlgorithmDecisionTree(self, learnerName):
        """
        Implements a decision tree based off of the predicted labels
        returned from the datasets.

        Fundamentally, if the classifier dataset has no error, that
        means the algorithm is likely a classifier, but it could be a
        regressor, if its error is low, however, the algorithm is likely
        a regressor, and if its error is high, or the algorithm crashes
        with the dataset, then the algorithm is likely neither
        classifier nor regressor.

        Next, if the classifier dataset had no error, we want to see if
        the error on the regressor dataset is low. Also, we want to see
        if the algorithm is capable of generating labels that it hasn't
        seen (interpolating a la a regressor).

        If the algorithm doesn't produce any new labels, despite no
        repeated labels, then we assume it is a classifier. If the error
        on the classifier dataset is low, however, and the algorithm
        interpolates labels, then we assume it is a regressor.
        """

        regressorTrialResult = self._regressorTrial(learnerName)
        classifierTrialResult = self._classifierTrial(learnerName)

        # decision tree:
        # if classifier tests gives exact results
        if classifierTrialResult == 'exact':
            # could be classifier or regressor at this point
            # if when given unrepeating labels, algorithm generates duplicate
            # of already seen labels, it is classifer
            if regressorTrialResult == 'repeated_labels':
                return 'classification'
            if regressorTrialResult == 'near':
                return 'regression'
            if regressorTrialResult == 'other':
                return 'classification'
            #should be covered by all cases, raise exception
            msg = 'Decision tree needs to be updated to account for other '
            msg += 'results from regressorTrialResult'
            raise AttributeError(msg)

        # if the classifer data set genereated a low error, but not exact,
        # it is regressor
        if classifierTrialResult == 'near':
            return 'regression'

        # if the classifier dataset doesn't see classifier or regressor
        # behavior, return other
        # todo this is where to insert future sensors for other types of
        # algorithms, but currently we can only resolve classifiers,
        # regressors, and UNKNOWN.
        return 'UNKNOWN'

    def _regressorTrial(self, learnerName):
        """
        Run trainAndApply on the regressor dataset and make judgments
        about the learner based on the results of trainAndApply.
        """
        #unpack already-initialized datasets
        regressorTrainData, trainLabels, _ = self.regressorDataTrain
        regressorTestData, _, noiselessTestLabels = self.regressorDataTest

        try:
            runResults = nimble.trainAndApply(
                learnerName, trainX=regressorTrainData, trainY=trainLabels,
                testX=regressorTestData)
        except Exception: # pylint: disable=broad-except
            return 'other'

        try:
            sumError = sumAbsoluteDifference(runResults, noiselessTestLabels)
        except InvalidArgumentValueCombination:
            return 'other'

        # if the labels are repeated from those that were trained on, then
        # it is a classifier so pass back that labels are repeated
        # if runResults are all in trainLabels, then it's repeating:
        alreadySeenLabelsList = []
        for curPointIndex in range(len(trainLabels.points)):
            alreadySeenLabelsList.append(trainLabels[curPointIndex, 0])

        # check if the learner generated any new label
        # (one it hadn't seen in training)
        unseenLabelFound = False
        for curResultPointIndex in range(len(runResults.points)):
            if runResults[curResultPointIndex, 0] not in alreadySeenLabelsList:
                unseenLabelFound = True
                break

        if not unseenLabelFound:
            return 'repeated_labels'

        if sumError > self.NEAR_THRESHHOLD:
            return 'other'

        return 'near'


    def _classifierTrial(self, learnerName):
        """
        Run trainAndApply on the classifer dataset and make judgments
        about the learner based on the results of trainAndApply.
        """
        #unpack initialized datasets
        trainData, trainLabels, _ = self.classifierDataTrain
        testData, testLabels, _ = self.classifierDataTest

        try:
            runResults = nimble.trainAndApply(learnerName, trainX=trainData,
                                              trainY=trainLabels,
                                              testX=testData)
        except Exception: # pylint: disable=broad-except
            return 'other'

        try:
            #should be identical to noiselessTestLabels
            sumError = sumAbsoluteDifference(runResults, testLabels)
        except InvalidArgumentValueCombination:
            return 'other'

        if sumError > self.NEAR_THRESHHOLD:
            return 'other'
        if sumError > self.EXACT_THRESHHOLD:
            return 'near'

        return 'exact'


def _unpackLearnerName(learnerName):
    """
    Split a learnerName parameter into the portion defining the package,
    and the portion defining the learner.
    """
    if isinstance(learnerName, str):
        splitList = learnerName.split('.', 1)
        if len(splitList) < 2:
            msg = "Recieved ill formed learner name '" + learnerName + "'. "
            msg += "The learner name must identify both the desired package "
            msg += "and learner, separated by a dot. Example:"
            msg += "'sklearn.KNeighborsClassifier'"
            raise InvalidArgumentValue(msg)
        package, name = splitList
    else:
        module = learnerName.__module__.split('.')
        package = module[0]
        if package == 'tensorflow' and 'keras' in module:
            package = 'keras'
        name = learnerName.__name__
        if (issubclass(learnerName, nimble.CustomLearner)
                and package != 'nimble'):
            package = 'custom'
            customInterface = nimble.core.interfaces.available['custom']
            customInterface.registerLearnerClass(learnerName)
    interface = findBestInterface(package)

    return interface, name


def _validTrainData(trainX, trainY):
    """
    Check that the data parameters to train() trainAndApply(), etc. are
    in accepted formats.
    """
    if not isinstance(trainX, Base):
        msg = "trainX may only be an object derived from Base"
        raise InvalidArgumentType(msg)

    if trainY is not None:
        if not isinstance(trainY, (Base, str, int, np.int64)):
            msg = "trainY may only be an object derived from Base, or an "
            msg += "ID of the feature containing labels in testX"
            raise InvalidArgumentType(msg)
        if isinstance(trainY, Base):
            if len(trainY.points) != len(trainX.points):
                msg = "If trainY is a Data object, then it must have the same "
                msg += "number of points as trainX"
                raise InvalidArgumentValueCombination(msg)

def _validTestData(testX, testY):
    # testX is allowed to be None, sometimes it is appropriate to have it be
    # filled using the trainX argument (ie things which transform data, or
    # learn internal structure)
    if testX is not None:
        if not isinstance(testX, Base):
            msg = "testX may only be an object derived from Base"
            raise InvalidArgumentType(msg)

    if testY is not None:
        if not isinstance(testY, (Base, str, numbers.Integral)):
            msg = "testY may only be an object derived from Base, or an ID "
            msg += "of the feature containing labels in testX"
            raise InvalidArgumentType(msg)
        if isinstance(testY, Base):
            if len(testY.points) != len(testX.points):
                msg = "If testY is a Data object, then it must have the same "
                msg += "number of points as testX"
                raise InvalidArgumentValueCombination(msg)


def _validArguments(arguments):
    """
    Check that an arguments parmeter to train() trainAndApply(), etc. is
    an accepted format.
    """
    if not isinstance(arguments, dict) and arguments is not None:
        msg = "The 'arguments' parameter must be a dictionary or None"
        raise InvalidArgumentType(msg)


def _validScoreMode(scoreMode):
    """
    Check that a scoreMode flag to train() trainAndApply(), etc. is an
    accepted value.
    """
    if scoreMode is not None:
        if scoreMode.lower() not in ['bestscore', 'allscores']:
            msg = "scoreMode may only be None, 'bestScore', or 'allScores'"
            raise InvalidArgumentValue(msg)


def _validMultiClassStrategy(multiClassStrategy):
    """
    Check that a multiClassStrategy flag to train() trainAndApply(),
    etc. is an accepted value.
    """
    if multiClassStrategy is not None:
        multiClassStrategy = multiClassStrategy.lower()
        if multiClassStrategy not in ['default', 'onevsall', 'onevsone']:
            msg = "multiClassStrategy may be 'default', 'OneVsAll', or "
            msg += "'OneVsOne'"
            raise InvalidArgumentValue(msg)


def _2dOutputFlagCheck(X, Y, scoreMode, multiClassStrategy):
    outputData = X if Y is None else Y
    if isinstance(outputData, Base):
        needToCheck = len(outputData.features) > 1
    elif isinstance(outputData, (list, tuple)):
        needToCheck = len(outputData) > 1
    elif isinstance(outputData, bool):
        needToCheck = outputData
    else:
        needToCheck = False

    if needToCheck:
        if scoreMode is not None:
            msg = "When dealing with multi dimensional outputs / predictions, "
            msg += "the scoreMode flag is required to be set to None"
            raise InvalidArgumentValueCombination(msg)
        if multiClassStrategy is not None:
            msg = "When dealing with multi dimensional outputs / predictions, "
            msg += "the multiClassStrategy flag is required to be set to "
            msg += "None"
            raise InvalidArgumentValueCombination(msg)


def validateLearningArguments(
        trainX, trainY=None, testX=None, testY=None, arguments=None,
        multiClassStrategy=None, scoreMode=None):
    """
    Argument validation for learning functions.
    """
    _validTrainData(trainX, trainY)
    _validTestData(testX, testY)
    _validArguments(arguments)
    _validScoreMode(scoreMode)
    _validMultiClassStrategy(multiClassStrategy)
    _2dOutputFlagCheck(trainX, trainY, scoreMode, multiClassStrategy)


def trackEntry(func):
    """
    Determine if a function call is the user entry point.
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        if trackEntry.isEntryPoint is None: # entry point
            trackEntry.isEntryPoint = True
        elif trackEntry.isEntryPoint: # previous call was entry point
            trackEntry.isEntryPoint = False
        try:
            ret = func(*args, **kwargs)
        finally:
            trackEntry.isEntryPoint = None # reset

        return ret

    return wrapped

trackEntry.isEntryPoint = None
