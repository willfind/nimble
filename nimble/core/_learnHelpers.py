"""
Helper functions for any functions and objects defined in learn.py.

They are separated here so that that (most) top level user-facing
functions are contained in learn.py without the distraction of helpers.
"""

import itertools

import numpy

import nimble
from nimble.exceptions import InvalidArgumentValue, InvalidArgumentType
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.core.data import Base
from nimble.random import pythonRandom
from nimble.random import numpyRandom
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
            except Exception:
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
            except Exception:
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


class FoldIterator(object):
    """
    Create and iterate through folds.

    Parameters
    ----------
    dataList : list
        A list of data objects to divide into folds.
    folds : int
        The number of folds to create.
    """
    def __init__(self, dataList, folds):
        self.dataList = dataList
        if folds <= 0:
            msg = "Number of folds must be greater than 0"
            raise InvalidArgumentValue(msg)
        self.folds = folds
        self.foldList = self._makeFoldList()
        self.index = 0
        for dat in self.dataList:
            if dat is not None and dat.getTypeString() == 'Sparse':
                dat._sortInternal('point')

    def __iter__(self):
        return self

    def next(self):
        """
        Get next item.
        """
        if self.index >= len(self.foldList):
            raise StopIteration
        # we're going to be separating training and testing sets through
        # extraction, so we have to copy the data in order not to destroy the
        # original sets across multiple folds
        copiedList = []
        for data in self.dataList:
            if data is None:
                copiedList.append(None)
            else:
                copiedList.append(data.copy())

            # we want each training set to be permuted wrt its ordering in the
            # original data. This is setting up a permutation to be applied to
            # each object
            #		indices = range(0, len(copiedList[0].points)
            #                              - len(self.foldList[self.index])))
            #		pythonRandom.shuffle(indices)
        indices = numpy.arange(0, (len(copiedList[0].points)
                                   - len(self.foldList[self.index])))
        numpyRandom.shuffle(indices)

        resultsList = []
        for copied in copiedList:
            if copied is None:
                resultsList.append((None, None))
            else:
                currTest = copied.points.extract(self.foldList[self.index],
                                                 useLog=False)
                currTrain = copied
                currTrain.points.permute(indices, useLog=False)
                resultsList.append((currTrain, currTest))
        self.index = self.index + 1
        return resultsList

    def __next__(self):
        return self.next()

    def _makeFoldList(self):
        if self.dataList is None:
            raise InvalidArgumentType('dataList may not be None')
        if len(self.dataList) == 0:
            raise InvalidArgumentValue("dataList may not be or empty")

        points = len(self.dataList[0].points)
        for data in self.dataList:
            if data is not None:
                if len(data.points) == 0:
                    msg = "One of the objects has 0 points, it is impossible "
                    msg += "to specify a valid number of folds"
                    raise InvalidArgumentValueCombination(msg)
                if len(data.points) != len(self.dataList[0].points):
                    msg = "All data objects in the list must have the same "
                    msg += "number of points and features"
                    raise InvalidArgumentValueCombination(msg)

        # note: we want truncation here
        numInFold = int(points / self.folds)
        if numInFold == 0:
            msg = "Must specify few enough folds so there is a point in each"
            raise InvalidArgumentValue(msg)

        # randomly select the folded portions
        indices = list(range(points))
        pythonRandom.shuffle(indices)
        foldList = []
        for fold in range(self.folds):
            start = fold * numInFold
            if fold == self.folds - 1:
                end = points
            else:
                end = (fold + 1) * numInFold
            foldList.append(indices[start:end])
        return foldList

class ArgumentIterator(object):
    """
    Create and iterate through argument permutations.

    Parameters
    ----------
    rawArgumentInput : dict
        Mapping of argument names (strings) to values.
        e.g. {'a': CV([1, 2, 3]), 'b': nimble.CV([4,5]), 'c': 6}
    """

    def __init__(self, rawArgumentInput):
        self.rawArgumentInput = rawArgumentInput
        self.index = 0
        if not isinstance(rawArgumentInput, dict):
            msg = "ArgumentIterator objects require dictionary's to "
            msg += "initialize- e.g. {'a':CV([1,2,3]), 'b':CV([4,5])} This "
            msg += "is the form generated by **args in a function argument."
            raise InvalidArgumentType(msg)

        # i.e. if rawArgumentInput == {}
        if len(rawArgumentInput) == 0:
            self.numPermutations = 1
            self.permutationsList = [{}]
        else:
            iterableArgDict = {}
            self.numPermutations = 1
            for key in rawArgumentInput.keys():
                if isinstance(rawArgumentInput[key], nimble.CV):
                    self.numPermutations *= len(rawArgumentInput[key])
                    iterableArgDict[key] = rawArgumentInput[key]
                else: # numPermutations not increased
                    # wrap in iterable so that itertools.product will treat
                    # whatever this value is as a single argument value even
                    # if the value itself is an iterable
                    iterableArgDict[key] = (rawArgumentInput[key],)

            # note: calls to keys() and values() will directly correspond as
            # since no modification is made to iterableArgDict between calls.
            self.permutationsList = []
            for permutation in itertools.product(*iterableArgDict.values()):
                permutationDict = {}
                for i, argument in enumerate(iterableArgDict.keys()):
                    permutationDict[argument] = permutation[i]
                self.permutationsList.append(permutationDict)

            assert len(self.permutationsList) == self.numPermutations

    def __iter__(self):
        return self

    def next(self):
        """
        Get next item.
        """
        if self.index >= self.numPermutations:
            self.index = 0
            raise StopIteration
        permutation = self.permutationsList[self.index]
        self.index += 1
        return permutation

    def __next__(self):
        return self.next()

    def reset(self):
        """
        Reset index to 0.
        """
        self.index = 0


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
    each point associated with a cluster has numFeaturesPerPoint
    features. The value of each entry in the feature vector is
    clusterNumber+noise. Each point in the cluster has the same feature
    vector, with different noise.

    NOTE: if addFeatureNoise and addLabelNoise are false, then the
    'clusters' are actually all contain just repeated points, where each
    point in the cluster has the same features and the same labels.

    Returns
    -------
    tuple of nimble.Base objects:
    (pointsObj, labelsObj, noiselessLabelsObj)
    """

    pointsList = []
    labelsList = []
    clusterNoiselessLabelList = []

    def _noiseTerm():
        return pythonRandom.random() * 0.0001 - 0.00005

    for curCluster in range(numClusters):
        for _ in range(numPointsPerCluster):
            curFeatureVector = [float(curCluster) for x
                                in range(numFeaturesPerPoint)]

            if addFeatureNoise:
                curFeatureVector = [_noiseTerm() + entry for entry
                                    in curFeatureVector]

            if addLabelNoise:
                curLabel = _noiseTerm() + curCluster
            else:
                curLabel = curCluster

            if addLabelColumn:
                curFeatureVector.append(curLabel)

            #append curLabel as a list to maintain dimensionality
            labelsList.append([curLabel])

            pointsList.append(curFeatureVector)
            clusterNoiselessLabelList.append([float(curCluster)])

    pointsArray = numpy.array(pointsList, dtype=numpy.float)
    labelsArray = numpy.array(labelsList, dtype=numpy.float)
    clusterNoiselessLabelArray = numpy.array(clusterNoiselessLabelList,
                                             dtype=numpy.float)
    # todo verify that your list of lists is valid initializer for all
    # datatypes, not just matrix
    # then convert
    # finally make matrix object out of the list of points w/ labels in last
    # column of each vector/entry:
    pointsObj = nimble.data('Matrix', pointsArray, useLog=False)

    labelsObj = nimble.data('Matrix', labelsArray, useLog=False)

    # todo change actuallavels to something like associatedClusterCentroid
    noiselessLabelsObj = nimble.data('Matrix', clusterNoiselessLabelArray,
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

    absoluteDifferences = numpy.abs(differences)

    sumAbsoluteDifferences = numpy.sum(absoluteDifferences)

    return sumAbsoluteDifferences


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
        self.regressorDataTrain, self.regressorDataTest = (
            self._regressorDataset())
        #todo use classifier
        self.classifierDataTrain, self.classifierDataTest = (
            self._classifierDataset())

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
        # regressors, and other.
        return 'other'

    def _regressorDataset(self):
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

    def _classifierDataset(self):
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
        except Exception:
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
        else:
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
        except Exception:
            return 'other'

        try:
            #should be identical to noiselessTestLabels
            sumError = sumAbsoluteDifference(runResults, testLabels)
        except InvalidArgumentValueCombination:
            return 'other'

        if sumError > self.NEAR_THRESHHOLD:
            return 'other'
        elif sumError > self.EXACT_THRESHHOLD:
            return 'near'
        else:
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
            msg += "and learner, separated by a dot. Example:'mlpy.KNN'"
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
            if name not in customInterface.registeredLearners:
                customInterface.registerLearnerClass(learnerName)
    interface = findBestInterface(package)

    return interface, name


class TrainValidator:
    """
    Perform and record validation for training, applying, and testing.

    Provides methods for validating parameters when training, applying,
    and testing. Repeated validation of the same parameters can be
    prevented by passing an instance as the useLog parameter. Instances
    should be instantiated with the useLog parameter provided to the
    function whenever it will be passed to the nimble.train function so
    that cross-validation is logged correctly.
    """
    def __init__(self, useLog=None):
        self.useLog = useLog
        self.validTrainData = False
        self.validTestData = False
        self.validArgs = False
        self.validScoreMode = False
        self.validMultiClassStrategy = False
        self.validOutputFlags = False

    def validateTrainData(self, trainX, trainY):
        """
        Check the types of the training data.
        """
        if not self.validTrainData:
            if not isinstance(trainX, Base):
                msg = "trainX may only be an object derived from Base"
                raise InvalidArgumentType(msg)

            if trainY is not None:
                if not isinstance(trainY, (Base, str, int, numpy.int64)):
                    msg = "trainY may only be an object derived from Base, or "
                    msg += "an ID of the feature containing labels in testX"
                    raise InvalidArgumentType(msg)
                if (isinstance(trainY, Base) and
                        len(trainY.points) != len(trainX.points)):
                    msg = "If trainY is a Data object, then it must have the "
                    msg += "same number of points as trainX"
                    raise InvalidArgumentValueCombination(msg)

            self.validTrainData = True

    def validateTestData(self, testX, testY=None, testXRequired=True,
                         testYRequired=False):
        """
        Check the types of the testing data.
        """
        if not self.validTestData:
            if testXRequired and testX is None:
                raise InvalidArgumentType("testX must be provided")
            if testX is not None and not isinstance(testX, Base):
                msg = "testX may only be an object derived from Base"
                raise InvalidArgumentType(msg)

            if testYRequired and testY is None:
                raise InvalidArgumentType("testY must be provided")
            acceptedTypes = (Base, str, int)
            if testY is not None and not isinstance(testY, acceptedTypes):
                msg = "testY may only be an object derived from Base, or "
                msg += "an ID of the feature containing labels in testX"
                raise InvalidArgumentType(msg)
            if (isinstance(testY, Base)
                    and len(testY.points) != len(testX.points)):
                msg = "If testY is a Data object, then it must have "
                msg += "the same number of points as testX"
                raise InvalidArgumentValueCombination(msg)

            self.validTestData = True

    def validateArguments(self, arguments):
        """
        Check the types of the argument parameters.
        """
        if not self.validArgs:
            if not isinstance(arguments, dict) and arguments is not None:
                msg = "The 'arguments' parameter must be a dictionary or None"
                raise InvalidArgumentType(msg)
            self.validArgs = True

    def validateScoreMode(self, scoreMode):
        """
        Check that a scoreMode flag is an accepted value.
        """
        if not self.validScoreMode:
            scoreMode = scoreMode.lower()
            if scoreMode not in ['label', 'bestscore', 'allscores']:
                msg = "scoreMode may only be 'label' 'bestScore' or "
                msg += "'allScores'"
                raise InvalidArgumentValue(msg)

            self.validScoreMode = True

    def validateMultiClassStrategy(self, multiClassStrategy):
        """
        Check that a multiClassStrategy flag is an accepted value.
        """
        if not self.validMultiClassStrategy:
            multiClassStrategy = multiClassStrategy.lower()
            if multiClassStrategy not in ['default', 'onevsall', 'onevsone']:
                msg = "multiClassStrategy may be 'default' 'OneVsAll' or "
                msg += "'OneVsOne'"
                raise InvalidArgumentValue(msg)

            self.validMultiClassStrategy = True

    def validateOutputFlags(self, X, Y, scoreMode, multiClassStrategy):
        """
        Check flags are compatible when output is 2D.
        """
        if not self.validOutputFlags:
            outputData = X if Y is None else Y
            if isinstance(outputData, Base):
                needToCheck = len(outputData.features) > 1
            elif isinstance(outputData, (list, tuple)):
                needToCheck = len(outputData) > 1
            elif isinstance(outputData, bool):
                needToCheck = outputData
            else:
                needToCheck = False

            if needToCheck and scoreMode not in [None, 'label']:
                msg = "When dealing with multi dimensional outputs / "
                msg += "predictions, the scoreMode flag is required to be "
                msg += "set to 'label'"
                raise InvalidArgumentValueCombination(msg)
            if needToCheck and multiClassStrategy not in [None, 'default']:
                msg = "When dealing with multi dimensional outputs / "
                msg += "predictions, the multiClassStrategy flag is "
                msg += "required to be set to 'default'"
                raise InvalidArgumentValueCombination(msg)

            self.validOutputFlags = True
