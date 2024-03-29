
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
Utility functions that could be useful in multiple interfaces
"""

import sys
import importlib
import configparser
import warnings
import inspect
import re

import nimble
from nimble.exceptions import InvalidArgumentValue
from nimble.random import pythonRandom
from nimble.core._learnHelpers import _validTestData, _validArguments
from nimble.core._learnHelpers import _validScoreMode, _2dOutputFlagCheck
from nimble.core._learnHelpers import generateClusteredPoints


class PythonSearcher(object):
    """
    Explore a python package.

    Parameters
    ----------
    baseModule : module
        The imported package.
    baseContents : list
        A list of the exposed attributes. Often the __all__ attribute of
        the ``baseModule``.
    specialCases : dict
        TODO
    isLearner : function
        Returns True if an attribute is a learner.
    allowedDepth : int
        The maximum depth to search the package's directory tree.
    """
    def __init__(self, baseModule, isLearner, allowedDepth):
        self._baseModule = baseModule
        self._baseContents = self._getContents(baseModule)
        self._isLearner = isLearner
        self._allowedDepth = allowedDepth

        self._locationCache = {}
        self._havePopulated = False

    def allLearners(self):
        """
        Return a list of names of modules that satisfy the isLearner
        function found according the search parameters defined in this
        object.
        """
        # We make it impossible to satisfy search query, as a consequence, we
        # populate the cache with every possible learner
        if not self._havePopulated:
            self.findInPackage(None, None)
        self._havePopulated = True

        ret = []
        # the location cache holds names to object mappings
        for name, val in self._locationCache.items():
            if self._isLearner(val):
                ret.append(name)

        return ret

    def _getContents(self, module):
        if hasattr(module, '__all__'):
            return module.__all__
        return [d for d in dir(module)
                if d == '__init__' or d == '__call__' or not d.startswith("_")]

    def findInPackage(self, parent, name):
        """
        Import the desired python package, and search for the module
        containing the wanted learner. For use by interfaces to python
        packages.
        """
        contents = self._baseContents
        searchIn = self._baseModule
        allowedDepth = self._allowedDepth
        if parent is not None:
            if parent in self._locationCache:
                searchIn = self._locationCache[parent]
            else:
                searchIn = self._findInPackageRecursive(searchIn, parent,
                                                        allowedDepth, contents)
            if searchIn is None:
                return None
            allowedDepth = 0
            contents = self._getContents(searchIn)

        if name in self._locationCache:
            ret = self._locationCache[name]
        else:
            ret = self._findInPackageRecursive(searchIn, name, allowedDepth,
                                               contents)

        return ret

    def _findInPackageRecursive(self, parent, target, allowedDepth, contents):
        for name in contents:
            isModule = False
            # we want to add learners to the cache
            # NOTE: this adds learners regardless of the state of the target
            try:
                attr = getattr(parent, name)
                if self._isLearner(attr):
                    self._locationCache[name] = attr
                elif inspect.ismodule(attr):
                    isModule = True
            except AttributeError:
                try:
                    fullName = parent.__name__ + "." + name
                    attr = importlib.import_module(fullName)
                    isModule = True
                except ImportError:
                    continue

            if name == target:
                return attr

            if isModule and allowedDepth > 0:
                subContents = self._getContents(attr)

                ret = self._findInPackageRecursive(attr, target,
                                                   allowedDepth - 1,
                                                   subContents)
                if ret is not None:
                    return ret

        return None


def checkClassificationStrategy(interface, learnerName, trainArgs, scoreArgs,
                                shape, seed):
    """
    Helper to determine the classification strategy used for a given
    learner called using the given interface with the given args. Runs a
    trial on data with 4 classes so that we can use structural.
    """
    # we need classes > 2 to test the multiclass strategy, and we should be
    # able to tell structurally when classes != 3. Shape of data must match
    # original for some learners
    numPts, numFts = shape
    # ensure there are enough total points to match original data and reorder
    # labels as 0, 1, 2 ,3, 0, 1, 2, 3, ... so testing object is guaranteed to
    # have one of each label
    numPointsPerCluster = max(1, numPts // 4) * 2
    dataX, _, dataY = generateClusteredPoints(4, numPointsPerCluster,
                                              numFts, addLabelNoise=False)
    reorder = []
    for i in range(numPointsPerCluster):
        for j in range(4):
            reorder.append(i + (j * numPointsPerCluster))
    dataX.points.permute(reorder, useLog=False)
    dataY.points.permute(reorder, useLog=False)
    # Note this check will fail if the learner expects data with shape[0] < 4
    endIdx = max(4, numPts)
    xObj = dataX[:endIdx, :]
    yObj = dataY[:endIdx, :]
    testObj = xObj.copy()
    try:
        tlObj = interface.train(learnerName, xObj, yObj, arguments=trainArgs,
                                randomSeed=seed)
        applyResults = tlObj.apply(testObj, arguments=scoreArgs, useLog=False)

        transformer = interface._inputTransformation
        (_, _, testTrans, _) =  transformer(learnerName, None, None, testObj,
                                            None, trainArgs, tlObj._customDict)

        rawScores = interface._getScores(tlObj.learnerName, tlObj._backend,
                                         testTrans, scoreArgs,
                                         tlObj._transformedArguments,
                                         tlObj._customDict)
        return ovaNotOvOFormatted(rawScores, applyResults, 4)

    except Exception as e:
        # 4 labels is more than Keras learner expected, should be ova
        labelErr = r'Received a label value of 3 which is outside the '
        labelErr += 'valid range'
        if re.search(labelErr, str(e)):
            return True
        raise

def ovaNotOvOFormatted(scoresPerPoint, predictedLabels, numLabels,
                       useSize=True):
    """
    Return True if the scoresPerPoint list of list has scores formatted
    for a one vs all strategy, False if it is for a one vs one strategy.
    None if there are no definitive cases. May throw an
    InvalidArgumentValue if there are conflicting definitive votes for
    different strategies.
    """
    if not isinstance(scoresPerPoint, nimble.core.data.Base):
        scoresPerPoint = nimble.data(scoresPerPoint, copyData=True,
                                     useLog=False)
    if not isinstance(predictedLabels, nimble.core.data.Base):
        predictedLabels = nimble.data(predictedLabels,
                                      copyData=True, useLog=False)
    length = len(scoresPerPoint.points)
    scoreLength = len(scoresPerPoint.features)
    # let n = number of classes
    # ova : number scores = n
    # ovo : number scores = (n * (n-1) ) / 2
    # only at n = 3 are they equal
    if useSize and numLabels != 3:
        return scoreLength == numLabels
    # we want to check random points out of all the possible data
    check = 20
    if length < check:
        check = length
    checkList = pythonRandom.sample(range(length), check)
    results = []
    for i in checkList:
        strategy = verifyOvANotOvOSingleList(scoresPerPoint.pointView(i),
                                             predictedLabels[i, 0], numLabels)
        results.append(strategy)

    ovaVote = results.count(True)
    ovoVote = results.count(False)

    # different points were unambigously in different scoring strategies.
    # Can't make sense of that
    if ovoVote > 0 and ovaVote > 0:
        msg = "We found conflicting scoring strategies for multiclass "
        msg += "classification, cannot verify one way or the other"
        raise InvalidArgumentValue(msg)
    # only definitive votes were ova
    if ovaVote > 0:
        return True
    # only definitive votes were ovo
    if ovoVote > 0:
        return False
    # no unambiguous cases: return None as a sentinal for unsure
    return None


def verifyOvANotOvOSingleList(scoreList, predictedLabelIndex, numLabels):
    """
    We cannot determine from length whether scores are produced using a
    one-vs-all strategy or a one-vs-one strategy. This checks a
    particular set of scores by simulating OvA and OvO prediction
    strategies, and checking the results.

    Returns True if it is OvA consistent and not OvO consistent.
    Returns False if it is not OvA consistent but is OvO consistent.
    Returns None otherwise.
    """
    # simulate OvA prediction strategy
    maxScoreIndex = -1
    maxScore = -sys.maxsize - 1
    for i, score in enumerate(scoreList):
        if score > maxScore:
            maxScore = score
            maxScoreIndex = i

    ovaConsistent = maxScoreIndex == predictedLabelIndex

    # simulate OvO prediction strategy
    combinedScores = calculateSingleLabelScoresFromOneVsOneScores(scoreList,
                                                                  numLabels)
    maxScoreIndex = -1
    maxScore = -sys.maxsize - 1
    for i, combinedScore in enumerate(combinedScores):
        if combinedScore > maxScore:
            maxScore = combinedScore
            maxScoreIndex = i
    ovoConsistent = maxScoreIndex == predictedLabelIndex

    if ovaConsistent and not ovoConsistent:
        return True
    if not ovaConsistent and ovoConsistent:
        return False
    if ovaConsistent and ovoConsistent:
        return None

    msg = "The given scoreList does not produce the predicted label with "
    msg += "either of our combination strategies. We therefore cannot "
    msg += "verify the format of the scores"
    raise InvalidArgumentValue(msg)


def calculateSingleLabelScoresFromOneVsOneScores(oneVOneData, numLabels):
    """
    oneVOneData is the flat list of scores of each least ordered pair of
    labels, ordered (score label0 vs label1... score label0 vs labeln-1,
    score label1 vs label2 ... score labeln-2 vs labeln-1). We return a
    length n list where the ith value is the ratio of wins for label i
    in the label vs label tournament.
    """
    ret = []
    for i in range(numLabels):
        wins = 0
        for j in range(numLabels):
            score = valueFromOneVOneData(oneVOneData, i, j, numLabels)
            if score is not None and score > 0:
                wins += 1
        ret.append(float(wins) / (numLabels - 1))

    return ret


def valueFromOneVOneData(oneVOneData, posLabel, negLabel, numLabels):
    """
    Get value from one-vs-one data.
    """
    flagNegative = False
    if posLabel == negLabel:
        return None
    if posLabel > negLabel:
        flagNegative = True
        negLabel, posLabel = posLabel, negLabel

    start = (posLabel * numLabels) - ((posLabel * (posLabel + 1)) // 2)
    offset = negLabel - (posLabel + 1)
    value = oneVOneData[start + offset]
    if flagNegative:
        return 0 - value

    return value


def generateBinaryScoresFromHigherSortedLabelScores(scoresPerPoint):
    """
    Given an indexable containing the score for the label with a higher
    natural ordering corresponding to the ith test point of an n point
    binary classification problem set, construct and return an array
    with two columns and n rows, where the ith row corresponds to the
    ith test point, the first column contains the score for the label
    with the lower natural sort order, and the second column contains
    the score for the label with the higher natural sort order.
    """
    newScoresPerPoint = []
    for i in range(len(scoresPerPoint.points)):
        pointScoreList = []
        currScore = scoresPerPoint[i, 0]
        pointScoreList.append((-1) * currScore)
        pointScoreList.append(currScore)
        newScoresPerPoint.append(pointScoreList)
    return newScoresPerPoint


def cacheWrapper(toWrap):
    """
    Decorator to be used in universal Interface which will record the
    results of call so that they can be easily returned again if the
    same call is made later.
    """
    cache = {}

    def wrapped(*args):
        if args in cache:
            return cache[args]

        ret = toWrap(*args)
        cache[args] = ret
        return ret

    return wrapped


def collectAttributes(obj, generators, checkers):
    """
    Helper to collect, validate, and return all (relevant) attributes
    associated with a python object (learner, kernel, etc.). The
    returned value will be a dict, mapping names of attributes to values
    of attributes. In the case of collisions (especially in the
    recursive case) the attribute names will be prefaced with the name
    of the object from which they originate.

    Parameters
    ----------
    obj : object
        The python object (learner, kernel, etc.) to collect from. It
        will be passed as the first argument to all checker functions.
    generators : list
        List of functions which generate possible attributes. Each will
        be called with a single argument: the obj parameter, and must
        return a dict. If None is passed, we will automatically use
        attributes as accessed via dir(obj) as the only possiblities.
    checkers : list
        List of functions which will be called to see if a possible
        attribute is to be included in the output. Each checker function
        must take three arguments: the object, the name of the possible
        attribute, and finally the value of the possible attribute. If
        the possible attribute is to be included in the output, the
        function must return True.
    """
    with warnings.catch_warnings():
        # interfaces can issue many warnings when getting certain attributes
        # and they are not relevant to the user at this point so ignore them
        warnings.simplefilter('ignore')
        if generators is None:
            def wrappedDir(obj):
                ret = {}
                keys = dir(obj)
                for k in keys:
                    try:
                        val = getattr(obj, k)
                        ret[k] = val
                    # safety against any sort of error someone may have in
                    # their property code.
                    except (AttributeError, ValueError):
                        pass
                return ret

            generators = [wrappedDir]

        ret = {}

        for gen in generators:
            possibleDict = gen(obj)
            for possibleName, possibleValue in possibleDict.items():
                add = True
                for check in checkers:
                    if not check(obj, possibleName, possibleValue):
                        add = False
                if add:
                    ret[possibleName] = possibleValue

        return ret


def noLeading__(obj, name, value): # pylint: disable=unused-argument, invalid-name
    """
    Determine if a name does NOT begin with two leading underscores.
    """
    if name.startswith('__'):
        return False
    return True


def notCallable(obj, name, value): # pylint: disable=unused-argument
    """
    Determine if a value is NOT callable.
    """
    if hasattr(value, '__call__'):
        return False
    return True


def notABCAssociated(obj, name, value): # pylint: disable=unused-argument
    """
    Determine if a name is NOT ABC associated.
    """
    if name.startswith("_abc"):
        return False
    return True

def removeFromArray(orig, toIgnore):
    """
    Remove objects from an array.
    """
    temp = []
    for entry in orig:
        if not entry in toIgnore:
            temp.append(entry)
    return temp


def removeFromTailMatchedLists(full, matched, toIgnore):
    """
    'full' is some list n, 'matched' is a list with length m, where
    m is less than or equal to n, where the last m values of full
    are matched against their positions in matched. If one of those
    is to be removed, it is to be removed in both.
    """
    temp = {}
    if matched is not None:
        for i in range(len(full)):
            fullIdx = len(full) - 1 - i
            if i < len(matched):
                matchedIdx = len(matched) - 1 - i
                temp[full[fullIdx]] = matched[matchedIdx]
            else:
                temp[full[fullIdx]] = None
    else:
        retFull = removeFromArray(full, toIgnore)
        return (retFull, matched)

    for ignoreKey in toIgnore:
        if ignoreKey in temp:
            del temp[ignoreKey]

    retFull = []
    retMatched = []
    for i, name in enumerate(full):
        name = full[i]
        if name in temp:
            retFull.append(name)
            if (i - (len(full) - len(matched))) >= 0:
                retMatched.append(temp[name])

    return (retFull, retMatched)


def modifyImportPathAndImport(canonicalName, importPackage):
    """
    Supports importing packages and subpackages for interfaces from
    an alternative location specified by nimble.settings.

    Parameters
    ----------
    canonicalName : str
        The canonical name of the interface using this helper.
        Alternative locations are located in nimble.settings under this
        name.
    importPackage : str
        The name of the package to import. Most often this mirrors the
        canonicalName. It can also be a subpackage of the interface or
        another location from which to import the interface package.
        Other locations are provided by us, for example we prioritize
        tensorflow.keras over keras, but a user's keras location takes
        priority so importPackage is modified to be the canonicalName.
    """
    sysPathBackup = sys.path.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            try:
                location = nimble.settings.get(canonicalName, 'location')
                if location:
                    sys.path.insert(0, location)
                    if not importPackage.startswith(canonicalName):
                        importPackage = canonicalName
            except configparser.Error:
                pass

            return importlib.import_module(importPackage)
        finally:
            sys.path = sysPathBackup


def generateAllPairs(items):
    """
    Given a list of items, generate a list of all possible pairs
    (2-combinations) of items from the list, and return as a list
    of tuples.  Assumes that no two items in the list refer to the same
    object or number.  If there are duplicates in the input list, there
    will be duplicates in the output list.
    """
    if items is None or len(items) == 0:
        return None

    pairs = []
    for i, firstItem in enumerate(items):
        for j in range(i + 1, len(items)):
            secondItem = items[j]
            pair = (firstItem, secondItem)
            pairs.append(pair)

    return pairs


def countWins(predictions):
    """
    Count how many contests were won by each label in the set.  If a
    class label doesn't win any predictions, it will not be included in
    the results.  Return a dictionary: {classLabel: # of contests won}.
    """
    predictionCounts = {}
    for prediction in predictions:
        if prediction in predictionCounts:
            predictionCounts[prediction] += 1
        else:
            predictionCounts[prediction] = 1

    return predictionCounts


def extractWinningPredictionLabel(predictions):
    """
    Provided a list of tournament winners (class labels) for one
    point/row in a test set, choose the label that wins the most
    tournaments.  Returns the winning label.
    """
    #Count how many times each class won
    predictionCounts = countWins(predictions)

    #get the class that won the most tournaments
    #TODO: what if there are ties?
    return max(predictionCounts.keys(),
               key=(lambda key: predictionCounts[key]))


def extractWinningPredictionIndex(predictionScores):
    """
    Provided a list of confidence scores for one point/row in a test
    set, return the index of the column (i.e. label) of the highest
    score.  If no score in the list of predictionScores is a number
    greater than negative infinity, returns None.
    """
    maxScore = float("-inf")
    maxScoreIndex = -1
    for i, score in enumerate(predictionScores):
        if score > maxScore:
            maxScore = score
            maxScoreIndex = i

    if maxScoreIndex == -1:
        return None

    return maxScoreIndex


def extractWinningPredictionIndexAndScore(predictionScores, featureNamesItoN):
    """
    Provided a list of confidence scores for one point/row in a test
    set, return the index of the column (i.e. label) of the highest
    score.  If no score in the list of predictionScores is a number
    greater than negative infinity, returns None.
    """
    allScores = extractConfidenceScores(predictionScores, featureNamesItoN)

    if allScores is None:
        return None

    bestScore = float("-inf")
    bestLabel = None
    for key in allScores:
        value = allScores[key]
        if value > bestScore:
            bestScore = value
            bestLabel = key
    return (bestLabel, bestScore)


def extractConfidenceScores(predictionScores, featureNamesItoN):
    """
    Provided a list of confidence scores for one point/row in a test
    set, and a dict mapping indices to featureNames, return a dict
    mapping featureNames to scores.
    """
    if predictionScores is None or len(predictionScores) == 0:
        return None

    scoreMap = {}
    for i, score in enumerate(predictionScores):
        label = featureNamesItoN[i]
        scoreMap[label] = score

    return scoreMap

def validateTestingArguments(testX, testY=None, arguments=None,
                             has2dOutput=False, scoreMode=None):
    """
    Argument validation for trained learner methods.
    """
    _validTestData(testX, testY)
    _validArguments(arguments)
    _validScoreMode(scoreMode)
    _2dOutputFlagCheck(has2dOutput, None, scoreMode, None)

def checkArgsForRandomParam(arguments, randomParam):
    """
    Raise exception if user provides a value for the interface's
    randomness parameter.
    """
    if randomParam in arguments:
        msg = 'Nimble disallows the {0} parameter and provides the '
        msg += 'randomSeed parameter for randomness control. Provide '
        msg += '{1} as the value of the randomSeed parameter instead.'
        value = arguments[randomParam]
        raise InvalidArgumentValue(msg.format(randomParam, value))

def validInitParams(initNames, arguments, randomSeed, randomParam):
    """
    Generate a seed when the learner parameter controls randomness.

    Only applies if the interface's random parameter has not been
    specified and the learner uses the random parameter. The
    generated seed will be added to the initParams dictionary so
    that the learner is always instantiated with a set state.
    """
    checkArgsForRandomParam(arguments, randomParam)
    initParams = {name: arguments[name] for name in initNames
                  if name in arguments}
    if randomParam in initNames:
        initParams[randomParam] = randomSeed

    return initParams
