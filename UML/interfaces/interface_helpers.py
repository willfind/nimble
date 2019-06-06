"""
Utility functions that could be useful in multiple interfaces
"""

from __future__ import absolute_import
import sys
import importlib
import configparser

import numpy
import six
from six.moves import range

import UML
from UML.exceptions import InvalidArgumentValue
from UML.randomness import pythonRandom


def makeArgString(wanted, argDict, prefix, infix, postfix):
    """
    Construct and return a long string containing argument and value
    pairs, each separated and surrounded by the given strings. If wanted
    is None, then no args are put in the string.
    """
    argString = ""
    if wanted is None:
        return argString
    for arg in wanted:
        if arg in argDict:
            value = argDict[arg]
            if isinstance(value, six.string_types):
                value = "\"" + value + "\""
            else:
                value = str(value)
            argString += prefix + arg + infix + value + postfix
    return argString


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
    def __init__(self, baseModule, baseContents, specialCases, isLearner,
                 allowedDepth):
        self._baseModule = baseModule
        self._baseContents = baseContents
        self._specialCases = specialCases
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
        for name in self._locationCache.keys():
            val = self._locationCache[name]
            if self._isLearner(val):
                ret.append(name)

        return ret


    def findInPackage(self, parent, name):
        """
        Import the desired python package, and search for the module
        containing the wanted learner. For use by interfaces to python
        packages.
        """
        specialKey = parent + '.' + name if parent is not None else name
        if specialKey in self._specialCases:
            return self._specialCases[specialKey]

        contents = self._baseContents

        searchIn = self._baseModule
        allowedDepth = self._allowedDepth
        if parent is not None:
            if parent in self._locationCache:
                searchIn = self._locationCache[parent]
            else:
                searchIn = self._findInPackageRecursive(searchIn, parent,
                                                        allowedDepth, contents)
            allowedDepth = 0
            if hasattr(searchIn, '__all__'):
                contents = searchIn.__all__
            else:
                contents = dir(searchIn)
            if searchIn is None:
                return None

        if name in self._locationCache:
            ret = self._locationCache[name]
        else:
            ret = self._findInPackageRecursive(searchIn, name, allowedDepth,
                                               contents)

        return ret

    def _findInPackageRecursive(self, parent, target, allowedDepth, contents):
        for name in contents:
            if name.startswith("_") and name != '__init__':
                continue
            try:
                subMod = getattr(parent, name)
            except AttributeError:
                try:
                    fullName = parent.__name__ + "." + name
                    subMod = importlib.import_module(fullName)
                except ImportError:
                    continue

            # we want to add learners, and the parents of learners to the cache
            # NOTE: this adds learners regardless of the state of the target
            if self._isLearner(subMod):
                self._locationCache[str(name)] = subMod

            if name == target:
                return subMod

            if hasattr(subMod, '__all__'):
                subContents = subMod.__all__
            else:
                subContents = dir(subMod)

            if allowedDepth > 0:
                ret = self._findInPackageRecursive(subMod, target,
                                                   allowedDepth - 1,
                                                   subContents)
                if ret is not None:
                    return ret

        return None


#TODO what about multiple levels???
def findModule(learnerName, packageName, packageLocation):
    """
    Import the desired python package, and search for the module
    containing the wanted learner. For use by interfaces to python
    packages.
    """
    putOnSearchPath(packageLocation)
    exec("import " + packageName)

    contents = eval("dir(" + packageName + ")")

    # if all is defined, then we defer to the package to provide
    # a sensible list of what it contains
    if "__all__" in contents:
        contents = eval(packageName + ".__all__")

    for moduleName in contents:
        if moduleName.startswith("__"):
            continue
        cmd = "import " + packageName + "." + moduleName
        try:
            exec(cmd)
        except ImportError:
            continue
        subContents = eval("dir(" + packageName + "." + moduleName + ")")
        if ".__all__" in subContents:
            contents = eval(packageName + "." + moduleName + ".__all__")
        if learnerName in subContents:
            return moduleName

    return None


def putOnSearchPath(wantedPath):
    """
    Add a path to sys.path.
    """
    if wantedPath is None:
        return
    elif wantedPath in sys.path:
        return
    else:
        sys.path.append(wantedPath)


def checkClassificationStrategy(interface, learnerName, algArgs):
    """
    Helper to determine the classification strategy used for a given
    learner called using the given interface with the given args. Runs a
    trial on data with 4 classes so that we can use structural.
    """
    dataX = [[-100, 3], [-122, 1], [118, 1], [117, 5],
             [1, -191], [-2, -118], [-1, 200], [3, 222]]
    xObj = UML.createData("Matrix", dataX, useLog=False)
    # we need classes > 2 to test the multiclass strategy, and we should be
    # able to tell structurally when classes != 3
    dataY = [[0], [0], [1], [1], [2], [2], [3], [3]]
    yObj = UML.createData("Matrix", dataY, useLog=False)
    dataTest = [[0, 0], [-100, 0], [100, 0], [0, -100], [0, 100]]
    testObj = UML.createData("Matrix", dataTest, useLog=False)

    tlObj = interface.train(learnerName, xObj, yObj, arguments=algArgs)
    applyResults = tlObj.apply(testObj, arguments=algArgs, useLog=False)
    (_, _, testTrans, _) = interface._inputTransformation(
        learnerName, None, None, testObj, algArgs, tlObj.customDict)
    rawScores = interface._getScores(tlObj.learnerName, tlObj.backend,
                                     testTrans, algArgs,
                                     tlObj.transformedArguments,
                                     tlObj.customDict)

    return ovaNotOvOFormatted(rawScores, applyResults, 4)


def ovaNotOvOFormatted(scoresPerPoint, predictedLabels, numLabels,
                       useSize=True):
    """
    Return True if the scoresPerPoint list of list has scores formatted
    for a one vs all strategy, False if it is for a one vs one strategy.
    None if there are no definitive cases. May throw an
    InvalidArgumentValue if there are conflicting definitive votes for
    different strategies.
    """
    if not isinstance(scoresPerPoint, UML.data.Base):
        scoresPerPoint = UML.createData('Matrix', scoresPerPoint,
                                        reuseData=True, useLog=False)
    if not isinstance(predictedLabels, UML.data.Base):
        predictedLabels = UML.createData('Matrix', predictedLabels,
                                         reuseData=True, useLog=False)
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
    elif ovaVote > 0:
        return True
    # only definitive votes were ovo
    elif ovoVote > 0:
        return False
    # no unambiguous cases: return None as a sentinal for unsure
    else:
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
    for i in range(len(scoreList)):
        if scoreList[i] > maxScore:
            maxScore = scoreList[i]
            maxScoreIndex = i

    ovaConsistent = maxScoreIndex == predictedLabelIndex

    # simulate OvO prediction strategy
    combinedScores = calculateSingleLabelScoresFromOneVsOneScores(scoreList,
                                                                  numLabels)
    maxScoreIndex = -1
    maxScore = -sys.maxsize - 1
    for i in range(len(combinedScores)):
        if combinedScores[i] > maxScore:
            maxScore = combinedScores[i]
            maxScoreIndex = i
    ovoConsistent = maxScoreIndex == predictedLabelIndex

    if ovaConsistent and not ovoConsistent:
        return True
    elif not ovaConsistent and ovoConsistent:
        return False
    elif ovaConsistent and ovoConsistent:
        return None
    else:
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
        tempLabel = negLabel
        negLabel = posLabel
        posLabel = tempLabel

    start = (posLabel * numLabels) - ((posLabel * (posLabel + 1)) // 2)
    offset = negLabel - (posLabel + 1)
    value = oneVOneData[start + offset]
    if flagNegative:
        return 0 - value
    else:
        return value


def scoreModeOutputAdjustment(predLabels, scores, scoreMode, labelOrder):
    """
    Helper to set up the correct output data for different scoreModes in
    the multiclass case.

    Parameters
    ----------
    predLabels : 2d column vector array
        Each row contains a single predicted label.
    scores : 2d array
        Each row corresponds to the confidence. Used to predict the
        corresponding label in predLabels.
    scoreMode : str
        Valued flag determining the output format.
    labelOrder : 1d array
        The ith entry is the label name corresponding to the ith
        confidence value in each row of scores.
    """
    # if 'labels' we just want the predicted labels
    if scoreMode == 'label':
        outData = predLabels
    # in this case we want the first column to be the predicted labels, and
    # the second column to be that label's score
    elif scoreMode == 'bestScore':
        labelToIndexMap = {}
        for i in range(len(labelOrder)):
            ithLabel = labelOrder[i]
            labelToIndexMap[ithLabel] = i
        outData = predLabels
        bestScorePerPrediction = numpy.empty((len(scores), 1))
        for i in range(len(scores)):
            label = predLabels[i, 0]
            index = labelToIndexMap[label]
            matchingScore = scores[i][index]
            bestScorePerPrediction[i] = matchingScore
        outData = numpy.concatenate((outData, bestScorePerPrediction), axis=1)
    else:
        outData = scores

    return outData


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
        else:
            ret = toWrap(*args)
            cache[args] = ret
        return ret

    return wrapped


def collectAttributes(obj, generators, checkers, recursive=True):
    """
    Helper to collect, validate, and return all (relevant) attributes
    associated with a python object (learner, kernel, etc.). The
    returned value will be a dict, mapping names of attribtues to values
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
    if generators is None:
        def wrappedDir(obj):
            ret = {}
            keys = dir(obj)
            for k in keys:
                try:
                    val = getattr(obj, k)
                    ret[k] = val
                # safety against any sort of error someone may have in their
                # property code.
                except Exception:
                    pass
            return ret

        generators = [wrappedDir]

    ret = {}

    for gen in generators:
        possibleDict = gen(obj)
        for possibleName in possibleDict:
            possibleValue = possibleDict[possibleName]
            add = True
            for check in checkers:
                if not check(obj, possibleName, possibleValue):
                    add = False
            if add:
                ret[possibleName] = possibleValue

    return ret


def noLeading__(obj, name, value):
    """
    Determine if a name does NOT begin with two leading underscores.
    """
    if name.startswith('__'):
        return False
    return True


def notCallable(obj, name, value):
    """
    Determine if a value is NOT callable.
    """
    if hasattr(value, '__call__'):
        return False
    return True


def notABCAssociated(obj, name, value):
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

def removeFromDict(orig, toIgnore):
    """
    Remove objects from a dictionary.
    """
    for entry in toIgnore:
        if entry in orig:
            del orig[entry]
    return orig

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
    for i in range(len(full)):
        name = full[i]
        if name in temp:
            retFull.append(name)
            if (i - (len(full) - len(matched))) >= 0:
                retMatched.append(temp[name])

    return (retFull, retMatched)


def modifyImportPath(altDirectory, configSection):
    if altDirectory is not None:
        sys.path.insert(0, altDirectory)
    try:
        location = UML.settings.get(configSection, 'location')
        if location:
            if location not in sys.path:
                sys.path.insert(0, location)
    except configparser.Error:
        pass
