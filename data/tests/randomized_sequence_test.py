"""
Implementation and support functions for a single randomized
test which performs a sequence of operations on each
implemented type of data representation, comparing both the
results, and an approximately equal representation of their
contained values.

"""

from __future__ import absolute_import
from __future__ import print_function
import inspect
import sys
import pdb
import functools
import numpy
import random

import UML
from UML.data import Base
from UML.data import BaseView
from UML.exceptions import ArgumentException

from UML.randomness import pythonRandom
from six.moves import range


numberOperations = 100
numPoints = 4
numFeatures = 4

unavailableNoPoints = [
    'replaceFeatureWithBinaryFeatures',
    'transformFeatureToIntegers',
    'featureIterator',
    'shufflePoints',
    'shuffleFeatures',
    'writeFile',
    'copyPoints',
    'pointView',
    'elementwiseMultiply',
    'sortPoints',
]

unavailableNoFeatures = [
    'setFeatureName',
    'setFeatureNames',
    'replaceFeatureWithBinaryFeatures',
    'transformFeatureToIntegers',
    'mapReducePoints',
    'pointIterator',
    'shufflePoints',
    'shuffleFeatures',
    'writeFile',
    'copyFeatures',
    'featureView',
    'elementwiseMultiply',
    'sortFeatures',
]

mutuallyExclusiveParams = {}
mutuallyExclusiveParams['sortPoints'] = ('sortBy', 'sortHelper')
mutuallyExclusiveParams['sortFeatures'] = ('sortBy', 'sortHelper')
mutuallyExclusiveParams['extractFeatures'] = ('toExtract', ('start', 'end'))
mutuallyExclusiveParams['extractPoints'] = ('toExtract', ('start', 'end'))
mutuallyExclusiveParams['copyFeatures'] = ('features', ('start', 'end'))
mutuallyExclusiveParams['copyPoints'] = ('points', ('start', 'end'))


def TODO_RandomSequenceOfMethods():
    # always use this number of points and features
    points = numPoints
    features = numFeatures

    # dense int trial
    sparcity = 0.05
    objectList = []
    first = UML.createRandomData('List', points, features, sparcity, elementType='int')
    objectList.append(first)
    objectList.append(first.copyAs(format='Matrix'))
    objectList.append(first.copyAs(format='Sparse'))
    runSequence(objectList)

    ## dense float trial
    sparcity = 0.05
    objectList = []
    first = UML.createRandomData('List', points, features, sparcity, elementType='float')
    objectList.append(first)
    objectList.append(first.copyAs(format='Matrix'))
    objectList.append(first.copyAs(format='Sparse'))
    runSequence(objectList)

    # sparse int trial
    sparcity = 0.9
    objectList = []
    first = UML.createRandomData('List', points, features, sparcity, elementType='int')
    objectList.append(first)
    objectList.append(first.copyAs(format='Matrix'))
    objectList.append(first.copyAs(format='Sparse'))
    runSequence(objectList)

    # sparse float trial
    sparcity = 0.9
    objectList = []
    first = UML.createRandomData('List', points, features, sparcity, elementType='float')
    objectList.append(first)
    objectList.append(first.copyAs(format='Matrix'))
    objectList.append(first.copyAs(format='Sparse'))
    runSequence(objectList)


def runSequence(objectList):
    # setup method list, from Base dir
    availableMethods = list(generators.keys())

    # loop over specified number of operations
    #	for trial in xrange(numberOperations):
    for trial in range(len(availableMethods)):
    # random number as index into available operations list
    #		index = pythonRandom.randint(0,len(availableMethods)-1)
        index = trial
        currFunc = availableMethods[index]
        print(currFunc)
        # exclude operation we know are not runnabel given certain configurations
        if objectList[0].points == 0:
            if currFunc in unavailableNoPoints:
                continue
        if objectList[0].features == 0:
            if currFunc in unavailableNoFeatures:
                continue

            #		if currFunc == 'mapReducePoints':
            #			import pdb
            #			pdb.set_trace()

        # set up parameters
        paramsPerObj = []
        randomseed = UML.randomness.pythonRandom.randint(0, 2**32 - 1)
        for i in range(len(objectList)):
            paramsPerObj.append(makeParams(currFunc, objectList[i], randomseed))

        # call method on each object, collect results
        results = []
        for i in range(len(objectList)):
            funcToCall = getattr(objectList[i], currFunc) #eval('objectList[i].' + currFunc)
            UML.randomness.startAlternateControl(randomseed)
            currResult = funcToCall(*paramsPerObj[i])
            UML.randomness.endAlternateControl()
            results.append(currResult)

        # need to check equality of results
        for i in range(len(results)):
            ithResult = results[i]
            for j in range(i + 1, len(results)):
                jthResult = results[j]
                equalityWrapper(ithResult, jthResult)

        # check approximate equality on each of the objects
        for i in range(len(objectList)):
            ithDataObject = objectList[i]
            for j in range(i + 1, len(objectList)):
                jthDataObject = objectList[j]
                assert ithDataObject.isApproximatelyEqual(jthDataObject)
                assert ithDataObject._equalPointNames(jthDataObject)
                assert ithDataObject._equalFeatureNames(jthDataObject)


def equalityWrapper(left, right):
    """ takes two parameters and uses the most probable equality function to
    assert that they are equal.
    It is important to note that the preference of equality checks is
    strictly ordered: we prefer using isApproximatelyEqual() more than
    isIdentical(), more than equality of contained, iterable objects,
    more than == equality. """

    if isinstance(left, Base):
        assert isinstance(right, Base)
        if left.getTypeString() == right.getTypeString():
            assert left.isIdentical(right)
        assert left.isApproximatelyEqual(right)
    elif isinstance(left, BaseView):
        assert isinstance(right, BaseView)
        assert left.equals(right)
    elif hasattr(left, '__iter__'):
        leftIter = left.__iter__()
        rightIter = right.__iter__()
        leftList = []
        rightList = []
        try:
            while (True): leftList.append(next(leftIter))
        except StopIteration:
            pass
        try:
            while (True): rightList.append(next(rightIter))
        except StopIteration:
            pass

        assert len(leftList) == len(rightList)
        for i in range(len(leftList)):
            currLeft = leftList[i]
            currRight = rightList[i]
            equalityWrapper(currLeft, currRight)
    else:
        assert left == right


def simpleMapper(point):
    idInt = point[0]
    intList = []
    for i in range(0, len(point)):
        intList.append(point[i])
    ret = []
    for value in intList:
        ret.append((idInt, value))
    return ret


def oddOnlyReducer(identifier, valuesList):
    if identifier % 2 == 0:
        return None
    total = 0
    for value in valuesList:
        total += value
    return (identifier, total)


def addingReducer(identifier, valuesList):
    total = 0
    for value in valuesList:
        total += value
    return (identifier, total)


def genObj(dataObj, seed, matchType=True, matchPoints=False, matchFeatures=False):
    random.seed(seed)
    shape = (dataObj.points, dataObj.features)
    if matchType:
        dataType = dataObj.getTypeString()
    else:
        trial = random.randint(0, 2)
        if trial == 0:
            dataType = 'List'
        elif trial == 1:
            dataType = 'Matrix'
        else:
            dataType = 'Sparse'

    points = random.randint(1, numPoints)
    features = random.randint(1, numFeatures)
    if matchPoints:
        points = shape[0]
    if matchFeatures:
        features = shape[1]

    if points == 0 or features == 0:
        rawData = numpy.empty((points, features))
        ret = UML.createData('Matrix', rawData)
        ret = ret.copyAs(dataType)
    else:
        UML.randomness.startAlternateControl()
        UML.setRandomSeed(random.randint(0, 2**32 - 1))
        ret = UML.createRandomData("Matrix", points, features, .5, elementType='int')
        ret = ret.copyAs(dataType)
        UML.randomness.endAlternateControl()
    return ret


genObjMatchFeatures = functools.partial(genObj, matchFeatures=True)
genObjMatchPoints = functools.partial(genObj, matchPoints=True)
genObjMatchShape = functools.partial(genObj, matchPoints=True, matchFeatures=True)
genObjMatchAll = functools.partial(genObj, matchPoints=True, matchFeatures=True)


def genViewFunc(dataObj, seed, onElements=False):
    random.seed(seed)
    toAdd = random.randint(0, 9)
    if onElements:
        def addToElements(element):
            return element + toAdd

        return addToElements
    else:
        def addToView(view):
            ret = []
            for value in view:
                ret.append(value + toAdd)
            return ret

        return addToView


genViewFuncElmts = functools.partial(genViewFunc, onElements=True)


def genID(dataObj, seed, axis):
    random.seed(seed)
    retIntID = random.randint(0, 1)
    if axis == 'point':
        numInAxis = dataObj.points
        source = dataObj.getPointName
    else:
        numInAxis = dataObj.features
        source = dataObj.getFeatureName

    intID = random.randint(0, numInAxis - 1)
    if retIntID:
        ret = intID
    else:
        ret = source(intID)
    assert ret is not None
    if axis == 'point':
        dataObj._getPointIndex(ret)
    else:
        dataObj._getFeatureIndex(ret)
    return ret


def genIDList(dataObj, seed, axis):
    random.seed(seed)
    if axis == 'point':
        numInAxis = dataObj.points
        source = dataObj.getPointName
    else:
        numInAxis = dataObj.features
        source = dataObj.getFeatureName

    numToSample = random.randint(1, numInAxis)
    IDList = random.sample(list(range(numInAxis)), numToSample)
    for i in range(len(IDList)):
        if random.randint(0, 1):
            IDList[i] = source(IDList[i])

    return IDList


genPID = functools.partial(genID, axis='point')
genFID = functools.partial(genID, axis='feature')
genPIDList = functools.partial(genIDList, axis='point')
genFIDList = functools.partial(genIDList, axis='feature')


def genPermArr(dataObj, seed, axis):
    random.seed(seed)
    if axis == 'point':
        numInAxis = dataObj.points
    else:
        numInAxis = dataObj.features

    permArr = random.sample(list(range(numInAxis)), numInAxis)

    return permArr


genPPermArr = functools.partial(genPermArr, axis='point')
genFPermArr = functools.partial(genPermArr, axis='feature')


def genBool(dataObj, seed):
    random.seed(seed)
    if random.randint(0, 1):
        return True
    else:
        return False


def genFalse(dataObj, seed):
    return False


def genTrue(dataObj, seed):
    return True


def genCopyAsFormat(dataObj, seed):
    random.seed(seed)
    poss = ['List', 'Matrix', 'Sparse', 'pythonlist', 'numpyarray', 'numpymatrix',
            'scipycsr', 'scipycsc']
    return poss[random.randint(0, len(poss) - 1)]


def genStartEnd(dataObj, seed, axis):
    random.seed(seed)
    if axis == 'point':
        numInAxis = dataObj.points
    else:
        numInAxis = dataObj.features

    start = random.randint(0, numInAxis - 1)
    end = random.randint(start, numInAxis - 1)
    return (start, end)


genStartEndPoints = functools.partial(genStartEnd, axis='point')
genStartEndFeatures = functools.partial(genStartEnd, axis='feature')


def genNumLimit(dataObj, seed, axis):
    random.seed(seed)
    if axis == 'point':
        numInAxis = dataObj.points
    else:
        numInAxis = dataObj.features

    return random.randint(1, numInAxis - 1)


genPNumLim = functools.partial(genNumLimit, axis='point')
genFNumLim = functools.partial(genNumLimit, axis='faeture')


def checkNameNums(dataObj, axis):
    if axis == 'point':
        source = dataObj.getPointName
        length = dataObj.points
    else:
        source = dataObj.getFeatureName
        length = dataObj.features

    maxNum = 0
    for i in range(length):
        name = source(i)
        currNum = 0
        if name.startswith('FNAME') or name.startswith('PNAME'):
            endJunk = 5
            if '=' in name:
                endJunk = name.index('=') + 1
            currNum = int(float(name[endJunk:]))
        if currNum > maxNum:
            maxNum = currNum
    return maxNum


def genName(dataObj, seed, axis):
    if axis == 'point':
        name = 'PNAME'
    else:
        name = 'FNAME'

    maxID = checkNameNums(dataObj, axis)

    return name + str(maxID + 1)


genPName = functools.partial(genName, axis='point')
genFName = functools.partial(genName, axis='feature')


def genNameList(dataObj, seed, axis):
    if axis == 'point':
        retLen = dataObj.points
        name = 'PNAME'
    else:
        retLen = dataObj.features
        name = 'FNAME'

    ret = []
    if retLen == 0:
        return ret
    first = genName(dataObj, seed, axis)
    next = int(first[5:]) + 1
    ret.append(first)
    for i in range(1, retLen):
        ret.append(name + str(next))
        next += 1

    return ret


def genNameDict(dataObj, seed, axis):
    retList = genNameList(dataObj, seed, axis)
    ret = {}
    for i in range(len(retList)):
        ret[retList[i]] = i
    return ret


def genNewNames(dataObj, seed, axis):
    random.seed(seed)
    coinFlip = random.randint(0, 1)
    if coinFlip:
        genNameList(dataObj, seed, axis)
    else:
        genNameDict(dataObj, seed, axis)


genPNamesNew = functools.partial(genNewNames, axis='point')
genFNamesNew = functools.partial(genNewNames, axis='feature')


def genScorer(dataObj, seed):
    def sumScorer(view):
        total = 0
        for value in view:
            total += value
        return value

    return sumScorer


def genComparator(dataObj, seed):
    def firstValComparator(view):
        if len(view) == 0:
            return None
        else:
            return view[0]

    return firstValComparator


def genMapper(dataObj, seed):
    return simpleMapper


def genReducer(dataObj, seed):
    return oddOnlyReducer


def genProb(dataObj, seed):
    random.seed(seed)
    return random.random()


def genObjName(dataObj, seed):
    random.seed(seed)
    return "NAME:" + str(random.randint(0, 1000))


def genZero(dataObj, seed):
    return 0


def genOne(dataObj, seed):
    return 1


def pickGen(dataObj, seed, genList):
    random.seed(seed)
    picked = random.randint(0, len(genList) - 1)
    return genList[picked](dataObj, seed)


ftp = functools.partial

generators = {'appendFeatures': [genObjMatchPoints],
              'appendPoints': [genObjMatchFeatures],
              'calculateForEachElement': [genViewFuncElmts, ftp(pickGen, genList=(genPID, genPIDList)),
                                          ftp(pickGen, genList=(genFID, genFIDList)), genBool, genBool],
              'calculateForEachFeature': [genViewFunc, genFIDList],
              'calculateForEachPoint': [genViewFunc, genPIDList],
              'containsZero': [],
              'copy': [],
              # TODO !!!! last arg should be bool, but 1d outputs aren't implemented
              # / they're dependent on a raw output in the first arg
              'copyAs': [genCopyAsFormat, genBool, genFalse],
              'copyFeatures': [ftp(pickGen, genList=(genFID, genFIDList)), None, None],
              'copyPoints': [ftp(pickGen, genList=(genPID, genPIDList)), None, None],
              'elementwiseMultiply': [genObjMatchShape],
              #TODO!!!! first arg can also be function!!!!
              'extractFeatures': [genFIDList, None, None, genFNumLim, genBool],
              #TODO!!!! first arg can also be function!!!!
              'extractPoints': [genPIDList, None, None, genPNumLim, genBool],
              'featureView': [genFID],
              'hashCode': [],
              'isApproximatelyEqual': [genObj],
              'isIdentical': [genObj],
              'mapReducePoints': [genMapper, genReducer],
              'pointView': [genPID],
              'referenceDataFrom': [genObjMatchAll],
              'replaceFeatureWithBinaryFeatures': [genFID],
              'setFeatureName': [genFID, genFName],
              'setFeatureNames': [genFNamesNew],
              'setPointName': [genPID, genPName],
              'setPointNames': [genPNamesNew],
              'shuffleFeatures': [genFPermArr],
              'shufflePoints': [genPPermArr],
              'sortFeatures': [genPID, ftp(pickGen, genList=(genScorer, genComparator))],
              'sortPoints': [genFID, ftp(pickGen, genList=(genScorer, genComparator))],
              'transformEachElement': [genViewFuncElmts, ftp(pickGen, genList=(genPID, genPIDList)),
                                       ftp(pickGen, genList=(genFID, genFIDList)), genBool, genBool],
              'transformEachFeature': [genViewFunc, genFIDList],
              'transformEachPoint': [genViewFunc, genPIDList],
              'transformFeatureToIntegers': [genFID],
              'transpose': [],
              'validate': [genOne],
}

untested = ['featureIterator', 'pointIterator', #iterator equality isn't a sensible thing to check
            'writeFile', # lets not test this yet
            'getTypeString', # won't actually be equal
            'summaryReport', # do we really care about testing this?
            'featureReport', # floating point equality errors? / do we care?
            'pointCount', 'featureCount', # not callable
            'toString' # different underlying types will produce different outputs
            'fillUsingPoints', 'fillUsingFeatures', 'fillUsingAllData'
]


def makeParams(funcName, dataObj, seed):
    random.seed(seed)
    generatorList = generators[funcName]
    (args, varargs, keywords, defaults) = inspect.getargspec(getattr(dataObj, funcName))

    #	if funcName.startswith('appendFeatures'):
    #		pdb.set_trace()

    argList = []
    for i in range(len(args) - 1):
        argList.append(None)
    if defaults is None:
        defaults = []

    # +1 because we're ranging over negative indices
    startEnd = None
    if 'feature' in funcName.lower():
        startEnd = genStartEndFeatures(dataObj, seed)
    else:
        startEnd = genStartEndPoints(dataObj, seed)
    # we start by just generating values for everything
    for i in range(len(argList)):
        if generatorList[i] is None:
            if args[i + 1] == 'start':
                toUse = startEnd[0]
            else:
                toUse = startEnd[1]
        else:
            toUse = generatorList[i](dataObj, seed)
        argList[i] = toUse

    # deal with exclusive params by making everything except one
    # thing default
    if funcName in mutuallyExclusiveParams:
        exclusives = mutuallyExclusiveParams[funcName]
        toKeep = random.randint(0, len(exclusives) - 1)
        makeDefault = []
        for i in range(len(exclusives)):
            curr = exclusives[i]
            if i != toKeep:
                if isinstance(curr, tuple):
                    for sub in curr:
                        makeDefault.append(sub)
                else:
                    makeDefault.append(curr)
        for name in makeDefault:
            index = args.index(name)
            argList[index - 1] = defaults[index - (len(args) - len(defaults))]

    return argList


def TODO_GeneratorListSandC():
    data = [[1, 2, 3], [4, 5, 6]]
    dobj = UML.createData('List', data)

    allMethods = dir(Base)
    testable = []
    for name in allMethods:
        if not name.startswith('_') and not name in untested:
            testable.append(name)

    # check that each testable method has generators
    for funcName in testable:
        assert funcName in generators

    for funcName in generators.keys():
        # test that there are generators for each possible argument
        # (except self)
        (a, v, k, d) = inspect.getargspec(getattr(dobj, funcName))
        assert (len(a) - 1) == len(generators[funcName])

        assert funcName in testable


def TODO_MakeParamsExclusivity():
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    dobj = UML.createData('List', data)

    for funcName in mutuallyExclusiveParams.keys():
        # random trials
        for i in range(50):
            genArgs = makeParams(funcName, dobj, UML.randomness.pythonRandom.random())
            (args, v, k, defaults) = inspect.getargspec(getattr(dobj, funcName))
            seenNonDefault = False
            for ex in mutuallyExclusiveParams[funcName]:
                if isinstance(ex, tuple):
                    subSeen = False
                    for subEx in ex:
                        currIndex = args.index(subEx)
                        if genArgs[currIndex - 1] != defaults[currIndex - (len(args) - len(defaults))]:
                            subSeen = True
                    if subSeen:
                        assert seenNonDefault == False
                        seenNonDefault = True
                else:
                    currIndex = args.index(ex)
                    if genArgs[currIndex - 1] != defaults[currIndex - (len(args) - len(defaults))]:
                        assert seenNonDefault == False
                        seenNonDefault = True
                    # at least one of the params must be nondefault in order for the func
                    # to be callable
                    # WRONG! sometimes defaults are acceptable.
                    #assert seenNonDefault

