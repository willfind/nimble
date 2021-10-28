import inspect
from functools import wraps

import numpy as np

import nimble
from nimble.core._createHelpers import DEFAULT_MISSING

def objConstructorMaker(returnType):
    """
    Creates the constructor method for a test object, given the return type.
    """

    def constructor(
            source, pointNames='automatic', featureNames='automatic',
            name=None, convertToType=None, treatAsMissing=DEFAULT_MISSING,
            replaceMissingWith=np.nan, rowsArePoints=True, paths=(None, None)):
        # Case: source is a path to a file
        if isinstance(source, str):
            return nimble.data(
                returnType, source=source, pointNames=pointNames,
                featureNames=featureNames, name=name,
                convertToType=convertToType, treatAsMissing=treatAsMissing,
                replaceMissingWith=replaceMissingWith,
                rowsArePoints=rowsArePoints, useLog=False)
        # Case: source is some in-python format. We must call initDataObject
        # instead of nimble.data because we sometimes need to specify a
        # particular path attribute.
        return nimble.core._createHelpers.initDataObject(
            returnType, source, pointNames=pointNames,
            featureNames=featureNames, name=name,
            convertToType=convertToType, treatAsMissing=treatAsMissing,
            replaceMissingWith=replaceMissingWith, rowsArePoints=rowsArePoints,
            paths=paths)

    return constructor


def viewConstructorMaker(concreteType):
    """
    Creates the constructor method for a View test object, given a concrete
    return type. The constructor will create an object with more data than is
    provided to it, and then will take a view which contains the expected
    values.
    """

    def constructor(
            source, pointNames='automatic', featureNames='automatic',
            name=None, convertToType=None, treatAsMissing=DEFAULT_MISSING,
            replaceMissingWith=np.nan, rowsArePoints=True, paths=(None, None)):
        construct = objConstructorMaker(concreteType)
        orig = construct(source, pointNames, featureNames, name, convertToType,
                         treatAsMissing, replaceMissingWith, rowsArePoints,
                         paths)
        origHasPts = orig.points._namesCreated()
        origHasFts = orig.features._namesCreated()
        # generate points of data to be present before and after the viewable
        # data in the concrete object
        origPaths = (orig.absolutePath, orig.relativePath)
        if len(orig.points) != 0:
            firstPRaw = np.zeros([1] + orig._dims[1:], dtype=int).tolist()
            fNamesParam = orig.features._getNamesNoGeneration()
            firstPoint = nimble.core._createHelpers.initDataObject(
                concreteType, rawData=firstPRaw, pointNames=['firstPNonView'],
                featureNames=fNamesParam, name=name,
                convertToType=convertToType, paths=origPaths)

            lastPRaw = (np.ones([1] + orig._dims[1:], dtype=int) * 3).tolist()
            lastPoint = nimble.core._createHelpers.initDataObject(
                concreteType, rawData=lastPRaw, pointNames=['lastPNonView'],
                featureNames=fNamesParam, name=name,
                convertToType=convertToType, paths=origPaths)

            firstPoint.points.append(orig, useLog=False)
            full = firstPoint
            full.points.append(lastPoint, useLog=False)

            pStart = 1
            pEnd = len(full.points) - 2
        else:
            full = orig
            pStart = None
            pEnd = None

        # generate features of data to be present before and after the viewable
        # data in the concrete object
        if len(orig.features) != 0 and not len(orig._dims) > 2:
            lastFRaw = [[1] * len(full.points)]
            fNames = full.points._getNamesNoGeneration()
            lastFeature = nimble.core._createHelpers.initDataObject(
                concreteType, rawData=lastFRaw, featureNames=fNames,
                pointNames=['lastFNonView'], name=name,
                convertToType=convertToType, paths=origPaths)

            lastFeature.transpose(useLog=False)

            full.features.append(lastFeature, useLog=False)
            fStart = None
            fEnd = len(full.features) - 2
        else:
            fStart = None
            fEnd = None

        if not origHasPts:
            full.points.setNames(None, useLog=False)
        if not origHasFts:
            full.features.setNames(None, useLog=False)

        ret = full.view(pStart, pEnd, fStart, fEnd)
        ret.name = orig.name

        return ret

    return constructor


class DataTestObject(object):

    def setup_method(self):
        startObjectValidation()

    def teardown_method(self):
        stopObjectValidation()


def getOtherPaths(argList, kwargDict):
    # other object for these functions will always be first positional
    # arg or in kwargs; numeric binary other is not always a nimble object
    if argList and hasattr(argList[0], '_absPath'):
        otherAbsPath = argList[0]._absPath
    elif 'other' in kwargDict and hasattr(kwargDict['other'], '_absPath'):
        otherAbsPath = kwargDict['other']._absPath
    else:
        otherAbsPath = None

    if argList and hasattr(argList[0], '_relPath'):
        otherRelPath = argList[0]._relPath
    elif 'other' in kwargDict and hasattr(kwargDict['other'], '_relPath'):
        otherRelPath = kwargDict['other']._relPath
    else:
        otherRelPath = None

    return otherAbsPath, otherRelPath


def methodObjectValidation(func):
    @wraps(func)
    def wrapped(self, *args, **kwargs):
        if hasattr(self, '_base'):
            source = self._base
        else:
            source = self
        assert isinstance(source, nimble.core.data.Base)
        # store Base arguments for validation after function call
        baseArgs = []
        for argVal in (list(args) + list(kwargs.values())):
            if isinstance(argVal, nimble.core.data.Base):
                baseArgs.append(argVal)
        # name and path preservation
        startName = source._name
        startAbsPath = source._absPath
        startRelPath = source._relPath

        ret = func(self, *args, **kwargs)
        source.validate()
        if isinstance(ret, nimble.core.data.Base):
            ret.validate()
        for arg in baseArgs:
            arg.validate()

        assert source._name == startName
        funcName = func.__name__
        inplaceNumeric = ['__iadd__', '__isub__', '__imul__', '__idiv__',
                          '__ifloordiv__', '__itruediv__', '__imod__',
                          '__imatmul__']

        finalAbsPath = startAbsPath
        finalRelPath = startRelPath
        # _referenceFrom always gets path from other object, inplace numeric
        # binary will follow _dataHelpers.binaryOpNamePathMerge logic
        if funcName == '_referenceFrom':
            finalAbsPath, finalRelPath = getOtherPaths(args, kwargs)
        elif funcName in inplaceNumeric:
            otherAbsPath, otherRelPath = getOtherPaths(args, kwargs)
            if startAbsPath is None and otherAbsPath is not None:
                finalAbsPath = otherAbsPath
            elif startAbsPath is not None and otherAbsPath is not None:
                finalAbsPath = None
            if startRelPath is None and otherRelPath is not None:
                finalRelPath = otherRelPath
            elif startRelPath is not None and otherRelPath is not None:
                finalRelPath = None

        assert source._absPath == finalAbsPath
        assert source._relPath == finalRelPath

        return ret
    return wrapped


def objectValidationMethods(cls):
    methodMap = {'class': cls}
    for attr in cls.__dict__:
        if inspect.isfunction(getattr(cls, attr)):
            func = getattr(cls, attr)
            # ignore functions that interfere with __init__ or recurse
            # because they are used in validate
            ignore = ['__init__', 'validate', 'getTypeString', 'setNames',
                      'getName', 'getNames', 'getIndex', '__len__',
                      '__getitem__'] # __getitem__ ignored for efficiency
            if (func.__name__ not in ignore and
                    (not func.__name__.startswith('_')
                     or func.__name__.startswith('__'))):
                methodMap[attr] = func
    return methodMap


objectValidationDict = {}
objectValidationDict['Base'] = objectValidationMethods(nimble.core.data.base.Base)
objectValidationDict['Features'] = objectValidationMethods(nimble.core.data.features.Features)
objectValidationDict['Points'] = objectValidationMethods(nimble.core.data.points.Points)


def setClassAttributes(classes, wrapper=None):
    for cls in classes:
        for key, function in objectValidationDict[cls].items():
            if key != 'class':
                if wrapper is not None:
                    function = wrapper(function)
                setattr(objectValidationDict[cls]['class'], key, function)


def startObjectValidation():
    classList = ['Base', 'Features', 'Points']
    setClassAttributes(classList, methodObjectValidation)
    for cls in classList:
        # set an attribute to allow tests to check this has been setup
        setattr(objectValidationDict[cls]['class'], 'objectValidation', True)


def stopObjectValidation():
    classList = ['Base', 'Features', 'Points']
    setClassAttributes(classList)
    for cls in classList:
        delattr(objectValidationDict[cls]['class'], 'objectValidation')
