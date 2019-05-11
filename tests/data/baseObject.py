from __future__ import absolute_import
import inspect
from functools import wraps

import six
import numpy

import UML as nimble

def objConstructorMaker(returnType):
    """
    Creates the constructor method for a test object, given the return type.

    """

    def constructor(
            data, pointNames='automatic', featureNames='automatic', elementType=None,
            name=None, path=(None, None),
            treatAsMissing=[float('nan'), numpy.nan, None, '', 'None', 'nan'],
            replaceMissingWith=numpy.nan):
        # Case: data is a path to a file
        if isinstance(data, six.string_types):
            return nimble.createData(
                returnType, data=data, pointNames=pointNames,
                featureNames=featureNames, name=name,
                treatAsMissing=treatAsMissing,
                replaceMissingWith=replaceMissingWith, elementType=elementType,
                useLog=False)
        # Case: data is some in-python format. We must call initDataObject
        # instead of createData because we sometimes need to specify a
        # particular path attribute.
        else:
            return nimble.createData(
                returnType, data=data, pointNames=pointNames,
                featureNames=featureNames, elementType=elementType, name=name,
                path=path, keepPoints='all', keepFeatures='all',
                treatAsMissing=treatAsMissing,
                replaceMissingWith=replaceMissingWith, useLog=False)

    return constructor


def viewConstructorMaker(concreteType):
    """
    Creates the constructor method for a View test object, given a concrete return type.
    The constructor will create an object with more data than is provided to it,
    and then will take a view which contains the expected values.

    """

    def constructor(
            data, pointNames='automatic', featureNames='automatic',
            name=None, path=(None, None), elementType=None,
            treatAsMissing=[float('nan'), numpy.nan, None, '', 'None', 'nan'],
            replaceMissingWith=numpy.nan):
        # Case: data is a path to a file
        if isinstance(data, six.string_types):
            orig = nimble.createData(
                concreteType, data=data, pointNames=pointNames,
                featureNames=featureNames, name=name,
                treatAsMissing=treatAsMissing,
                replaceMissingWith=replaceMissingWith, elementType=elementType,
                useLog=False)
        # Case: data is some in-python format. We must call initDataObject
        # instead of createData because we sometimes need to specify a
        # particular path attribute.
        else:
            orig = nimble.helpers.initDataObject(
                concreteType, rawData=data, pointNames=pointNames,
                featureNames=featureNames, name=name, path=path,
                elementType=elementType, keepPoints='all', keepFeatures='all',
                treatAsMissing=treatAsMissing,
                replaceMissingWith=replaceMissingWith)
        origHasPts = orig.points._namesCreated()
        origHasFts = orig.features._namesCreated()
        # generate points of data to be present before and after the viewable
        # data in the concrete object
        if len(orig.points) != 0:
            firstPRaw = [[0] * len(orig.features)]
            fNamesParam = orig.features._getNamesNoGeneration()
            firstPoint = nimble.helpers.initDataObject(
                concreteType, rawData=firstPRaw, pointNames=['firstPNonView'],
                featureNames=fNamesParam, name=name, path=orig.path,
                keepPoints='all', keepFeatures='all', elementType=elementType)

            lastPRaw = [[3] * len(orig.features)]
            lastPoint = nimble.helpers.initDataObject(
                concreteType, rawData=lastPRaw, pointNames=['lastPNonView'],
                featureNames=fNamesParam, name=name, path=orig.path,
                keepPoints='all', keepFeatures='all', elementType=elementType)

            firstPoint.points.add(orig, useLog=False)
            full = firstPoint
            full.points.add(lastPoint, useLog=False)

            pStart = 1
            pEnd = len(full.points) - 2
        else:
            full = orig
            pStart = None
            pEnd = None

        # generate features of data to be present before and after the viewable
        # data in the concrete object
        if len(orig.features) != 0:
            lastFRaw = [[1] * len(full.points)]
            fNames = full.points._getNamesNoGeneration()
            lastFeature = nimble.helpers.initDataObject(
                concreteType, rawData=lastFRaw, featureNames=fNames,
                pointNames=['lastFNonView'], name=name, path=orig.path,
                keepPoints='all', keepFeatures='all', elementType=elementType)

            lastFeature.transpose(useLog=False)

            full.features.add(lastFeature, useLog=False)
            fStart = None
            fEnd = len(full.features) - 2
        else:
            fStart = None
            fEnd = None

        if not origHasPts:
            full.points.setNames(None)
        if not origHasFts:
            full.features.setNames(None)

        ret = full.view(pStart, pEnd, fStart, fEnd)
        ret._name = orig.name

        return ret

    return constructor


class DataTestObject(object):
    def __init__(self, returnType):
        if returnType.endswith("View"):
            self.constructor = viewConstructorMaker(returnType[:-len("View")])
        else:
            self.constructor = objConstructorMaker(returnType)

        self.returnType = returnType


    def setUp(self):
        startObjectValidation()

    def tearDown(self):
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
        if hasattr(self, '_source'):
            source = self._source
        else:
            source = self
        assert isinstance(source, nimble.data.Base)
        # store Base arguments for validation after function call
        baseArgs = []
        for argVal in (list(args) + list(kwargs.values())):
            if isinstance(argVal, nimble.data.Base):
                baseArgs.append(argVal)
        # name and path preservation
        startName = source._name
        startAbsPath = source._absPath
        startRelPath = source._relPath

        ret = func(self, *args, **kwargs)

        source.validate()
        if isinstance(ret, nimble.data.Base):
            ret.validate()
        for arg in baseArgs:
            arg.validate()

        assert source._name == startName
        funcName = func.__name__
        inplaceNumeric = ['__iadd__', '__isub__', '__imul__', '__idiv__',
                          '__ifloordiv__', '__itruediv__', '__imod__']

        finalAbsPath = startAbsPath
        finalRelPath = startRelPath
        # referenceDataFrom always gets path from other object, inplace numeric
        # binary will follow dataHelpers.binaryOpNamePathMerge logic
        if funcName == 'referenceDataFrom':
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
objectValidationDict['Base'] = objectValidationMethods(nimble.data.base.Base)
objectValidationDict['Elements'] = objectValidationMethods(nimble.data.elements.Elements)
objectValidationDict['Features'] = objectValidationMethods(nimble.data.features.Features)
objectValidationDict['Points'] = objectValidationMethods(nimble.data.points.Points)


def setClassAttributes(classes, wrapper=None):
    for cls in classes:
        for key, function in objectValidationDict[cls].items():
            if key != 'class':
                if wrapper is not None:
                    function = wrapper(function)
                setattr(objectValidationDict[cls]['class'], key, function)


def startObjectValidation():
    classList = ['Base', 'Elements', 'Features', 'Points']
    setClassAttributes(classList, methodObjectValidation)
    for cls in classList:
        # set an attribute to allow tests to check this has been setup
        setattr(objectValidationDict[cls]['class'], 'objectValidation', True)


def stopObjectValidation():
    classList = ['Base', 'Elements', 'Features', 'Points']
    setClassAttributes(classList)
    for cls in classList:
        delattr(objectValidationDict[cls]['class'], 'objectValidation')
