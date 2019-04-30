"""
Contains the discoverable test object for all classes in the data hierarchy.

Makes use of multiple inheritance to reuse (non-discoverable) test objects
higher in the the test object hierarchy associated with specific portions
of functionality. For example, View objects only inherit from those test
objects associated with non-destructive methods. Furthermore, each of
those higher objects are collections of unit tests generic over a
construction method, which is provided by the discoverable test objects
defined in this file.

"""

from __future__ import absolute_import
import inspect
from functools import wraps

import UML

from .numerical_backend import AllNumerical
from .numerical_backend import NumericalDataSafe

from .query_backend import QueryBackend

from .high_level_backend import HighLevelAll
from .high_level_backend import HighLevelDataSafe

from .low_level_backend import LowLevelBackend

from .structure_backend import StructureAll
from .structure_backend import StructureDataSafe

from .view_access_backend import ViewAccess


class TestListView(HighLevelDataSafe, NumericalDataSafe, QueryBackend,
                   StructureDataSafe, ViewAccess):
    def __init__(self):
        super(TestListView, self).__init__('ListView')


class TestMatrixView(HighLevelDataSafe, NumericalDataSafe, QueryBackend,
                     StructureDataSafe, ViewAccess):
    def __init__(self):
        super(TestMatrixView, self).__init__('MatrixView')


class TestSparseView(HighLevelDataSafe, NumericalDataSafe, QueryBackend,
                     StructureDataSafe, ViewAccess):
    def __init__(self):
        super(TestSparseView, self).__init__('SparseView')


class TestDataFrameView(HighLevelDataSafe, NumericalDataSafe, QueryBackend,
                        StructureDataSafe, ViewAccess):
    def __init__(self):
        super(TestDataFrameView, self).__init__('DataFrameView')


class TestList(HighLevelAll, AllNumerical, QueryBackend, StructureAll):
    def __init__(self):
        super(TestList, self).__init__('List')


class TestMatrix(HighLevelAll, AllNumerical, QueryBackend, StructureAll):
    def __init__(self):
        super(TestMatrix, self).__init__('Matrix')


class TestSparse(HighLevelAll, AllNumerical, QueryBackend, StructureAll):
    def __init__(self):
        super(TestSparse, self).__init__('Sparse')


class TestDataFrame(HighLevelAll, AllNumerical, QueryBackend, StructureAll):
    def __init__(self):
        super(TestDataFrame, self).__init__('DataFrame')


class TestBaseOnly(LowLevelBackend):
    def __init__(self):
        def makeConst(num):
            def const(dummy=2):
                return num

            return const

        def makeAndDefine(pointNames=None, featureNames=None, psize=0, fsize=0):
            """ Make a base data object that will think it has as many features as it has featureNames,
            even though it has no actual data """
            rows = psize if pointNames is None else len(pointNames)
            cols = fsize if featureNames is None else len(featureNames)
            ret = UML.data.Base((rows, cols), pointNames=pointNames, featureNames=featureNames)
            return ret

        self.constructor = makeAndDefine


def getOtherPaths(argList, kwargDict):
    # other object for these functions will always be first positional
    # arg or in kwargs; numeric binary other is not always a UML object
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
        assert isinstance(source, UML.data.Base)
        # store Base arguments for validation after function call
        baseArgs = []
        for argVal in (list(args) + list(kwargs.values())):
            if isinstance(argVal, UML.data.Base):
                baseArgs.append(argVal)
        # name and path preservation
        startName = source._name
        startAbsPath = source._absPath
        startRelPath = source._relPath

        ret = func(self, *args, **kwargs)

        source.validate()
        if isinstance(ret, UML.data.Base):
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


def setClassAttributes(classes, wrapper=None):
    for cls in classes:
        for key, function in objectValidationDict[cls].items():
            if key != 'class':
                if wrapper is not None:
                    function = wrapper(function)
                setattr(objectValidationDict[cls]['class'], key, function)


def startObjectValidation(self):
    classList = ['Base', 'Elements', 'Features', 'Points']
    setClassAttributes(classList, methodObjectValidation)
    for cls in classList:
        # set an attribute to allow tests to check this has been setup
        setattr(objectValidationDict[cls]['class'], 'objectValidation', True)


def stopObjectValidation(self):
    classList = ['Base', 'Elements', 'Features', 'Points']
    setClassAttributes(classList)
    for cls in classList:
        delattr(objectValidationDict[cls]['class'], 'objectValidation')


def addSetupAndTeardown(classList, setup, teardown):
    for cls in classList:
        setattr(cls, 'setUp', setup)
        setattr(cls, 'tearDown', teardown)

objectValidationDict = {}
objectValidationDict['Base'] = objectValidationMethods(UML.data.base.Base)
objectValidationDict['Elements'] = objectValidationMethods(UML.data.elements.Elements)
objectValidationDict['Features'] = objectValidationMethods(UML.data.features.Features)
objectValidationDict['Points'] = objectValidationMethods(UML.data.points.Points)

classesToObjectValidate = [AllNumerical, NumericalDataSafe, QueryBackend,
                           LowLevelBackend, HighLevelAll, HighLevelDataSafe,
                           StructureAll, StructureDataSafe, ViewAccess]

addSetupAndTeardown(classesToObjectValidate, startObjectValidation,
                    stopObjectValidation)
