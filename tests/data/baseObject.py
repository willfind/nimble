from __future__ import absolute_import

import six
import numpy

import UML

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
            return UML.createData(
                returnType, data=data, pointNames=pointNames,
                featureNames=featureNames, name=name, treatAsMissing=treatAsMissing,
                replaceMissingWith=replaceMissingWith, elementType=elementType,
                useLog=False)
        # Case: data is some in-python format. We must call initDataObject
        # instead of createData because we sometimes need to specify a
        # particular path attribute.
        else:
            return UML.createData(returnType, data=data, pointNames=pointNames,
                featureNames=featureNames, elementType=elementType, name=name, path=path,
                keepPoints='all', keepFeatures='all', treatAsMissing=treatAsMissing,
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
            orig = UML.createData(
                concreteType, data=data, pointNames=pointNames,
                featureNames=featureNames, name=name, treatAsMissing=treatAsMissing,
                replaceMissingWith=replaceMissingWith, elementType=elementType,
                useLog=False)
        # Case: data is some in-python format. We must call initDataObject
        # instead of createData because we sometimes need to specify a
        # particular path attribute.
        else:
            orig = UML.helpers.initDataObject(
                concreteType, rawData=data, pointNames=pointNames,
                featureNames=featureNames, name=name, path=path, elementType=elementType,
                keepPoints='all', keepFeatures='all', treatAsMissing=treatAsMissing,
                replaceMissingWith=replaceMissingWith)
        origHasPts = orig.points._namesCreated()
        origHasFts = orig.features._namesCreated()
        # generate points of data to be present before and after the viewable
        # data in the concrete object
        if len(orig.points) != 0:
            firstPRaw = [[0] * len(orig.features)]
            fNamesParam = orig.features.getNames() if orig.features._namesCreated() else 'automatic'
            firstPoint = UML.helpers.initDataObject(concreteType, rawData=firstPRaw,
                                                    pointNames=['firstPNonView'], featureNames=fNamesParam,
                                                    name=name, path=orig.path, keepPoints='all', keepFeatures='all',
                                                    elementType=elementType)

            lastPRaw = [[3] * len(orig.features)]
            lastPoint = UML.helpers.initDataObject(concreteType, rawData=lastPRaw,
                                                   pointNames=['lastPNonView'], featureNames=fNamesParam,
                                                   name=name, path=orig.path, keepPoints='all', keepFeatures='all',
                                                   elementType=elementType)

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
            fNames = full.points.getNames() if full.points._namesCreated() else 'automatic'
            lastFeature = UML.helpers.initDataObject(concreteType, rawData=lastFRaw,
                                                     featureNames=fNames, pointNames=['lastFNonView'],
                                                     name=name, path=orig.path, keepPoints='all', keepFeatures='all',
                                                     elementType=elementType)

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
