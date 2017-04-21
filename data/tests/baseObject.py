
import UML
#import numpy

def objConstructorMaker(returnType):
	"""
	Creates the constructor method for a test object, given the return type.

	"""
	def constructor(
			data, pointNames='automatic', featureNames='automatic',
			name=None, path=(None,None)):
		# Case: data is a path to a file
		if isinstance(data, basestring):
			return UML.createData(
				returnType, data=data, pointNames=pointNames,
				featureNames=featureNames, name=name)
		# Case: data is some in-python format. We must call initDataObject
		# instead of createData because we sometimes need to specify a
		# particular path attribute.
		else:
			return UML.helpers.initDataObject(
				returnType, rawData=data, pointNames=pointNames,
				featureNames=featureNames, name=name, path=path,
				keepPoints='all', keepFeatures='all')

	return constructor


def viewConstructorMaker(concreteType):
	"""
	Creates the constructor method for a View test object, given a concrete return type.
	The constructor will create an object with more data than is provided to it,
	and then will take a view which contains the expected values.

	"""
	def constructor(
			data, pointNames='automatic', featureNames='automatic',
			name=None, path=(None,None)):
		# Case: data is a path to a file
		if isinstance(data, basestring):
			orig = UML.createData(
				concreteType, data=data, pointNames=pointNames,
				featureNames=featureNames, name=name)
		# Case: data is some in-python format. We must call initDataObject
		# instead of createData because we sometimes need to specify a
		# particular path attribute.
		else:
			orig = UML.helpers.initDataObject(
				concreteType, rawData=data, pointNames=pointNames,
				featureNames=featureNames, name=name, path=path,
				keepPoints='all', keepFeatures='all')

		# generate points of data to be present before and after the viewable
		# data in the concrete object
		if orig.pointCount != 0:
			firstPRaw = [[0] * orig.featureCount]
			firstPoint = UML.helpers.initDataObject(concreteType, rawData=firstPRaw,
					pointNames=['firstPNonView'], featureNames=orig.getFeatureNames(),
					name=name, path=orig.path, keepPoints='all', keepFeatures='all')

			lastPRaw = [[3] * orig.featureCount]
			lastPoint = UML.helpers.initDataObject(concreteType, rawData=lastPRaw,
					pointNames=['lastPNonView'], featureNames=orig.getFeatureNames(),
					name=name, path=orig.path, keepPoints='all', keepFeatures='all')

			firstPoint.appendPoints(orig)
			full = firstPoint
			full.appendPoints(lastPoint)

			pStart = 1
			pEnd = full.pointCount-2
		else:
			full = orig
			pStart = None
			pEnd = None

		# generate features of data to be present before and after the viewable
		# data in the concrete object
		if orig.featureCount != 0:
			lastFRaw = [[1] * full.pointCount]
			lastFeature = UML.helpers.initDataObject(concreteType, rawData=lastFRaw,
					featureNames=full.getPointNames(), pointNames=['lastFNonView'],
					name=name, path=orig.path, keepPoints='all', keepFeatures='all')
			lastFeature.transpose()

			full.appendFeatures(lastFeature)
			fStart = None
			fEnd = full.featureCount-2
		else:
			fStart = None
			fEnd = None

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
