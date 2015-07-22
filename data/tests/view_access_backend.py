"""
Tests of the object oriented access methods for Views which are
limited to only a portion of the data in the wrapped concrete
object.

Methods tested in this file:


"""

import UML
from UML.data.tests.baseObject import DataTestObject


class ViewAccess(DataTestObject):

	def backend(self, pStart, pEnd, fStart, fEnd):
		pSize = pEnd - pStart
		fSize = fEnd - fStart

		data = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
		pnames = ['p1', 'p2', 'p3']
		fnames = ['f1', 'f2', 'f3']

		# Should remove the "View" off the end of, for example "ListView"
		origType = self.returnType[:-(len("View"))]
		orig = UML.createData(origType, data, pointNames=pnames, featureNames=fnames)
		test = orig.view(pointStart=pStart, pointEnd=pEnd, featureStart=fStart,
				featureEnd=fEnd)

		def checkAccess(v, recurse):
			for i in xrange(pSize):
				for j in xrange(fSize):
					assert v[i,j] == orig[i+pStart, j+fStart]

			# getPointNames 
			assert v.getPointNames() == pnames[pStart:pEnd]

			# getPointIndex, getPointName
			for i in xrange(pStart, pEnd):
				origName = pnames[i]
				assert v.getPointName(i) == origName
				assert v.getPointIndex(origName) == i

			# getFeatureNames
			assert v.getFeatureNames() == fnames[fStart:fEnd]

			# getFeatureIndex, getFeatureName
			for i in xrange(fStart, fEnd):
				origName = fnames[i]
				assert v.getFeatureName(i) == origName
				assert v.getFeatureIndex(origName) == i

			# view
			if recurse:
				checkAccess(v, False)

		checkAccess(test, True)


	def test_ViewAccess_AllLimits(self):
		# the backed method uses data with only 3 points and features
		cap = 4

		for pStart in xrange(cap):
			for pEnd in xrange(pStart, cap):
				for fStart in xrange(cap):
					for fEnd in xrange(fStart, cap):
						self.backend(pStart, pEnd, fStart, fEnd)
						


# get retType TODO -- still not sure what is correct functionality.
