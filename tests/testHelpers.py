from nose.tools import *

import UML
import random
import numpy

from math import fabs
from UML.exceptions import ArgumentException, ImproperActionException
from UML.umlHelpers import foldIterator

##########
# TESTER #
##########

class FoldIteratorTester(object):
	def __init__(self, constructor):
		self.constructor = constructor

	@raises(ArgumentException)
	def test_foldIterator_exceptionPEmpty(self):
		""" Test foldIterator() for ArgumentException when object is point empty """
		data = [[],[]]
		data = numpy.array(data).T
		toTest = self.constructor(data)
		foldIterator([toTest],2)

#	@raises(ImproperActionException)
#	def test_foldIterator_exceptionFEmpty(self):
#		""" Test foldIterator() for ImproperActionException when object is feature empty """
#		data = [[],[]]
#		data = numpy.array(data)
#		toTest = self.constructor(data)
#		foldIterator([toTest],2)


	@raises(ArgumentException)
	def test_foldIterator_exceptionTooManyFolds(self):
		""" Test foldIterator() for exception when given too many folds """
		data = [[1],[2],[3],[4],[5]]
		names = ['col']
		toTest = self.constructor(data,names)
		foldIterator([toTest, toTest],6)



	def test_foldIterator_verifyPartitions(self):
		""" Test foldIterator() yields the correct number folds and partitions the data """
		data = [[1],[2],[3],[4],[5]]
		names = ['col']
		toTest = self.constructor(data,names)
		folds = foldIterator([toTest],2)

		[(fold1Train, fold1Test)] = folds.next()
		[(fold2Train, fold2Test)] = folds.next()

		try:
			folds.next()
			assert False
		except StopIteration:
			pass

		assert fold1Train.pointCount + fold1Test.pointCount == 5
		assert fold2Train.pointCount + fold2Test.pointCount == 5

		fold1Train.appendPoints(fold1Test)
		fold2Train.appendPoints(fold2Test)



	def test_foldIterator_verifyMatchups(self):
		""" Test foldIterator() maintains the correct pairings when given multiple data objects """
		data0 = [[1],[2],[3],[4],[5],[6],[7]]
		toTest0 = self.constructor(data0)

		data1 = [[1,1],[2,2],[3,3],[4,4],[5,5], [6,6], [7,7]]
		toTest1 = self.constructor(data1)
		
		data2 = [[-1],[-2],[-3],[-4],[-5],[-6],[-7]]
		toTest2 = self.constructor(data2)


		folds = foldIterator([toTest0, toTest1, toTest2], 2)

		fold0 = folds.next()
		fold1 = folds.next()
		[(fold0Train0, fold0Test0), (fold0Train1, fold0Test1), (fold0Train2, fold0Test2)] = fold0
		[(fold1Train0, fold1Test0), (fold1Train1, fold1Test1), (fold1Train2, fold1Test2)] = fold1

		try:
			folds.next()
			assert False
		except StopIteration:
			pass

		# check that the partitions are the right size (ie, no overlap in training and testing)
		assert fold0Train0.pointCount + fold0Test0.pointCount == 7
		assert fold1Train0.pointCount + fold1Test0.pointCount == 7

		assert fold0Train1.pointCount + fold0Test1.pointCount == 7
		assert fold1Train1.pointCount + fold1Test1.pointCount == 7

		assert fold0Train2.pointCount + fold0Test2.pointCount == 7
		assert fold1Train2.pointCount + fold1Test2.pointCount == 7

		# check that the data is in the same order accross objects, within
		# the training or testing sets of a single fold
		for fold in [fold0, fold1]:
			trainList = []
			testList = []
			for (train, test) in fold:
				trainList.append(train)
				testList.append(test)

			for train in trainList:
				assert train.pointCount == trainList[0].pointCount
				for index in xrange(train.pointCount):
					assert fabs(train[index,0]) == fabs(trainList[0][index,0])

			for test in testList:
				assert test.pointCount == testList[0].pointCount
				for index in xrange(test.pointCount):
					assert fabs(test[index,0]) == fabs(testList[0][index,0])



class TestList(FoldIteratorTester):
	def __init__(self):
		def maker(data=None, featureNames=None):
			return UML.createData("List", data=data, featureNames=featureNames)

		super(TestList, self).__init__(maker)


class TestMatrix(FoldIteratorTester):
	def __init__(self):
		def maker(data, featureNames=None):
			return UML.createData("Matrix", data=data, featureNames=featureNames)

		super(TestMatrix, self).__init__(maker)


class TestSparse(FoldIteratorTester):
	def __init__(self):	
		def maker(data, featureNames=None):
			return UML.createData("Sparse", data=data, featureNames=featureNames)

		super(TestSparse, self).__init__(maker)

class TestRand(FoldIteratorTester):
	def __init__(self):	
		def maker(data, featureNames=None):
			possible = ['List', 'Matrix', 'Sparse']
			retType = possible[random.randint(0, 2)]
			return UML.createData(retType=retType, data=data, featureNames=featureNames)

		super(TestRand, self).__init__(maker)
