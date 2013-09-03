
from nose.tools import *

import UML
import random
from math import fabs
from UML.exceptions import ArgumentException, ImproperActionException
from UML.umlHelpers import foldIterator

########################
# Object instantiation #
########################

def listInit(data,featureNames=None):
	return UML.createData(retType='List', data=data, featureNames=featureNames)

def matrixInit(data,featureNames=None):
	return UML.createData(retType='Matrix', data=data, featureNames=featureNames)

def sparseInit(data,featureNames=None):
	return UML.createData(retType='Sparse', data=data, featureNames=featureNames)

def randInit(data,featureNames=None):
	possible = ['List', 'Matrix', 'Sparse']
	retType = possible[random.randint(0, 2)]
	return UML.createData(retType=retType, data=data, featureNames=featureNames)

def callAll(func):
	func(listInit)
	func(matrixInit)
	func(sparseInit)
	func(randInit)


#########
# TESTS #
#########

@raises(ImproperActionException)
def test_foldIterator_exceptionEmpty():
	""" Test foldIterator() for exception when object is empty """
	def tester(constructor):
		data = []
		toTest = constructor(data)
		foldIterator([toTest],2)
	callAll(tester)

@raises(ArgumentException)
def test_foldIterator_exceptionTooManyFolds():
	""" Test foldIterator() for exception when given too many folds """
	def tester(constructor):
		data = [[1],[2],[3],[4],[5]]
		names = ['col']
		toTest = constructor(data,names)
		foldIterator([toTest, toTest],6)
	callAll(tester)


def test_foldIterator_verifyPartitions():
	""" Test foldIterator() yields the correct number folds and partitions the data """
	def tester(constructor):
		data = [[1],[2],[3],[4],[5]]
		names = ['col']
		toTest = constructor(data,names)
		folds = foldIterator([toTest],2)

		[(fold1Train, fold1Test)] = folds.next()
		[(fold2Train, fold2Test)] = folds.next()

		try:
			folds.next()
			assert False
		except StopIteration:
			pass

		assert fold1Train.points() + fold1Test.points() == 5
		assert fold2Train.points() + fold2Test.points() == 5

		fold1Train.appendPoints(fold1Test)
		fold2Train.appendPoints(fold2Test)

		#TODO some kind of rigourous partition check
	callAll(tester)


def test_foldIterator_verifyMatchups():
	""" Test foldIterator() maintains the correct pairings when given multiple data objects """
	def tester(constructor):
		data0 = [[1],[2],[3],[4],[5],[6],[7]]
		toTest0 = constructor(data0)

		data1 = [[1,1],[2,2],[3,3],[4,4],[5,5], [6,6], [7,7]]
		toTest1 = constructor(data1)
		
		data2 = [[-1],[-2],[-3],[-4],[-5],[-6],[-7]]
		toTest2 = constructor(data2)

#		import pdb
#		pdb.set_trace()

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
		assert fold0Train0.points() + fold0Test0.points() == 7
		assert fold1Train0.points() + fold1Test0.points() == 7

		assert fold0Train1.points() + fold0Test1.points() == 7
		assert fold1Train1.points() + fold1Test1.points() == 7

		assert fold0Train2.points() + fold0Test2.points() == 7
		assert fold1Train2.points() + fold1Test2.points() == 7

		# check that the data is in the same order accross objects, within
		# the training or testing sets of a single fold
		for fold in [fold0, fold1]:
			trainList = []
			testList = []
			for (train, test) in fold:
				trainList.append(train)
				testList.append(test)

			for train in trainList:
				assert train.points() == trainList[0].points()
				for index in xrange(train.points()):
					assert fabs(train[index,0]) == fabs(trainList[0][index,0])

			for test in testList:
				assert test.points() == testList[0].points()
				for index in xrange(test.points()):
					assert fabs(test[index,0]) == fabs(testList[0][index,0])

	callAll(tester)

