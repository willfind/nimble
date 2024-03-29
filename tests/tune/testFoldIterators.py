
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################


from math import fabs

import numpy as np

import nimble
from nimble.random import numpyRandom
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.exceptions import ImproperObjectAction
from nimble.core.tune import FoldIterator, KFoldIterator, GroupFoldIterator

from tests.helpers import raises, getDataConstructors

################
# FoldIterator #
################
class GenericFolder(FoldIterator):
    """Test the methods defined in the abstract class"""
    def makeFoldList(self):
        return [[0, 1], [2, 3], [4, 5]]

@raises(InvalidArgumentType)
def test_FoldIterator_exceptionDataListNone():
    """ Test FoldIterator() for InvalidArgumentType when dataList is None """
    GenericFolder(None)

@raises(InvalidArgumentValue)
def test_FoldIterator_exceptionDataListEmpty():
    """ Test FoldIterator() for InvalidArgumentType when dataList is None """
    GenericFolder([])

@raises(InvalidArgumentValueCombination)
def test_FoldIterator_exceptionPEmpty():
    """ Test FoldIterator() for InvalidArgumentValueCombination when object is point empty """
    for constructor in getDataConstructors(False):
        data = [[], []]
        data = np.array(data).T
        toTest = constructor(data)
        GenericFolder([toTest])

@raises(InvalidArgumentValueCombination)
def test_FoldIterator_exceptionMixedLengths():
    """ Test KFoldIterator() for InvalidArgumentValueCombination when object is point empty """
    for constructor in getDataConstructors(False):
        dataX = [[1, 2], [3, 4]]
        X = constructor(dataX)
        dataY = [[1], [2], [3], [4]]
        Y = constructor(dataY)
        GenericFolder([X, Y])

def test_FoldIterator_expectedNext():
    """ Test FoldIterator() next is the expected value """
    for constructor in getDataConstructors(False):
        data = [[1], [2], [3], [4], [5], [6]]
        toTest = constructor(data)
        folds = GenericFolder([toTest])
        differentOrder = 0
        for fold in folds:
            trainX, testX = fold[0]
            if sorted(testX) == [1, 2]:
                inTrain = [3, 4, 5, 6]
            elif sorted(testX) == [3, 4]:
                inTrain = [1, 2, 5, 6]
            elif sorted(testX) == [5, 6]:
                inTrain = [1, 2, 3, 4]
            else:
                assert False
            assert sorted(trainX) == inTrain
            if list(trainX) != inTrain:
                differentOrder += 1
        # check that values in the data are randomized, random could be same
        # order but atleast one train set should be different than the original
        assert differentOrder

def test_FoldIterator_multipleDatalists():
    """ Test FoldIterator() for multiple datalists including None """
    for constructor in getDataConstructors(False):
        data = [[1], [2], [3], [4], [5], [6]]
        toTest = constructor(data)
        folds = GenericFolder([toTest, toTest, None, toTest])
        differentOrder = 0
        for fold in folds:
            ((tr1, te1), (tr2, te2), (tr3, te3), (tr4, te4)) = fold
            assert tr1 == tr2 == tr4
            assert te1 == te2 == te4
            assert tr3 is None and te3 is None

#################
# KFoldIterator #
#################
@raises(InvalidArgumentValueCombination)
def test_KFoldIterator_exceptionTooManyFolds():
    """ Test KFoldIterator() for exception when given too many folds """
    for constructor in getDataConstructors(False):
        data = [[1], [2], [3], [4], [5]]
        names = ['col']
        toTest = constructor(data, featureNames=names)
        KFoldIterator([toTest, toTest], 6)
        assert False

def test_KFoldIterator_verifyPartitions():
    """ Test KFoldIterator() yields the correct number folds and partitions the data """
    for constructor in getDataConstructors(False):
        data = [[1], [2], [3], [4], [5]]
        names = ['col']
        toTest = constructor(data, featureNames=names)
        folds = KFoldIterator([toTest], 2)

        [(fold1Train, fold1Test)] = next(folds)
        [(fold2Train, fold2Test)] = next(folds)

        with raises(StopIteration):
            next(folds)

        assert len(fold1Train.points) + len(fold1Test.points) == 5
        assert len(fold2Train.points) + len(fold2Test.points) == 5

        fold1Train.points.append(fold1Test)
        fold2Train.points.append(fold2Test)

def test_KFoldIterator_verifyPartitions_Unsupervised():
    """ Test KFoldIterator() yields the correct number folds and partitions the data, with a None data """
    for constructor in getDataConstructors(False):
        data = [[1], [2], [3], [4], [5]]
        names = ['col']
        toTest = constructor(data, featureNames=names)
        folds = KFoldIterator([toTest, None], 2)

        [(fold1Train, fold1Test), (fold1NoneTrain, fold1NoneTest)] = next(folds)
        [(fold2Train, fold2Test), (fold2NoneTrain, fold2NoneTest)] = next(folds)

        with raises(StopIteration):
            next(folds)

        assert len(fold1Train.points) + len(fold1Test.points) == 5
        assert len(fold2Train.points) + len(fold2Test.points) == 5

        fold1Train.points.append(fold1Test)
        fold2Train.points.append(fold2Test)

        assert fold1NoneTrain is None
        assert fold1NoneTest is None
        assert fold2NoneTrain is None
        assert fold2NoneTest is None


def test_KFoldIterator_verifyMatchups():
    """ Test KFoldIterator() maintains the correct pairings when given multiple data objects """
    for constructor in getDataConstructors(False):
        data0 = [[1], [2], [3], [4], [5], [6], [7]]
        toTest0 = constructor(data0)

        data1 = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]]
        toTest1 = constructor(data1)

        data2 = [[-1], [-2], [-3], [-4], [-5], [-6], [-7]]
        toTest2 = constructor(data2)

        folds = KFoldIterator([toTest0, toTest1, toTest2], 2)

        fold0 = next(folds)
        fold1 = next(folds)
        [(fold0Train0, fold0Test0), (fold0Train1, fold0Test1), (fold0Train2, fold0Test2)] = fold0
        [(fold1Train0, fold1Test0), (fold1Train1, fold1Test1), (fold1Train2, fold1Test2)] = fold1

        with raises(StopIteration):
            next(folds)

        # check that the partitions are the right size (ie, no overlap in training and testing)
        assert len(fold0Train0.points) + len(fold0Test0.points) == 7
        assert len(fold1Train0.points) + len(fold1Test0.points) == 7

        assert len(fold0Train1.points) + len(fold0Test1.points) == 7
        assert len(fold1Train1.points) + len(fold1Test1.points) == 7

        assert len(fold0Train2.points) + len(fold0Test2.points) == 7
        assert len(fold1Train2.points) + len(fold1Test2.points) == 7

        # check that the data is in the same order accross objects, within
        # the training or testing sets of a single fold
        for fold in [fold0, fold1]:
            trainList = []
            testList = []
            for (train, test) in fold:
                trainList.append(train)
                testList.append(test)

            for train in trainList:
                assert len(train.points) == len(trainList[0].points)
                for index in range(len(train.points)):
                    assert fabs(train[index, 0]) == fabs(trainList[0][index, 0])

            for test in testList:
                assert len(test.points) == len(testList[0].points)
                for index in range(len(test.points)):
                    assert fabs(test[index, 0]) == fabs(testList[0][index, 0])

def test_KFoldIterator_foldSizes():
    """ Test that the size of each fold created by KFoldIterator matches expectations """
    for constructor in getDataConstructors(False):
        X = [1, 2]
        data1 = constructor(np.random.random((98, 10)))
        data2 = constructor(np.ones((98, 1)))
        folds = KFoldIterator([data1, data2], 10)

        # first 8 folds should have 10, last 2 folds should have 9
        for i, fold in enumerate(folds):
            ((trainX, testX), (trainY, testY)) = fold
            if i < 8:
                exp = 10
            else:
                exp = 9

            assert len(trainX.points) == 98 - exp
            assert len(testX.points) == exp
            assert len(trainY.points) == 98 - exp
            assert len(testY.points) == exp

#####################
# GroupFoldIterator #
#####################
def test_GroupFoldIterator_foldsAreGroups():
    for constructor in getDataConstructors(False):
        X = constructor(numpyRandom.randint(5, size=(100, 2)))
        Y = X.features.extract(0)
        # set the foldFeature equal to the data
        folds = GroupFoldIterator([X, Y], foldFeature=X)
        for fold in folds:
            ((trainX, testX), (_, _)) = fold
            assert len(trainX.countUniqueElements()) == 4
            assert len(testX.countUniqueElements()) == 1
            assert testX[0, 0] not in iter(trainX)

def test_GroupFoldIterator_ignoresNanPoints():
    for constructor in getDataConstructors(False):
        X = constructor(numpyRandom.randint(5, size=(100, 2)))
        Y = X.features.extract(0)
        X.transformElements(lambda e: e if e else np.nan)
        # set the foldFeature equal to the data
        folds = GroupFoldIterator([X, Y], foldFeature=X)
        for fold in folds:
            ((trainX, testX), (_, _)) = fold
            assert all(v == v for v in iter(trainX))
            assert all(v == v for v in iter(testX))
            trainXElements = trainX.countUniqueElements()
            assert len(trainXElements) == 3
            testXElements = testX.countUniqueElements()
            assert len(testXElements) == 1
            assert testX[0, 0] not in iter(trainX)
