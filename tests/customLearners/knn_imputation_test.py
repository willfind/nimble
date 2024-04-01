
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

"""
KNNImputation tests.
"""

import os

import nimble
from nimble.exceptions import InvalidArgumentValue
from nimble.learners import KNNImputation

from tests.helpers import raises, patch
from tests.helpers import assertNoNamesGenerated
from tests.helpers import getDataConstructors

@raises(InvalidArgumentValue)
def test_KNNImputation_exception_invalidMode():
    data = [[1, 'na', 'x'], [1, 3, 6], [2, 1, 6], [1, 3, 7], ['na', 3, 'x']]
    toTest = nimble.data(data)
    learner = KNNImputation()
    learner.train(toTest, mode='classify')

def test_KNNImputation_classification():
    fNames = ['a', 'b', 'c']
    pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
    data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
    expData = [[1, 3, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
    for constructor in getDataConstructors():
        toTest = constructor(data, pointNames=pNames, featureNames=fNames, useLog=False)
        expTest = constructor(expData, pointNames=pNames, featureNames=fNames, useLog=False)
        learner = KNNImputation()
        learner.train(toTest, k=3, mode='classification')
        assert learner.learnerType == 'transformation'
        ret = learner.apply(toTest)

        assert ret == expTest

def test_KNNImputation_regression():
    fNames = ['a', 'b', 'c']
    pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
    data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
    expData = [[1, 2, 6], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, 6]]
    for constructor in getDataConstructors():
        toTest = constructor(data, pointNames=pNames, featureNames=fNames, useLog=False)
        expTest = constructor(expData, pointNames=pNames, featureNames=fNames, useLog=False)
        learner = KNNImputation()
        learner.train(toTest, k=3, mode='regression')
        ret = learner.apply(toTest)

        assert ret == expTest

@raises(InvalidArgumentValue)
@patch(nimble.learners.knn_imputation, '_sklPresent', lambda: False)
def test_KNNImputation_regression_exception_NoSKL():
    data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
    toTest = nimble.data(data)
    learner = KNNImputation()
    learner.train(toTest, mode='regression')

def test_KNNImputation_lazyNameGeneration():
    data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
    expData = [[1, 3, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
    for constructor in getDataConstructors():
        toTest = constructor(data)
        expTest = constructor(expData)
        learner = KNNImputation()
        learner.train(toTest, mode='classification', k=3)
        ret = learner.apply(toTest)

        assert ret == expTest
        assertNoNamesGenerated(toTest)

def test_KNNImputation_NamePath_preservation():
    data = [[None, None, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    for constructor in getDataConstructors():
        toTest = constructor(data)

        toTest._name = "TestName"
        if isinstance(toTest, nimble.core.data.BaseView):
            toTest._source._absPath = os.path.abspath('TestAbsPath')
            toTest._source._relPath = 'testRelPath'
        else:
            toTest._absPath = os.path.abspath("TestAbsPath")
            toTest._relPath = "testRelPath"

        learner = KNNImputation()
        learner.train(toTest, mode='classification', k=3)
        ret = learner.apply(toTest)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == os.path.abspath("TestAbsPath")
        assert toTest.relativePath == 'testRelPath'
