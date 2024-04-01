
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
fillMatching tests.
"""

import os

import nimble
from nimble import match
from nimble.learners import KNNImputation
from nimble.exceptions import ImproperObjectAction, InvalidArgumentValue

from tests.helpers import raises
from tests.helpers import logCountAssertionFactory
from tests.helpers import assertNoNamesGenerated
from tests.helpers import getDataConstructors

# fillMatching is inplace
constructors = getDataConstructors(includeViews=False)

def test_fillMatching_exception_nansUnmatched():
    raw = [[1, 1, 1, 0], [1, 1, 1, None], [2, 2, 2, 0], [2, 2, 2, 3],
           [2, 2, 2, 4], [2, 2, 2, 4]]
    for constructor in constructors:
        data = constructor(raw)
        with raises(ImproperObjectAction):
            nimble.fillMatching('nimble.KNNImputation', 1, data, mode='classification')

def test_fillMatching_trainXUnaffectedByFailure():
    raw = [[2, 2, 2, 4], [2, 2, 2, 4], [2, 2, 2, 0], [2, 2, 2, 3],
           [2, 2, 2, 4], [2, 2, 2, 4]]
    for constructor in constructors:
        data = constructor(raw)
        dataCopy = data.copy()
        # trying to fill 2 will fail because the training data will be empty
        with raises(InvalidArgumentValue):
            nimble.fillMatching('nimble.KNNImputation', 2, data, mode='classification')

@logCountAssertionFactory(len(nimble.core.data.available) * 2)
def backend_fillMatching(matchingElements, raw, expRaw):
    for constructor in constructors:
        data = constructor(raw, useLog=False)
        exp = constructor(expRaw, useLog=False)
        for value in ['nimble.KNNImputation', KNNImputation]:
            nimble.fillMatching(value, matchingElements, data,
                                mode='classification', k=1)
            assert data == exp

def test_fillMatching_matchingElementsAsSingleValue():
    matchingElements = 0
    raw = [[1, 1, 1, 0], [1, 1, 1, 1], [2, 2, 2, 0], [2, 2, 2, 4]]
    expRaw = [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 4], [2, 2, 2, 4]]
    backend_fillMatching(matchingElements, raw, expRaw)

def test_fillMatching_matchingElementsAsList():
    matchingElements = [-1, -2]
    raw = [[1, 1, 1, -1], [1, 1, 1, 1], [2, 2, 2, -2], [2, 2, 2, 4]]
    expRaw = [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 4], [2, 2, 2, 4]]
    backend_fillMatching(matchingElements, raw, expRaw)

def test_fillMatching_matchingElementsAsFunction():
    matchingElements = match.negative
    raw = [[1, 1, 1, -1], [1, 1, 1, 1], [2, 2, 2, -2], [2, 2, 2, 4]]
    expRaw = [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 4], [2, 2, 2, 4]]
    backend_fillMatching(matchingElements, raw, expRaw)

def test_fillMatching_matchingElementsAsBooleanMatrix():
    raw = [[1, 1, 1, -1], [1, 1, 1, 1], [2, 2, 2, -2], [2, 2, 2, 4]]
    obj = nimble.data(raw)
    matchingElements = obj.matchingElements(match.negative)
    expRaw = [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 4], [2, 2, 2, 4]]
    backend_fillMatching(matchingElements, raw, expRaw)

@raises(InvalidArgumentValue)
def test_fillMatching_matchingElementsAsBooleanMatrix_exception_wrongSize():
    raw = [[1, 1, 1, -1], [1, 1, 1, 1], [2, 2, 2, -2], [2, 2, 2, 4]]
    obj = nimble.data(raw)
    # reshaped to cause failure
    matchingElements = obj.matchingElements(match.negative)[:2, :2]
    data = nimble.data(raw, useLog=False)
    nimble.fillMatching('nimble.KNNImputation', matchingElements, data,
                        mode='classification', k=1)

@raises(InvalidArgumentValue)
def test_KNNImputation_exception_invalidMode():
    data = [[1, 'na', 'x'], [1, 3, 6], [2, 1, 6], [1, 3, 7], ['na', 3, 'x']]
    toTest = nimble.data(data)
    nimble.fillMatching('nimble.KNNImputation', match.nonNumeric, toTest,
                        k=3, mode='classify')


def test_fillMatching_pointsLimited():
    fNames = ['a', 'b', 'c']
    pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
    data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
    expData = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
    for constructor in constructors:
        toTest = constructor(data, pointNames=pNames, featureNames=fNames)
        expTest = constructor(expData, pointNames=pNames, featureNames=fNames)
        nimble.fillMatching('nimble.KNNImputation', match.missing, toTest,
                            points=[2, 3, 4], mode='classification', k=3)
        assert toTest == expTest

@raises(InvalidArgumentValue)
def test_fillMatching_sklDisallowedArgument():
    fNames = ['a', 'b', 'c']
    pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
    data = [[1, 0, 0], [1, 3, 6], [2, 1, 6], [1, 3, 7], [0, 3, 0]]
    toTest = nimble.data(data, pointNames=pNames, featureNames=fNames)
    nimble.fillMatching('skl.SimpleImputer', match.zero, toTest,
                        missing_values=0)

def test_fillMatching_featuresLimited():
    fNames = ['a', 'b', 'c']
    pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
    data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
    expData = [[1, 3, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, None]]
    for constructor in constructors:
        toTest = constructor(data, pointNames=pNames, featureNames=fNames)
        expTest = constructor(expData, pointNames=pNames, featureNames=fNames)
        nimble.fillMatching('nimble.KNNImputation', match.missing, toTest,
                            features=[1,0], mode='classification', k=3)
        assert toTest == expTest

def test_fillMatching_pointsFeaturesLimited():
    fNames = ['a', 'b', 'c']
    pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
    data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
    expData = [[1, None, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
    for constructor in constructors:
        toTest = constructor(data, pointNames=pNames, featureNames=fNames)
        expTest = constructor(expData, pointNames=pNames, featureNames=fNames)
        nimble.fillMatching('nimble.KNNImputation', match.missing, toTest,
                            points=0, features=2, mode='classification', k=3)
        assert toTest == expTest

def test_fillMatching_lazyNameGeneration():
    data = [[1, 'na', 'x'], [1, 3, 6], [2, 1, 6], [1, 3, 7], ['na', 3, 'x']]
    expData = [[1, 3, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
    for constructor in getDataConstructors(includeViews=False, includeSparse=False):
        toTest = constructor(data)
        expTest = constructor(expData)
        nimble.fillMatching('nimble.KNNImputation', match.nonNumeric, toTest,
                            k=3, mode='classification')

        assert toTest == expTest
        assertNoNamesGenerated(toTest)

def test_fillMatching_NamePath_preservation():
    data = [[None, None, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    for constructor in constructors:
        toTest = constructor(data)

        toTest._name = "TestName"
        toTest._absPath = os.path.abspath("TestAbsPath")
        toTest._relPath = "testRelPath"

        nimble.fillMatching('nimble.KNNImputation', match.missing, toTest,
                            k=3, mode='regression')

        assert toTest.name == "TestName"
        assert toTest.absolutePath == os.path.abspath("TestAbsPath")
        assert toTest.relativePath == 'testRelPath'
