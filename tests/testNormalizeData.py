
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
Tests for the top level function nimble.normalizeData
"""

import nimble
from tests.helpers import logCountAssertionFactory

# successful run no testX
def test_normalizeData_successTest_noTestX():
    data = [[0, 1, 3], [-1, 1, 2], [1, 2, 2]]
    trainX = nimble.data(data)
    orig = trainX.copy()

    norm = nimble.normalizeData('scikitlearn.PCA', trainX, n_components=2)

    assert norm != trainX
    assert trainX == orig

# successful run trainX and testX
def test_normalizeData_successTest_BothDataSets():
    learners = ['scikitlearn.PCA', 'scikitlearn.StandardScaler']
    for learner, args in zip(learners, [{'n_components': 2}, {}]):
        data1 = [[0, 1, 3], [-1, 1, 2], [1, 2, 2]]
        ftNames = ['a', 'b', 'c']
        trainX = nimble.data(data1, pointNames=['0', '1', '2'],
                             featureNames=ftNames)

        data2 = [[-1, 0, 5]]
        testX = nimble.data(data2, pointNames=['4'],
                            featureNames=ftNames)

        norms = nimble.normalizeData(learner, trainX, testX=testX,
                                     arguments=args)
        assert isinstance(norms, tuple)

        normTrainX, normTestX = norms
        assert normTrainX != trainX
        assert normTestX != testX

        # pointNames should be preserved for both learners
        # featureNames not preserved when number of features changes (PCA)
        assert normTrainX.points.getNames() == trainX.points.getNames()
        if learner == 'scikitlearn.PCA':
            assert not normTrainX.features._namesCreated()
            assert len(normTrainX.features) == 2
        else:
            assert normTrainX.features.getNames() == trainX.features.getNames()

        assert normTestX.points.getNames() == testX.points.getNames()
        if learner == 'scikitlearn.PCA':
            assert not normTestX.features._namesCreated()
            assert len(normTestX.features) == 2
        else:
            assert normTestX.features.getNames() == testX.features.getNames()

def test_normalizeData_returnNamesDefault():
    data1 = [[0, 1, 3], [-1, 1, 2], [1, 2, 2]]
    trainX = nimble.data(data1, name='trainX')

    data2 = [[-1, 0, 5]]
    testX = nimble.data(data2, name='testX')

    norms = nimble.normalizeData('scikitlearn.PCA', trainX, testX=testX,
                                 n_components=2)

    assert norms[0].name is None
    assert norms[1].name is None

@logCountAssertionFactory(2)
def test_normalizeData_logCount():
    data1 = [[0, 1, 3], [-1, 1, 2], [1, 2, 2]]
    trainX = nimble.data(data1, useLog=False)
    data2 = [[-1, 0, 5]]
    testX = nimble.data(data2, useLog=False)

    _ = nimble.normalizeData('scikitlearn.StandardScaler', trainX, testX=testX)
    _ = nimble.normalizeData('scikitlearn.PCA', trainX, testX=testX,
                             n_components=2)
