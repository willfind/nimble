
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
Short module demonstrating a call to normalizeData and the effect on the passed
datasets.
"""

import nimble
from nimble import trainAndApply
from nimble import normalizeData

if __name__ == "__main__":

    # we separate into classes accoring to whether x1 is positive or negative
    variables = ["y", "x1", "x2", "x3"]
    data1 = [[1, 6, 0, 0], [1, 3, 0, 0], [0, -5, 0, 0], [0, -3, 0, 0]]
    trainObj = nimble.data(source=data1, featureNames=variables)
    trainObjY = trainObj.features.extract('y')

    # data we're going to classify
    variables2 = ["x1", "x2", "x3"]
    data2 = [[1, 0, 0], [4, 0, 0], [-1, 0, 0], [-2, 0, 0]]
    testObj = nimble.data(source=data2, featureNames=variables2)

    # baseline check
    assert len(trainObj.features) == 3
    assert len(testObj.features) == 3

    # reserve the original data for comparison
    trainObjOrig = trainObj.copy()
    testObjOrig = testObj.copy()

    # use normalize to modify our data; we call a dimentionality reduction algorithm to
    # simply our mostly redundant points. k is the desired number of dimensions in the output
    normTrain, normTest = normalizeData('skl.PCA', trainObj, testX=testObj,
                                        arguments={'n_components': 1})

    # assert that we actually do have fewer dimensions
    assert len(normTest.features) == 1
    assert len(normTest.features) == 1

    # assert we can predict the correct classes
    ret = trainAndApply('nimble.KNNClassifier', normTrain, trainObjY, normTest,
                        arguments={'k': 1})
    assert ret[0, 0] == 1
    assert ret[1, 0] == 1
    assert ret[2, 0] == 0
    assert ret[3, 0] == 0

    # demonstrate that the results have not changed, when compared to the original data;
    # uses python's **kwargs based argument passing.
    retOrig = trainAndApply('nimble.KNNClassifier', trainObjOrig, trainObjY,
                            testObjOrig, k=1)
    assert ret == retOrig
