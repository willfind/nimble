
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

import numpy as np

import nimble
from nimble.learners import RidgeRegression


def testRidgeRegressionShapes():
    """ Test ridge regression by checking the shapes of the inputs and outputs """

    data = [[0, 0, 0], [4, 3, 1], [12, 15, -3], ]
    trainObj = nimble.data(data)

    data2 = [[5.5, 5], [20, -3]]
    testObj = nimble.data(data2)

    for value in ['nimble.RidgeRegression', RidgeRegression]:
        tl = nimble.train(value, trainX=trainObj, trainY=0)
        assert tl.learnerType == 'regression'
        ret = nimble.trainAndApply(value, trainX=trainObj, trainY=0,
                                   testX=testObj, arguments={'lamb': 0})

        assert len(ret.points) == 2
        assert len(ret.features) == 1
        np.testing.assert_approx_equal(ret[0, 0], 10.5, significant=3)
        np.testing.assert_approx_equal(ret[1, 0], 18, significant=2)
