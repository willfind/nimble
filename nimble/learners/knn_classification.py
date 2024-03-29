
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
For a simple classifier you could implement a k-nearest neighbors
classifier. For any point you want to predict, just compute the
Euclidean distance of that point to every point in the training set.
Then sort this list. Take the k points that have the lowest distance,
and for your prediction use whichever label occurs most (among those k
training points with lowest distances to the point you are predicting).
If there is a tie, use k=1
"""

import numpy as np

from nimble import CustomLearner
from nimble.core._createHelpers import initDataObject
from nimble._utility import dtypeConvert

# pylint: disable=attribute-defined-outside-init, arguments-differ
class KNNClassifier(CustomLearner):
    """
    K-Nearest Neighbors Classifier using euclidean distance.
    """
    learnerType = 'classification'

    def train(self, trainX, trainY, k=5):
        self.k = k
        self._trainX = trainX
        self._trainY = trainY

    def apply(self, testX):
        dists = self._getDistanceMatrix(testX)

        predictions = []
        for point in dists:
            ordered, votes = self._kNeighborOrderedLabelsAndVotes(point)
            highCount = 0
            bestLabels = []
            for label, count in votes.items():
                if count > highCount:
                    bestLabels = [label]
                    highCount = count
                elif count == highCount:
                    bestLabels.append(label)
            # use the nearest neighbor as tie breaker
            if len(bestLabels) > 1:
                for label in ordered:
                    if label in bestLabels:
                        predictions.append([label])
                        break
            else:
                predictions.append([bestLabels[0]])

        return initDataObject(predictions, None, None, testX.getTypeString(),
                              skipDataProcessing=True)


    def getScores(self, testX):
        """
        If this learner is a classifier, then return the scores for each
        class on each data point, otherwise raise an exception. The
        scores must be returned in the natural ordering of the classes.
        """
        ret = None
        dists = self._getDistanceMatrix(testX)
        labelVals = list(self._trainY.countUniqueElements().keys())
        labelVals.sort()
        for point in dists:
            _, labelVotes = self._kNeighborOrderedLabelsAndVotes(point)
            scoreList = [labelVotes[val] if val in labelVotes else 0
                         for val in labelVals]
            scores = initDataObject(scoreList, None, None,
                                    testX.getTypeString(),
                                    skipDataProcessing=True)
            if ret is None:
                ret = scores
            else:
                ret.points.append(scores, useLog=False)
        return ret


    def _getDistanceMatrix(self, testX):
        """
        A matrix of the euclidean distances of each point in testX from
        each point in trainX. The resulting matrix will be shape:
        numTestPoints x numTrainPoints.
        """
        trainArray = dtypeConvert(self._trainX.copy('numpy array'))
        testArray = dtypeConvert(testX.copy('numpy array'))
        # euclidean distance for each point in test
        dists = np.sqrt(-2 * np.dot(testArray, trainArray.T)
                           + np.sum(trainArray**2, axis=1)
                           + np.sum(testArray**2, axis=1)[:, np.newaxis])
        return dists


    def _kNeighborOrderedLabelsAndVotes(self, point):
        """
        Returns a two-tuple of a list of the labels in order by distance
        and a dictionary mapping the labels for y to the number of times
        that label occurred.
        """
        labelVotes = {}
        idxDists = list(enumerate(point))
        idxDists.sort(key=lambda x: x[1])
        orderedLabels = [self._trainY[i] for i, _ in idxDists[:self.k]]
        for label in orderedLabels:
            if label in labelVotes:
                labelVotes[label] += 1
            else:
                labelVotes[label] = 1

        return orderedLabels, labelVotes
