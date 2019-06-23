"""
For a simple classifier you could implement a k-nearest neighbors
classifier. For any point you want to predict, just compute the
Euclidean distance of that point to every point in the training set.
Then sort this list. Take the k points that have the lowest distance,
and for your prediction use whichever label occurs most (among those k
training points with lowest distances to the point you are predicting).
If there is a tie, use k=1
"""

from __future__ import absolute_import

import numpy

from nimble.customLearners import CustomLearner
from nimble.helpers import initDataObject


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
            labelCounts, nearestNeighbor = self._getNearestNeighborLabelCounts(
                point, returnNearest=True)
            highCount = 0
            bestLabels = []
            for label, count in labelCounts.items():
                if count > highCount:
                    bestLabels = [label]
                    highCount = count
                elif count == highCount:
                    bestLabels.append(label)
            # use the nearest neighbor (k=1) as tie breaker, if possible
            if len(bestLabels) > 1 and nearestNeighbor in bestLabels:
                predictions.append([nearestNeighbor])
            else:
                predictions.append([bestLabels[0]])

        return initDataObject(testX.getTypeString(), predictions, None, None,
                              skipDataProcessing=True)


    def getScores(self, testX):
        """
        If this learner is a classifier, then return the scores for each
        class on each data point, otherwise raise an exception. The
        scores must be returned in the natural ordering of the classes.
        """
        ret = None
        dists = self._getDistanceMatrix(testX)
        labelVals = list(self._trainY.elements.countUnique().keys())
        labelVals.sort()
        for point in dists:
            labelCounts = self._getNearestNeighborLabelCounts(point)
            scoreList = [labelCounts[val] if val in labelCounts else 0
                         for val in labelVals]
            scores = initDataObject(testX.getTypeString(), scoreList,
                                    None, None, skipDataProcessing=True)
            if ret is None:
                ret = scores
            else:
                ret.points.add(scores, useLog=False)
        return ret


    def _getDistanceMatrix(self, testX):
        """
        A matrix of the euclidean distances of each point in testX from
        each point in trainX. The resulting matrix will be shape:
        numTestPoints x numTrainPoints.
        """
        trainArray = self._trainX.copy('numpy array')
        testArray = testX.copy('numpy array')
        # euclidean distance for each point in test
        dists = numpy.sqrt(-2 * numpy.dot(testArray, trainArray.T)
                           + numpy.sum(trainArray**2, axis=1)
                           + numpy.sum(testArray**2, axis=1)[:, numpy.newaxis])
        return dists


    def _getNearestNeighborLabelCounts(self, point, returnNearest=False):
        """
        A dictionary mapping the labels for y to the number of times
        that label occurred.
        """
        labelCounts = {}
        idxDists = [(i, d) for i, d in enumerate(point)]
        idxDists.sort(key=lambda x: x[1])
        kNearest = idxDists[:self.k]
        for i, _ in kNearest:
            label = self._trainY[i]
            if label in labelCounts:
                labelCounts[label] += 1
            else:
                labelCounts[label] = 1
        if returnNearest:
            nearestNeighbor = self._trainY[idxDists[0][0]]
            return labelCounts, nearestNeighbor
        return labelCounts
