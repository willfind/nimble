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

import nimble
from nimble.customLearners import CustomLearner
from nimble.exceptions import PackageException

scipy = nimble.importModule('scipy.spatial')


class KNNClassifier(CustomLearner):
    learnerType = 'classification'

    def train(self, trainX, trainY, k=5):
        self.k = k
        self._trainX = trainX
        self._trainY = trainY

    def apply(self, testX):
        trainArray = self._trainX.copy('numpy array')
        testArray = testX.copy('numpy array')

        # euclidean distance for each point in test
        dists = numpy.sqrt(-2 * numpy.dot(testArray, trainArray.T)
                    + numpy.sum(trainArray**2,axis=1)
                    + numpy.sum(testArray**2, axis=1)[:, numpy.newaxis])

        predictions = []
        for pt in dists:
            neighborLabels = []
            idxDists = [(i, d) for i, d in enumerate(pt)]
            idxDists.sort(key = lambda x: x[1])
            kNearest = idxDists[:self.k]
            for i, _ in kNearest:
                neighborLabels.append(self._trainY[i])
            labelCounts = {}
            for label in neighborLabels:
                if label in labelCounts:
                    labelCounts[label] += 1
                else:
                    labelCounts[label] = 1
            highCount = 0
            bestLabels = []
            for label, count in labelCounts.items():
                if count > highCount:
                    bestLabels = [label]
                    highCount = count
                elif count == highCount:
                    bestLabels.append(label)
            if len(bestLabels) > 1 and neighborLabels[0] in bestLabels:
                # if tied use the nearest neighbor (k=1) as tie breaker
                predictions.append([neighborLabels[0]])
            else:
                predictions.append([bestLabels[0]])

        return nimble.createData(testX.getTypeString(), predictions,
                                 useLog=False)


    def getScores(self, testX):
        """
        If this learner is a classifier, then return the scores for each
        class on each data point, otherwise raise an exception. The
        scores must be returned in the natural ordering of the classes.
        """
        ret = None
        for p in testX.points:
            nearestPoints = self._generatePointsSortedByDistance(p)
            results = self._voteNearest(nearestPoints)
            # sort ascending according to label ID
            results.points.sort(0, useLog=False)

            scores = results.features.extract(1, useLog=False)
            scores.transpose(useLog=False)

            if ret is None:
                ret = scores
            else:
                ret.points.add(scores, useLog=False)

        return ret

    def _generatePointsSortedByDistance(self, test):
        """
        Return a matrix where each row contains a point ID, and the
        distance to the point test.
        """
        if not scipy:
            msg = "scipy is not available"
            raise PackageException(msg)

        def distanceFrom(point):
            index = self._trainX.points.getIndex(point.points.getName(0))
            return [index, scipy.spatial.distance.euclidean(test, point)]

        distances = self._trainX.points.calculate(distanceFrom, useLog=False)
        distances.points.sort(1, useLog=False)
        return distances


    def _voteNearest(self, votes):
        """
        Takes a data object where each row contains a point ID, and the
        distance to the point we want to classify. Uses the point ID's
        to find labels in self.trainY, letting those be the votes. In
        case of a tie, we revert to k=1.
        """
        topK = votes.points.copy(end=self.k - 1, useLog=False)

        def mapper(point):
            labelIndex = self._trainY[int(point[0]), 0]
            return [(labelIndex, 1)]

        def reducer(key, valList):
            return (key, len(valList))

        results = topK.points.mapReduce(mapper, reducer, useLog=False)
        return results
