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

import UML
from UML.customLearners import CustomLearner
from UML.exceptions import PackageException
scipy = UML.importModule('scipy.spatial')


class KNNClassifier(CustomLearner):
    learnerType = 'classification'

    def train(self, trainX, trainY, k=5):
        self.k = k
        self._trainX = trainX
        self._trainY = trainY

    def apply(self, testX):
        def foo(p):
            nearestPoints = self._generatePointsSortedByDistance(p)
            results = self._voteNearest(nearestPoints)
            # sort according to number of votes received
            def scoreHelperDecending(point):
                return 0 - point[1]

            results.points.sort(sortHelper=scoreHelperDecending)
            # only one label received votes
            if len(results.points) == 1:
                prediction = results[0, 0]
            # there is a tie between labels, fall back to k=1
            elif results[0, 1] == results[1, 1]:
                prediction = self._trainY[int(nearestPoints[0, 0]), 0]
            # average case, top of the results has most number of votes
            else:
                prediction = results[0, 0]

            return prediction

        return testX.points.calculate(foo)


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
            results.points.sort(0)

            scores = results.features.extract(1)
            scores.transpose()

            if ret is None:
                ret = scores
            else:
                ret.points.add(scores)

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

        distances = self._trainX.points.calculate(distanceFrom)
        distances.points.sort(1)
        return distances


    def _voteNearest(self, votes):
        """
        Takes a data object where each row contains a point ID, and the
        distance to the point we want to classify. Uses the point ID's
        to find labels in self.trainY, letting those be the votes. In
        case of a tie, we revert to k=1.
        """
        topK = votes.points.copy(end=self.k - 1)

        def mapper(point):
            labelIndex = self._trainY[int(point[0]), 0]
            return [(labelIndex, 1)]

        def reducer(key, valList):
            return (key, len(valList))

        results = topK.points.mapReduce(mapper, reducer)
        return results
