"""
# Configuration, Logging, Randomness and Custom Learners

### Additional nimble functionality designed to support users.

Our data for this example contains 2000 points and 8 features. The first
7 features are wifi signal strength values picked up by a mobile phone
from seven different wifi sources . The final feature records which of
four rooms the mobile device was in. This dataset will help highlight
additional functionality included in the nimble library.
"""

## Getting Started ##

import tempfile

import nimble

## Configuration ##

## Nimble allows for user control of certain aspects of our library through
## a configuration file name configuration.ini. If configuration.ini does
## not exist when nimble is imported (as is expected when importing nimble
## for the first time) nimble generates configuration.ini. On import,
## configuration.ini is used to establish the settings for the
## session. Through `nimble.settings`, we can access the current settings
## as well as modify settings for this session or all future sessions.
## To start our session, let's make sure our logging output settings are set
## to the defaults used when nimble is first installed.
nimble.settings.set('logger', 'enabledByDefault', 'True')
nimble.settings.set('logger', 'enableCrossValidationDeepLogging', 'False')

## Logging ##

## Nimble keeps an ongoing log of each session, by default. The logging
## settings are configurable. Let's take a look at the default settings.
print(nimble.settings.get('logger', 'location'))
print(nimble.settings.get('logger', 'name'))
print(nimble.settings.get('logger', 'enabledByDefault'))
print(nimble.settings.get('logger', 'enableCrossValidationDeepLogging'))

## Let's use tempfile to create a new directory for our logs in this example.
## This will ensure log output in this example is consistent for everyone
## following along.
tempDir = tempfile.TemporaryDirectory('nimble-logs')
nimble.settings.set('logger', 'location', tempDir.name)

## All logged functions and methods have a `useLog` parameter. The default,
## value for `useLog` is `None`, meaning nimble will use the value stored in
## configuration.ini. The `nimble.data` method has a `useLog` parameter, but
## `show` does not. Since `enabledByDefault` is True in our configuration file,
## `showLog` should have one entry showing we loaded our data as a Matrix.
wifi = nimble.data('Matrix', 'wifi_localization.txt')
wifi.show(None, maxHeight=9)
nimble.showLog()

## Object methods like permute and transform can be logged. We can always
## override the configured setting `useLog` to `True` or `False` .
wifi.points.permute(useLog=False)
wifi.points.transform(lambda pt: abs(pt), useLog=True)
print(wifi[:3, :])
nimble.showLog()

## We only see a new log entry for `transform`, not `permute`, which is what we
## expected.

## Randomness ##

## For those following along, you may have not expected that `permute` would
## yield the same matrix for you as we displayed above. This is because
## nimble strives for reproducible results so we control for randomness. Of
## course, this can be disabled or you can choose your own random seed.
nimble.random.setSeed(None) # disable consistent results
wifi.points.permute()
print(wifi[:3, :])

## Everything we just did (except printing) was also logged. Let's check that
## our last 2 log entries show that a new random seed was assigned and that we
## permuted our object's points.
nimble.showLog(maximumEntries=2)

## Custom Learners ##

## Now we want to perform some machine learning. First, let's make our results
## reproducible again by setting a new random seed and reloading our data.
## This will ensure that everyone's data is divided into the same training and
## testing sets.
nimble.random.setSeed(1234)
wifi = nimble.data('Matrix', 'wifi_localization.txt', name='wifi')
trainX, trainY, testX, testY = wifi.trainAndTestSets(testFraction=0.3,
                                                     labels=7)

## Rather than use an algorithm from another package, let's create a simple
## `CustomLearner`. At a minimum, any `CustomLearner` must define a
## `learnerType` attribute and `train` and `apply` methods. In `train`, our
## learner will store the feature medians for each room. `apply` will examine
## the deviations in the test point from each room's feature medians and
## predict the room with the least deviation.
class LeastFeatureMedianDeviation(nimble.CustomLearner):
    learnerType = 'classification'

    def train(self, trainX, trainY):
        allData = trainY.copy()
        allData.features.append(trainX, useLog=False)
        self.featureMeans = {}
        byLabel = allData.groupByFeature(0, useLog=False)
        for label, group in byLabel.items():
            means = group.features.statistics('median')
            self.featureMeans[label] = means

    def apply(self, testX):

        def leastDeviation(point):
            least = None
            for label, means in self.featureMeans.items():
                sumSquaredDiffs = sum((point - means) ** 2)
                if least is None or sumSquaredDiffs < least[1]:
                    least = (label, sumSquaredDiffs)
            return least[0]

        return testX.points.calculate(leastDeviation)

performance = nimble.trainAndTest(LeastFeatureMedianDeviation,
                                  trainX, trainY, testX, testY,
                                  nimble.calculate.fractionCorrect)
print(performance)

## Our simple custom learner worked quite well, predicting the correct room in
## the test set over 96% of the time.

## Cross Validation Deep Logging ##

## Nimble also has a few builtin custom learners, which are registered under
## the package name 'nimble', let's try `KNNClassifier`. We will cross validate
## for two values of `k`, so let's also set `enableCrossValidationDeepLogging`
## to `'True'`.
nimble.settings.set('logger', 'enableCrossValidationDeepLogging', 'True')
performance = nimble.trainAndTest('nimble.KNNClassifier', trainX, trainY,
                                  testX, testY,
                                  nimble.calculate.fractionCorrect,
                                  folds=5, k=nimble.CV([1, 3]))

## Another way to check a learner's performance is to look at the log, let's
## see how `KNNClassifier` performed. By default, `showLog` does not display
## cross validation logs, so we need to increase the `levelOfDetail` to 3.
nimble.showLog(levelOfDetail=3, maximumEntries=2)

## We can see that k=3 just slightly outperformed k=1 and the learner correctly
## identified the room in 98% of the test points.

## Also note that showLog has many other parameters to query the log. Right now
## our log is small, but as it grows these can be very useful to find past
## information stored in the log file. For now, let's search try searching for
## our object's name, 'wifi'.
nimble.showLog(searchForText='wifi')

## While not required to do data science with nimble, configuration, logging,
## randomness and custom learners add a lot of helpful functionality to nimble
## and we expect you'll find yourself using some or all of it to support your
## data science work. That wraps it up for this example, so now is a good time
## to cleanup the temporary directory containing our log file.
tempDir.cleanup()

## **References:**

## Rajen Bhatt, 'Fuzzy-Rough Approaches for Pattern Classification: Hybrid
## measures, Mathematical analysis, Feature selection algorithms, Decision
## tree algorithms, Neural learning, and Applications', Amazon Books

## Jayant G Rohra, Boominathan Perumal, Swathi Jamjala Narayanan,
## Priya Thakur, and Rajen B Bhatt, 'User Localization in an Indoor
## Environment Using Fuzzy Hybrid of Particle Swarm Optimization &
## Gravitational Search Algorithm with Neural Networks', in Proceedings of
## Sixth International Conference on Soft Computing for Problem Solving,
## 2017, pp. 286-295.

## Dua, D. and Graff, C. (2019).
## UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
## Irvine, CA: University of California, School of Information and Computer Science.

## Link to dataset:
## https://archive.ics.uci.edu/ml/datasets/Wireless+Indoor+Localization
