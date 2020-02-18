"""
Supervised Learning

Using nimble to predict interstate traffic volume

The dataset, `'Metro_Interstate_Traffic_Volume_Cleaned.csv'`, contains hourly
data on interstate traffic volumes and features that may contribute to changes
in traffic volume. To learn more about this dataset you can see the data
cleaning example. Our goal here is to find a learner that performs well at
predicting traffic volumes.
Note: The code displayed here requires the optional dependencies mlpy and
scikit-learn to be installed.

Reference:
Dua, D. and Graff, C. (2019).
UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer Science.

Link to original dataset:
https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
"""

import nimble

### Create data object and split into training and testing sets

traffic = nimble.createData('Matrix',
                            'Metro_Interstate_Traffic_Volume_Cleaned.csv',
                            featureNames=True)
trainX, trainY, testX, testY = traffic.trainAndTestSets(0.25, 'traffic_volume')

### Test default performance of three different learners
# In nimble, learners from any installed packages can be trained, applied and
# tested using the same API. You can even create your own learners. Let's use
# `nimble.trainAndTest` to get calculate the performance of three learners.
# Common performance functions, like `rootMeanSquareError`, can be found in
# nimble's `calculate` module.

learners = ['mlpy.Ridge', 'sklearn.KNeighborsRegressor',
            'sklearn.RandomForestRegressor']
performanceFunction = nimble.calculate.rootMeanSquareError
for learner in learners:
    performance = nimble.trainAndTest(learner, trainX, trainY, testX, testY,
                                      performanceFunction)
    print(learner, 'root mean square error:', performance)

### Cross-validate arguments for the best learners to attempt to improve performance
# We will use `nimble.train` here because it returns a `TrainedLearner` that
# provides functionality beyond applying and testing. Additional arguments for
# a learner can be supplied through `arguments` as a dict or via keyword
# arguments and `nimble.CV` allows for multiple arguments to be passed for the
# same parameter. Each combination of arguments will be trained and tested
# using the `performanceFunction` to determine the best parameter set to use.

knnTL = nimble.train('skl.KNeighborsRegressor', trainX, trainY,
                     performanceFunction,
                     arguments={'n_neighbors': nimble.CV([1, 3, 7, 11])})
rfTL = nimble.train('skl.RandomForestRegressor', trainX, trainY,
                    performanceFunction,
                    min_samples_split=nimble.CV([4, 6]),
                    min_samples_leaf=nimble.CV([2, 4]))

# When cross-validation occurs, the returned `TrainedLearner` is trained using
# the best argument set. A `TrainedLearner` has many methods and also stores
# other training information, like the cross-validation results.

print(knnTL.crossValidation.allResults)
print(rfTL.crossValidation.bestArguments, rfTL.crossValidation.bestResult)

# Now that it is clear our best `TrainedLearner` is `rfTL`, let's test its
# performance on our testing data.

rfPerf = rfTL.test(trainX, trainY, nimble.calculate.rootMeanSquareError)
print(rfPerf)

# That performed well on our test set so we will use it to predict tomorrow's traffic.

forecast = nimble.createData('Matrix',
                             'Metro_Interstate_Traffic_Volume_Predict.csv',
                             featureNames=True)
tomorrow = rfTL.apply(forecast)
print(tomorrow)
