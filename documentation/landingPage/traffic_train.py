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

# The `traffic` object we will create contains our cleaned data from the
# [Data Cleaning](link) example. Our goal is to find a learner that can
# accurately predict the traffic volume (`traffic_volume` feature) in this
# object, by splitting this data into training and testing data sets.
# Ultimately, we will apply the best learner to `forecast`, which contains
# (cleaned) forecasted data for a future day, to predict that day's traffic
# volumes.

traffic = nimble.createData('Matrix',
                            'Metro_Interstate_Traffic_Volume_Cleaned.csv',
                            featureNames=True)

testFraction = 0.25
yFeature = 'traffic_volume'
trainX, trainY, testX, testY = traffic.trainAndTestSets(testFraction, yFeature)

forecast = nimble.createData('Matrix',
                             'Metro_Interstate_Traffic_Volume_Predict.csv',
                             featureNames=True)

### Test default performance of five different learners
# In nimble, learners from various packages can be trained, applied and tested
# using the same API. You can even create your own learners. As a starting
# point, we will train five sci-kit learn learners with their default
# arguments.`nimble.trainAndTest` allows us to quickly test the performance of
# each learner. Many common performance functions can be found in nimble's
# `calculate` module like the `rootMeanSquareError` function that we will use
# in this example.

learners = ['sklearn.Lasso', 'sklearn.ElasticNet', 'sklearn.Ridge',
            'sklearn.KNeighborsRegressor', 'sklearn.RandomForestRegressor']
performanceFunction = nimble.calculate.rootMeanSquareError
for learner in learners:
    performance = nimble.trainAndTest(learner, trainX, trainY, testX, testY,
                                      performanceFunction)
    print(learner, 'root mean square error:', performance)

### Cross-validate arguments to attempt to improve performance
# Since KNeighborsRegressor and RandomForestRegressor with default arguments
# outperformed the others, let's fine-tune some of the arguments using
# cross-validation to try to improve upon their default performance.

# Additional arguments for a learner can be supplied through `arguments` as a
# dict or via keyword arguments and `nimble.CV` allows for multiple arguments
# to be passed for the same parameter. The presence of `CV` will trigger k-fold
# cross validation where k is the value of the `folds` arguments. Each
# combination of arguments will be trained and tested using the
# `performanceFunction` to determine the best parameter set to use.

knnTL = nimble.train('skl.KNeighborsRegressor', trainX, trainY,
                     performanceFunction, folds=5,
                     arguments={'n_neighbors': nimble.CV([1, 5, 11, 99])})
rfTL = nimble.train('skl.RandomForestRegressor', trainX, trainY,
                    performanceFunction, folds=5,
                    min_samples_split=nimble.CV([2, 4]),
                    min_samples_leaf=nimble.CV([1, 2]))

# We used `nimble.train` above because it returns a `TrainedLearner`. A
# `TrainedLearner` allows us to apply and test, but also provides many
# additional methods and attributes. As an example, we can access our
# cross validation results through our `TrainedLearner`. Note, the returned
# `TrainedLearner` is always trained using the best argument set if cross
# validation occurred.

print(knnTL.crossValidation.allResults)
print(rfTL.crossValidation.bestArguments, rfTL.crossValidation.bestResult)

# `knnTL` found `n_neighbors` of 5 to be the best argument.  This is the same
# as the default value so we already know how it performs on the testing data.
# However, `rfTL` found `min_samples_leaf` of 2 outperformed the default, 1.
# Let's see how it performs on our testing data.

rfPerf = rfTL.test(trainX, trainY, performanceFunction)
print(rfPerf)

# We see a further improvement in the performance so let's use `rfTL`
# to predict the traffic volumes for our `forecast` object. Before printing,
# we will append the hour from `forecasts` to see get a better visual of
# the traffic throughout the day

predictedTraffic = rfTL.apply(forecast)
predictedTraffic.features.setName(0, 'volume')
predictedTraffic.features.append(forecast.features['hour'])
print(predictedTraffic)

# Our learner is predicting heavy traffic starting in the morning and
# continuing throughout the day peaking during rush-hour times.
