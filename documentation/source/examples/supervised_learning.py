"""
# Supervised Learning

### Training on interstate traffic data to make predictions about the future

In this example, we will use two datasets that contain data on
interstate traffic volumes and features that may contribute to changes
in traffic volume. `Metro_Interstate_Traffic_Volume_Cleaned.csv`, was
generated in our Cleaning Data example and is the cleaned data we will
use to build our supervised learning models.
`Metro_Interstate_Traffic_Volume_Predict.csv`, contains fictional
"forecast" data that we will use to simulate making traffic volume
predictions using our supervised machine learning model.

In this example we will learn about:

* [Testing machine learning algorithms](#Test-five-different-machine-learning-algorithms)
* [Hyperparameter tuning](#Improve-performance-by-tuning-hyperparameters)
* [Nimble's TraineLearner class](#Improve-performance-by-tuning-hyperparameters)
* [Applying a learner to new data](#Applying-our-learner)

[Open this example in Google Colab][colab]\
[Download this example as a script or notebook][files]\
[Download the dataset for this example][datasets]

[colab]: https://colab.research.google.com/drive/1XAqGkVFdI7UbJiegabkx5PynKZU-za3W?usp=sharing
[files]: files.rst#supervised-learning
[datasets]: ../datasets.rst#supervised-learning
"""

## Getting Started ##

import nimble

bucket = 'https://storage.googleapis.com/nimble/datasets/'
traffic = nimble.data(bucket + 'Metro_Interstate_Traffic_Volume_Cleaned.csv',
                      returnType="Matrix")
forecast = nimble.data(bucket + 'Metro_Interstate_Traffic_Volume_Predict.csv',
                       returnType="Matrix")

## Test five different machine learning algorithms ##

## We’ll divide our `traffic` data into training and testing sets. The test
## set (used to measure the out-of-sample performance) will contain 25% of our
## data and the remaining 75% will be used to train each machine learning
## algorithm.
testFraction = 0.25
yFeature = 'traffic_volume'
trainX, trainY, testX, testY = traffic.trainAndTestSets(testFraction, yFeature)

## For this example, we will use algorithms from the
## [Sci-kit Learn](https://scikit-learn.org/) package so it must be installed
## in the current environment. To check if Nimble has access to Sci-kit Learn
## in your environment, you can use `nimble.showAvailablePackages`.
## Additionally, we can see a list of all of the learners available to Nimble
## by using `nimble.showLearnerNames`. Uncomment the lines below if you would
## like to see the available packages and learners in your environment.
# nimble.showAvailablePackages()
# nimble.showLearnerNames()

## Nimble's training functions only need the package name and learner name to
## be identified. There is no need to recall, for example, that
## `LinearRegression` is in `sklearn.linear_model` or `KNeighborsRegressor` is
## in `sklearn.neighbors`, all Nimble requires are the strings
## 'sklearn.LinearRegression' and 'sklearn.KNeighborsRegressor', respectively.
## Using `nimble.trainAndTest`, we will quickly test the performance of five
## different regression algorithms (initially, we'll use default arguments to
## keep things simple). We can then analyze the performance by comparing each
## learning algorithm's root mean square error.
learners = ['sklearn.LinearRegression', 'sklearn.Ridge', 'sklearn.Lasso',
            'sklearn.KNeighborsRegressor', 'sklearn.GradientBoostingRegressor']
rootMeanSquareError = nimble.calculate.rootMeanSquareError
for learner in learners:
    performance = nimble.trainAndTest(learner, trainX, trainY, testX, testY,
                                      rootMeanSquareError)
    print(learner, 'error:', performance)

## `'sklearn.KNeighborsRegressor'` and `'sklearn.GradientBoostingRegressor'`
## had better performance for predicting traffic volume with this data than the
## linear regression based learners, so let's focus on optimizing those two.

## Improve performance by tuning hyperparameters ##

## The default arguments are unlikely to yield the best performance, so now we
## will adjust some parameter values for our two best learners. These
## adjustments can be made through `arguments` as a python `dict` or as keyword
## arguments. If we need more information about a learner's parameters, we can
## use `nimble.learnerParameters` and `nimble.learnerParameterDefaults`. Let's
## try it for KNeighborsRegressor.
nimble.showLearnerParameters('sklearn.KNeighborsRegressor')
nimble.showLearnerParameterDefaults('sklearn.KNeighborsRegressor')

## Furthermore, we can test multiple values for the same parameter
## by using the `nimble.Tune` object. The presence of `nimble.Tune` will
## trigger hyperparameter tuning. By default, it tunes the arguments
## consecutively using 5-fold cross-validation, but this can be modified by
## providing an `Tuning` object to the `tuning` parameter.
## The tuning will find the argument combination with the best average
## `performanceFunction` result and return the `TrainedLearner` using the best
## arguments.

## For KNeighborsRegressor, we will use `nimble.Tune` to try 3, 5, and 7 for
## the number of nearest neighbors and for `GradientBoostingRegressor` we will
## try different learning rate values. The `Tuning` object defines a method for
## selecting each argument set and how each argument set will be validated.
## Below, we will use the default "consecutive" method but instead of the
## default "cross validation", we will hold out a random 20% of our training
## data for validation. For details on all tuning options, see the `Tuning`
## documentation.
tuning = nimble.Tuning(validation=0.2, performanceFunction=rootMeanSquareError)
# some interfaces have alias options for the package name
# below we use the alias 'skl' for the 'sklearn' package.
knnTL = nimble.train('skl.KNeighborsRegressor', trainX, trainY,
                     arguments={'n_neighbors': nimble.Tune([3, 5, 7])},
                     tuning=tuning)
gbTL = nimble.train('skl.GradientBoostingRegressor', trainX, trainY,
                    learning_rate=nimble.Tune([0.01, 0.1, 1]),
                    tuning=tuning)

## The `nimble.train` function returns a `TrainedLearner`. With a
## `TrainedLearner` we can `apply` (make predictions on a test set), `test`
## (measure the performance on a test set with known labels) and it provides
## many other additional methods and attributes. The `knnTL` object was trained
## with the `n_neighbors` value that performed best during validation.
## When hyperparameter tuning occured, `TrainedLearner.tuning` provides access
## to the tuning results. The `allResults` and `allArguments` properties are
## sorted from best to worst. Let's see how each `knnTL` argument performed.
for result, args in zip(knnTL.tuning.allResults, knnTL.tuning.allArguments):
    print(result, args)

## Similarly `gbTL` was trained with the best of our three possible learning
## rates. Instead of seeing all the results, let's just see the best argument
## and best result this time.
print(gbTL.tuning.bestResult, gbTL.tuning.bestArguments)

## `knnTL` found `n_neighbors` of 5 to be the best setting.  This is the same
## as the default value so we already know how it performs on our testing data.
## However, `gbTL` found `learning_rate` of 1 outperformed the default, 0.1.
## Let's see how it performs on our testing (out-of-sample) data.
gbPerf = gbTL.test(testX, testY, rootMeanSquareError)
print('sklearn.GradientBoostingRegressor', 'learning_rate=1', 'error', gbPerf)

## Applying our learner ##

## We see a further improvement in the performance so the
## GradientBoostingRegressor with a learning rate of 1 is our best model. Now
## we will apply our `gbTL` trained learner to our `forecast` dataset to
## predict traffic volumes for a future day.
predictedTraffic = gbTL.apply(forecast)
predictedTraffic.features.setName(0, 'volume')

## Before printing, we will append the `hour` feature from `forecasts` to get
## a better visual of the traffic throughout the day.
predictedTraffic.features.append(forecast.features['hour'])
predictedTraffic.show('Traffic Volume Predictions')

## Based on our forecasted data, our learner is predicting heavier traffic
## volumes between 6 am and 6 pm with peak congestion expected around the 7 am
## hour for the morning commute and the 4 pm hour for the afternoon commute.

## **Reference:**

## Dua, D. and Graff, C. (2019).
## UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
## Irvine, CA: University of California, School of Information and Computer Science.

## Link to original dataset:
## https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
