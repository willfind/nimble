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
            'sklearn.KNeighborsRegressor', 'sklearn.HistGradientBoostingRegressor']
rootMeanSquareError = nimble.calculate.rootMeanSquareError
for learner in learners:
    performance = nimble.trainAndTest(learner, rootMeanSquareError, trainX,
                                      trainY, testX, testY)
    print(learner, 'error:', performance)

## `'sklearn.KNeighborsRegressor'` and `'sklearn.HistGradientBoostingRegressor'`
## had better out-of-the-box performance with this data than the linear
## regression based learners, so let's focus on optimizing those two.

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
## trigger hyperparameter tuning. By default, this tunes the arguments
## consecutively (optimizing one argument at a time while holding the others
## constant) and uses 5-fold cross-validation. This can be modified by
## providing a `Tuning` object to the `tuning` parameter. The tuning will find
## the argument combination with the best average `performanceFunction` result
## and return the `TrainedLearner` using the best arguments.

## For KNeighborsRegressor, we will use `nimble.Tune` to try 3, 5, and 7 for
## the number of nearest neighbors and for `HistGradientBoostingRegressor` we
## will try different learning rate values. The `Tuning` object defines both a
## method for selecting each argument set and how each argument set will be
## validated. Below, we will use the default "consecutive" method but instead
## of the default "cross validation", we will hold out a random 20% of our
## training data for validation. For details on all tuning options, see the
## `Tuning` documentation.
tuning = nimble.Tuning(validation=0.2, performanceFunction=rootMeanSquareError)
# some interfaces have alias options for the package name
# below we use the alias 'skl' for the 'sklearn' package.
knnTrained = nimble.train('skl.KNeighborsRegressor', trainX, trainY,
                     arguments={'n_neighbors': nimble.Tune([3, 5, 7])},
                     tuning=tuning)
hgbrTrained = nimble.train('skl.HistGradientBoostingRegressor', trainX, trainY,
                    learning_rate=nimble.Tune([0.1, 0.5, 1]),
                    tuning=tuning)

## The `nimble.train` function returns a `TrainedLearner`. With a
## `TrainedLearner` we can `apply` (make predictions on a test set), `test`
## (measure the performance on a test set with known labels) and it provides
## many other additional methods and attributes. In this case, beacuse
## hyperparameter tuning occured, `TrainedLearner.tuning` provides access to
## the tuning results. Let's check the best score and argument combination
## for `knnTrained`.
print(knnTrained.tuning.bestResult, knnTrained.tuning.bestArguments)

## As such, the `knnTrained` object we have access to was trained with
## `n_neighbors=3` since it had the best performance. Similarly `hgbrTrained`
## was trained with the best of our three possible learning rates. For
## `hgbrTrained` we will try checking the `allResults` and `allArguments`
## properties, which are sorted from best to worst performance and show the
## results for each of the tested argument sets.
for result, args in zip(hgbrTrained.tuning.allResults, hgbrTrained.tuning.allArguments):
    print(result, args)

## `knnTrained` found `n_neighbors` of 3 to be the best choice, but even so
## the best performance was not that great. However, `hgbrTrained` seems promising,
## with a `learning_rate` of 0.5 outperforming the default of 0.1. As a final
## check, let's see how it performs on our testing (out-of-sample) data.
hgbPerf = hgbrTrained.test(rootMeanSquareError, testX, testY)
print('sklearn.HistGradientBoostingRegressor', 'learning_rate=0.5', 'error', hgbPerf)

## Applying our learner ##

## We see a further improvement in the performance as compared to our original
## `nimble.trainAndTest` calls, so the `HistGradientBoostingRegressor` with a
## learning rate of 0.5 is our best model. Now we will apply our `hgbrTrained`
## trained learner to our `forecast` dataset to predict traffic volumes for a
## future day.
predictedTraffic = hgbrTrained.apply(forecast)
predictedTraffic.features.setNames('volume', oldIdentifiers=0)

## Before printing, we will append the `hour` feature from `forecasts` to get
## a better visual of the traffic throughout the day.
predictedTraffic.features.append(forecast.features['hour'])
predictedTraffic.show('Traffic Volume Predictions')

## Based on our forecasted data, our learner is predicting heavier traffic
## volumes between 6 am and 6 pm, trailing off into the evening. The peak
## congestion is expected around the 7 am hour for the morning commute and
## the 4 pm hour for the afternoon commute.

## **Reference:**

## Dua, D. and Graff, C. (2019).
## UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
## Irvine, CA: University of California, School of Information and Computer Science.

## Link to original dataset:
## https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
