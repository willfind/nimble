# Supervised Learning
### Using nimble to predict interstate traffic volume

This example will use two datasets that contain data on interstate traffic volumes and features that may contribute to changes in traffic volume. `Metro_Interstate_Traffic_Volume_Cleaned.csv`, was generated in our [Data Cleaning example](cleaningData.md) and is the cleaned data we will use to build our supervised learning models. `Metro_Interstate_Traffic_Volume_Predict.csv`, contains fictional "forecast" data that we will use to simulate making traffic volume predictions using our supervised learning model.

![cleanedTraffic](../image/cleanedTraffic.png)
![forecast](../image/forecast.png)

## Getting Started
Create data objects from our `csv` files, which can be downloaded [here](#).

```python
import nimble

traffic = nimble.createData('Matrix',
                            'Metro_Interstate_Traffic_Volume_Cleaned.csv',
                            featureNames=True)

forecast = nimble.createData('Matrix',
                             'Metro_Interstate_Traffic_Volume_Predict.csv',
                             featureNames=True)
```

## Test five different learners
We'll divide our `traffic` data into training and testing sets, then use `nimble.trainAndTest` to quickly test the performance of five different regressors from the sci-kit learn package. We will analyze the performance by comparing each learner's root mean square error.

```python
testFraction = 0.25
yFeature = 'traffic_volume'
nimble.random.setSeed(23)
trainX, trainY, testX, testY = traffic.trainAndTestSets(testFraction, yFeature)

learners = ['sklearn.Lasso', 'sklearn.ElasticNet', 'sklearn.Ridge',
            'sklearn.KNeighborsRegressor', 'sklearn.RandomForestRegressor']
performanceFunction = nimble.calculate.rootMeanSquareError
for learner in learners:
    performance = nimble.trainAndTest(learner, trainX, trainY, testX, testY,
                                      performanceFunction)
    print(learner, 'root mean square error:', performance)
```

![RMSE](../image/RMSE.png)

`'sklearn.KNeighborsRegressor'` and `'sklearn.RandomForestRegressor'` look to be better choices for predicting traffic volume with this data than the linear regression learners, so let's focus on optimizing those two.

## Cross-validate arguments to improve performance
Additional arguments for a learner can be supplied through `arguments` as a dict or via keyword arguments and `nimble.CV` allows for multiple arguments to be passed for the same parameter. The presence of `CV` will trigger k-fold cross validation where k is the value of the `folds` arguments. Each combination of arguments will be  trained and tested using the `performanceFunction` to determine the best parameter set to use.

```python
knnTL = nimble.train('skl.KNeighborsRegressor', trainX, trainY,
                     performanceFunction, folds=5,
                     arguments={'n_neighbors': nimble.CV([1, 5, 11, 99])})
rfTL = nimble.train('skl.RandomForestRegressor', trainX, trainY,
                    performanceFunction, folds=5,
                    min_samples_split=nimble.CV([2, 4]),
                    min_samples_leaf=nimble.CV([1, 2]))
```

We used `nimble.train` above because it returns a `TrainedLearner`. A `TrainedLearner` allows us to apply and test, but also provides many additional methods and attributes. As an example, we can access all of our cross validation results through our `TrainedLearner`.

```python
for result in knnTL.crossValidation.allResults:
    print(result)
```

![knnCV](../image/knnCV.png)

Or we can access only the best arguments and results.  Note, the returned `TrainedLearner` is always trained using the best argument set if cross validation occurred.

```
print(rfTL.crossValidation.bestArguments, rfTL.crossValidation.bestResult)
```

![rfCV](../image/rfCV.png)

`knnTL` found `n_neighbors` of 5 to be the best argument.  This is the same as the default value so we already know how it performs on the testing data. However, `rfTL` found `min_samples_leaf` of 2 outperformed the default, 1. Let's see how it performs on our testing data.

```python
rfPerf = rfTL.test(trainX, trainY, nimble.calculate.rootMeanSquareError)
print(rfPerf)
```

![rfPerformance](../image/rfPerformance.png)

## Applying our learner
We see a further improvement in the performance so let's use `rfTL` to predict the traffic volumes for our `forecast` object. Before printing, we will append the `hour` feature from `forecasts` to get a better visual of the traffic throughout the day.

```python
predictedTraffic = rfTL.apply(forecast)
predictedTraffic.features.setName(0, 'volume')
predictedTraffic.features.append(forecast.features['hour'])
print(predictedTraffic)
```

![predictions](../image/predictions.png)

Based on our forecasted data, our learner is predicting heavy traffic starting in the morning and continuing throughout the day. Traffic volumes are expected to peak during the 7am hour for the morning commute and again at 4pm for the afternoon commute.

**Link to original dataset:**  
https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume

**Reference:**  
Dua, D. and Graff, C. (2019).
UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer Science.