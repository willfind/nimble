"""
# Merging and Tidying Data

### Creating a tidy (cleaned up) data object from multiple data objects.

In this example, we have temperature data contained in 8 files. Each
row in every file records 12 hours of temperatures for each day, but
files vary by weather station, time period, and extreme (i.e., the file
contains either minimum or maximum temperatures). Each file covers the
same date range so we want to create a single object with all of our
data. Once combined, we would like to restructure our data using
[Hadley Wickham's Tidy Data][wickham] principles. Wickham defines "data
tidying" as "structuring datasets to facilitate analysis" and Nimble
provides functions that reorganize points and features to create data
that is "tidy". Our goal is to create one tidy data object containing
all of the data from our 8 original files.

[wickham]: http://dx.doi.org/10.18637/jss.v059.i10

In this example we will learn about:

* [Combining data objects together](#Combining-the-data)
* [Applying tidy data functions](#Tidying-the-data)

[Open this example in Google Colab][colab]\
[Download this example as a script or notebook][files]\
[Download the dataset for this example][datasets]

[colab]: https://colab.research.google.com/drive/1v6bayDRsovOnzFnjRDYRaJ-PRSN7DIlE?usp=sharing
[files]: files.rst#merging-and-tidying-data
[datasets]: ../datasets.rst#merging-and-tidying-data
"""

## Getting Started ##

import nimble

bucket = 'https://storage.googleapis.com/nimble/datasets/tidy/'
dwtnMinAM = nimble.data(bucket + 'downtown_am_min.csv', returnType="Matrix")
dwtnMaxAM = nimble.data(bucket + 'downtown_am_max.csv', returnType="Matrix")
dwtnMinPM = nimble.data(bucket + 'downtown_pm_min.csv', returnType="Matrix")
dwtnMaxPM = nimble.data(bucket + 'downtown_pm_max.csv', returnType="Matrix")
airptMinAM = nimble.data(bucket + 'airport_am_min.csv', returnType="Matrix")
airptMaxAM = nimble.data(bucket + 'airport_am_max.csv', returnType="Matrix")
airptMinPM = nimble.data(bucket + 'airport_pm_min.csv', returnType="Matrix")
airptMaxPM = nimble.data(bucket + 'airport_pm_max.csv', returnType="Matrix")

## To begin, we create 8 objects from 8 different files. The variable names
## and object names for each object represent the weather station location
## (downtown or airport), the temperature extreme recorded (Min or Max) and the
## time of day (AM or PM). All of our files have the same header row and cover
## the same date range. Let's look at one of our objects to see these headers
## and understand the current format of our data.
dwtnMinAM.show('Example of data file structure', maxHeight=12)

## Combining the data ##

## First, we can reduce our number of objects to 4 by combining AM and PM
## temperatures of objects at the same weather station (downtown or airport)
## and with the same extreme (min or max). The feature names for AM and PM are
## currently the same, so we will need to modify the feature names in the PM
## objects so that they denote the hour according to a 24 hour clock.
ftsPM = ['date', 'hr12', 'hr13', 'hr14', 'hr15', 'hr16', 'hr17',
         'hr18', 'hr19', 'hr20', 'hr21', 'hr22', 'hr23']

for obj in [dwtnMinPM, dwtnMaxPM, airptMinPM, airptMaxPM]:
    obj.features.setNames(ftsPM)

## Now that we've differentiated our features, we can use a `merge` operation
## to combine the data. We want to join these objects on the 'date' feature
## (i.e., we are combining data with the same date) and use `point='union'`
## (that is, we want all the points from both files) so that we keep all
## possible dates, even if a date is missing for the AM or PM data.
dwtnMinAM.merge(dwtnMinPM, onFeature='date', point='union')
dwtnMaxAM.merge(dwtnMaxPM, onFeature='date', point='union')
airptMinAM.merge(airptMinPM, onFeature='date', point='union')
airptMaxAM.merge(airptMaxPM, onFeature='date', point='union')

dwtnMinAM.show('Downtown data merged on date', maxHeight=12)

## Next, we can reduce our number of objects from 4 to 2 by combining the
## objects with different extremes (min vs. max) for the same location.
## Before combining, we will want to add an “extreme” feature to each object
## based on whether it contains min vs. max data. Without this step, we would
## not be able to differentiate between minimum and maximum temperature points
## in the combined objects. Once our new feature is added, we can
## `points.append` our objects from the same weather station.
for obj in [dwtnMinAM, dwtnMaxAM, airptMinAM, airptMaxAM]:
    extreme = 'min' if 'min' in obj.path else 'max'
    ftData = [[extreme] for _ in range(len(obj.points))]
    newFt = nimble.data(ftData, featureNames=['extreme'])
    # New feature will be added at index position 1 (after "date" feature)
    obj.features.insert(1, newFt)

dwtnMinAM.points.append(dwtnMaxAM)
airptMinAM.points.append(airptMaxAM)

dwtnMinAM.show('Downtown combined extreme data', maxHeight=12)

## Finally, we can combine our two objects into one by combining our two
## weather stations (downtown vs. airport). Just like in the last step, we need
## to create a new 'station' feature for each object based on which weather
## station location (downtown vs. airport) recorded the data.
for obj in [dwtnMinAM, airptMinAM]:
    station = 'downtown' if 'downtown' in obj.path else 'airport'
    stationData = [[station] for _ in range(len(obj.points))]
    newFt = nimble.data(stationData, featureNames=['station'])
    obj.features.insert(1, newFt)
    obj.features.setName(1, 'station')

dwtnMinAM.points.append(airptMinAM)

## Since all of these operations have been in-place, our `dwtnMinAM` object
## now contains all of our data from the 8 files. This variable name could be
## confusing so, for clarity, let's assign this object to a new variable name,
## `tempData`. Let's also sort our data by date, so that we can double check
## that each date has a minimum and maximum temperature recording for each
## weather station. Taking a look at our data will also help us start exploring
## how we can begin to tidy it.
tempData = dwtnMinAM
tempData.name = 'combinedTemperatureData'
tempData.points.sort('date')

tempData.show('Fully merged (untidy) data', maxHeight=16)

## Tidying the data ##

## Our dataset is combined but not in the format we want. To structure our data
## for analysis, we would like each point to be a single observation of the
## variables in our data. According to [Hadley Wickham's Tidy Data][wickham]
## principles, our dataset is not tidy for two reasons. First, 24 observations
## are made every day (one each hour). Points should represent observations so
## each day should be represented by 24 points. Second, our minimum and maximum
## temperatures are variables for the same observation. Variables should be
## represented as features.

## [wickham]: http://dx.doi.org/10.18637/jss.v059.i10

## As an example, our current (truncated) data for `2001-01-01` at the
## `downtown` station can be seen below. We see it is structured using two
## points.
## ```
##    date    station  extreme  hr0   hr1   hr2   --  hr22   hr23
##
## 2011-01-01 downtown   min   2.840 2.019        --  9.399  11.859
## 2011-01-01 downtown   max   2.840 2.021 2.022  --  9.401  11.861
## ```
## To tidy that same data, we modify the structure to include one point for
## each hour and identify the minimum and maximum temperatures in the `min` and
## `max` features.
## ```
##    date    station   min    max   hour
##
## 2011-01-01 downtown 2.840  2.840  hr0
## 2011-01-01 downtown 2.019  2.840  hr1
##     |         |       |     |      |
## 2011-01-01 downtown 9.399  11.859 hr22
## 2011-01-01 downtown 3.649  11.861 hr23
## ```

## Tidying our data will take two steps. First, we need each point to represent
## a single hour of time. So we will take our 24 hour features (hr0, hr1, …,
## hr23) and collapse them to represent this same data using 24 points (one
## point for each feature that is collapsed). The collapsed features become two
## new features: one named `hour` storing each feature's name as a value and one
## named `temp` storing the temperature recorded during that hour.
hourFts = ['hr' + str(i) for i in range(24)]
tempData.points.splitByCollapsingFeatures(featuresToCollapse=hourFts,
                                          featureForNames='hour',
                                          featureForValues='temp')
tempData.points.sort(['date', 'hour'])
tempData.show('Split points by collapsing the hour features', maxHeight=16)

## This is looking closer now that each point refers to a single hour of time.
## However, we still have separate points storing our maximum and minimum
## temperatures. This is not obvious in the output above, let's make a couple
## of modifications to see this more clearly. First, we can clean our `hour`
## feature by transforming the former feature name strings into integers. Then,
## we will sort our data so that `show` will clearly display point pairs
## that need to be combined for our data to be tidy.
tempData.features.transform(lambda ft: [int(v[2:]) for v in ft],
                            features=['hour'])
tempData.points.sort(['date', 'hour'])
tempData.show('Date and hour sorted', maxHeight=16)

## We see above that `hr0` on `2011-01-01` for the `downtown` station, for
## example, is still represented by two points. This is because each point
## identifies either the minimum or maximum temperature. Our second step is to
## combine these two point pairs by expanding the features to include features
## for the minimum and maximum temperatures. Our `extreme` feature contains the
## values (min and max) that will become our new feature names and the `temp`
## feature contains the values that fill the new `min` and `max` features.
tempData.points.combineByExpandingFeatures(featureWithFeatureNames='extreme',
                                           featuresWithValues='temp')
tempData.show('Combined points by expanding extreme feature', maxHeight=16)

## Our object is now organized how we wanted with a tidy structure. There is
## one more tidying function in Nimble as well. It is designed to separate a
## feature containing multiple pieces of information into multiple features. We
## can demonstrate its functionality by applying it to our 'date' feature to
## create `year`, `month` and `day` features.
tempData.features.splitByParsing('date', lambda val: val.split('-'),
                                 ['year', 'month', 'day'])
tempData.show('Split features by parsing the date feature', maxHeight=16)

## **Reference:**

## Wickham, H. (2014). Tidy Data. Journal of Statistical
## Software, 59(10), 1 - 23.
## doi:http://dx.doi.org/10.18637/jss.v059.i10
