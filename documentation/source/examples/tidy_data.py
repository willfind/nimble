"""
# Combining and Tidying Data

### Creating a tidy data object from multiple data objects.

In this example, we have temperature data contained in many files. Each
point in every file records 12 hours of temperatures for each day, but
files vary by the location, time period and extreme (min or max). This
data would also benefit from a reorganization of the points and features
to optimize for machine learning, often called "data tidying" (Wickham).
Our goal is to create a single, tidy, nimble object where each point has
five features: 'date', 'hour', 'location', 'min' and 'max'.
"""

## Getting Started ##

## Create objects from each of our 8 data files

import nimble

downtownMinAM = nimble.data('Matrix', 'downtown_am_min.csv', featureNames=True)
downtownMaxAM = nimble.data('Matrix', 'downtown_am_max.csv', featureNames=True)
downtownMinPM = nimble.data('Matrix', 'downtown_pm_min.csv', featureNames=True)
downtownMaxPM = nimble.data('Matrix', 'downtown_pm_max.csv', featureNames=True)
airportMinAM = nimble.data('Matrix', 'airport_am_min.csv', featureNames=True)
airportMaxAM = nimble.data('Matrix', 'airport_am_max.csv', featureNames=True)
airportMinPM = nimble.data('Matrix', 'airport_pm_min.csv', featureNames=True)
airportMaxPM = nimble.data('Matrix', 'airport_pm_max.csv', featureNames=True)
print(downtownMinAM.name, downtownMinAM.shape)
print(downtownMinAM[:5, :])

## Combining the data ##

## First, we can reduce our number of objects to 4 by combining AM and PM
## temperatures of objects with the same location and extreme (min or max).
## The featureNames for AM and PM are currently the same, so we will need to
## modify the featureNames in the PM objects so that they denote the hour
## according to a 24 hour clock.
ftsPM = ['date', 'hr12', 'hr13', 'hr14', 'hr15', 'hr16', 'hr17',
         'hr18', 'hr19', 'hr20', 'hr21', 'hr22', 'hr23']

for obj in [downtownMinPM, downtownMaxPM, airportMinPM, airportMaxPM]:
    obj.features.setNames(ftsPM)

## Now that we've differentiated our features, we can use a merge operation. We
## want to join these objects on the 'date' feature and use point='union' so
## that we keep all possible dates, even when a date is missing for the AM or
## PM data.
downtownMinAM.merge(downtownMinPM, onFeature='date', point='union')
downtownMaxAM.merge(downtownMaxPM, onFeature='date', point='union')
airportMinAM.merge(airportMinPM, onFeature='date', point='union')
airportMaxAM.merge(airportMaxPM, onFeature='date', point='union')

print(downtownMinAM.name, downtownMinAM.shape)
print(downtownMinAM[:5, :])

## Next, we can reduce our number of objects to 2 by combining the objects
## with different extremes (min or max) for the same location. Before
## combining, we will want to add an `extreme`feature to each object.
for obj in [downtownMinAM, downtownMaxAM, airportMinAM, airportMaxAM]:
    extreme = 'min' if 'min' in obj.name else 'max'
    ftData = [[extreme] for _ in range(len(obj.points))]
    newFt = nimble.data('Matrix', ftData, featureNames=['extreme'])
    obj.features.insert(1, newFt)
    obj.features.setName(1, 'extreme')

downtownMinAM.points.append(downtownMaxAM)
airportMinAM.points.append(airportMaxAM)

print(downtownMinAM.name, downtownMinAM.shape)
print(downtownMinAM[:5, :])

## Finally, we can combine our two objects into one after defining a
## new 'station' feature for each object.
for obj in [downtownMinAM, airportMinAM]:
    station = 'downtown' if 'downtown' in obj.name else 'airport'
    stationData = [[station] for _ in range(len(obj.points))]
    newFt = nimble.data('Matrix', stationData, featureNames=['station'])
    obj.features.insert(1, newFt)
    obj.features.setName(1, 'station')

downtownMinAM.points.append(airportMinAM)

## Let's clarify the variable name before we move to tidying this data.
tempData = downtownMinAM
tempData.name = 'combined temperature data'
tempData.points.sort('date')
print(tempData.name, tempData.shape)
print(tempData[:5, :])

## Tidying the data ##
## Our data is combined, but not in the format we want. We would consider a
## tidy point to contain the min and max temperatures at a station for one
## hour of a day. However, our current points are either the min or max
## temperatures at a station for all hours of the day.

## First, let's represent a change in hour by different points, not features.
## If we collapse the hour features, every point will be split into 24 points
## (one for each hour feature we collapsed).
hourFts = ['hr' + str(i) for i in range(24)]
tempData.points.splitByCollapsingFeatures(featuresToCollapse=hourFts,
                                          featureForNames='hour',
                                          featureForValues='temp')
print(tempData.name, tempData.shape)
print(tempData[:5, :])

## Next, let's represent different temperature extremes in different features,
## not points. If we expand the values in the 'extreme' feature to be new
## features, we can combine the points with the same 'date', 'hour', and
## 'station' features
tempData.points.combineByExpandingFeatures(featureWithFeatureNames='extreme',
                                           featuresWithValues='temp')
print(tempData.name, tempData.shape)
print(tempData[:5, :])

## Our object is now organized how we wanted with a much more tidy structure.
## There is one more tidying function in nimble as well. It is designed to
## separate a feature containing multiple pieces of information into multiple
## features. We can demonstrate its functionality by applying it to our 'date'
## feature to create 'year', 'month' and 'day' features.
tempData.features.splitByParsing('date', lambda val: val.split('-'),
                                 ['year', 'month', 'day'])
print(tempData.name, tempData.shape)
print(tempData[:5, :])

## **Reference:**

## Wickham, H. (2014). Tidy Data. Journal of Statistical
## Software, 59(10), 1 - 23.
## doi:http://dx.doi.org/10.18637/jss.v059.i10
