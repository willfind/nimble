"""
# Cleaning Data

### Preparing interstate traffic data for machine learning

The goal here is to address several issues present in the
`Metro_Interstate_Traffic_Volume.csv` dataset to make it suitable for
supervised learning. Each point in this dataset represents a point in
time where 9 features are recorded. The features are variables which may
affect the traffic volume on an interstate highway and the current
traffic volume. A new point is recorded each time a change occurs in one
of the weather forecast features.
"""

## Getting started ##

from datetime import datetime

import nimble

traffic = nimble.data('Matrix', 'Metro_Interstate_Traffic_Volume.csv',
                      featureNames=True)
# view all data by optimizing show parameters
showKwargs = {'includeObjectName': False, 'maxHeight': 9, 'maxWidth': 120}
traffic.show("Raw traffic data", **showKwargs)

## Our data contains 48,204 points and 9 features, but some points and features
## will need to be addressed before machine learning algorithms can be applied
## to this data. The machine learning algorithms we plan to use require numeric
## data and can be sensitive to outliers. Running `featureReport` can provide
## a good starting point for cleaning the data.
print(traffic.featureReport())

## Statistics could not be calculated for all features, indicating some are
## non-numeric. The statistics for the numeric features also indicate that
## some (physically impossible) outlier values may be present.

## Cleaning numeric data ##

## 0 Kelvin in `temp` and 9831.3 mm in `rain_1h` indicate some outliers exist
## in these features. Let's extract those values to decide how to proceed.

def outlierIdentifier(pt):
    if pt['temp'] == 0 or pt['rain_1h'] == 9831.3:
        return True
    return False

outliers = traffic.points.extract(outlierIdentifier)

print(traffic[:, ['temp', 'rain_1h']].featureReport())
print('Number of points with outliers:', len(outliers.points))

## After extracting those values, our `featureReport` statistics look much
## more reasonable for those features. We can assume the 11 extracted points
## contain recording errors so we will ignore `outliers` and continue with
## the 48,193 points still remaining in `traffic`.

## Cleaning non-numeric data ##

## The values in the `date_time` feature are strings, but we can parse each
## string to generate five new numeric features to replace this feature.
def dateTimeSplit(value):
    dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    return [dt.year, dt.month, dt.day, dt.hour, dt.weekday()]

traffic.features.splitByParsing('date_time', dateTimeSplit,
                                ['year', 'month', 'day', 'hour', 'weekday'])

traffic.show('New parsed features in traffic data', **showKwargs)

## The `holiday` feature contains a holiday name string for the first point of
## each holiday, all other values are missing.
samplePts = slice(1369,1374)
dateFts = ['holiday', 'year', 'month', 'day', 'hour']
print(traffic[samplePts, dateFts])

## We will transform the `holiday` feature to a numeric feature by writing a
## function that assigns each point a new binary value in the `holiday`
## feature.
# TODO using 1/0 for now until True/False can be loaded from csv
currentHoliday = {'date': None}
def holidayToBinary(point):
    filledPt = []
    dateTuple = (point['year'], point['month'], point['day'])
    if isinstance(point['holiday'], str):
        currentHoliday['date'] = dateTuple
    if currentHoliday['date'] == dateTuple:
        filledPt.append(1)
    else:
        filledPt.append(0)
        currentHoliday['date'] = None

    filledPt.extend(point[1:])

    return filledPt

traffic.points.transform(holidayToBinary)
print(traffic[samplePts, dateFts])

## The `weather_main` and `weather_description` features are very similar.
weatherFts = ['weather_main', 'weather_description']
print(traffic[samplePts, weatherFts])

## While `weather_description` is more detailed, many of its unique values
## represent a very small proportion of the data. So, we will utilize the more
## general `weather_main` and remove `weather_description`. To make the string
## values in `weather_main` suitable for machine learning, we will represent
## each of the 11 unique values as 11 new binary features.
traffic.features.delete('weather_description')
newCols = traffic.replaceFeatureWithBinaryFeatures('weather_main')
print(traffic[samplePts, newCols[:3]])

## Our dataset is now ready for machine learning.
traffic.show('Cleaned traffic data', **showKwargs)

## Writing to a file ##

## So that we can load the cleaned data for our supervised learning example, we
## will write it to a new csv file.
traffic.writeFile('Metro_Interstate_Traffic_Volume_Cleaned.csv')

## **Reference:**

## Dua, D. and Graff, C. (2019).
## UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
## Irvine, CA: University of California, School of Information and Computer Science.

## Link:
## https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
