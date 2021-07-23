"""
# Cleaning Data

### Preparing interstate traffic data for machine learning

The goal here is to address several issues present in the
`Metro_Interstate_Traffic_Volume.csv` dataset to make it suitable for
supervised machine learning (i.e., predicting one of the columns in the
data using other columns). Each data point in this dataset represents a
moment in time where 9 features are recorded. The features are variables
which may affect the traffic volume on an interstate highway as well as
the current traffic volume. A new data point is recorded each time a
change occurs in one of the weather forecast features.

[Open this example in Google Colab][colab]

[Download this example as a script or notebook][files]

[Download the dataset for this example][datasets]

[colab]: https://colab.research.google.com/drive/1TBeGjU7Jzt2XNzD9X_WFtmos4MvnrOje?usp=sharing
[files]: files.rst#cleaning-data
[datasets]: ../datasets.rst#cleaning-data
"""

## Getting started ##

## We first use `nimble.fetchFile` to retrieve our dataset. This will return
## the path to our dataset, downloading it from the web if it is not already
## available locally. Nimble has built in a shorthand for datasets in the
## [UCI repository](https://archive.ics.uci.edu/ml) that we use below. The
## second argument for `nimble.data` (`source`) can be a regular python data
## object, a path to a file, an open file, or URL pointing to the location on a
## website where the data resides. Here, we use the path returned by
## `nimble.fetchFile`.

from datetime import datetime

import nimble

path = nimble.fetchFile('uci::Metro Interstate Traffic Volume')
traffic = nimble.data('Matrix', path, name='Metro Interstate Traffic Volume')

## The `show` method provides control over the printed output for an object.
## It prints a description, the object name and shape and the object data
## (truncating if necessary) given the parameters. To see a good selection of
## our data throughout this example, we will want to adjust the ``maxWidth``
## and ``maxHeight``. Since we will often want to use these same width and
## height settings in many of the calls to `show` in this example, it will be
## best to pack the values for these keyword arguments into a dictionary.
keywordsForShow = {'maxWidth': 120, 'maxHeight': 9}
traffic.show("Raw traffic data", **keywordsForShow)

## The machine learning algorithms we plan to use require numeric data and can
## be sensitive to outliers. Our data contains 48,204 points and 9 features,
## but some points and features will require cleaning before these machine
## learning algorithms can be applied to the data. The default
## `features.report` contains 10 statistics, but we will limit it to the
## following four to help identify non-numeric data, missing data and outliers.
stats = ['mode', 'minimum', 'maximum', 'count']
report = traffic.features.report(stats)
report.show("Feature statistics")

## Statistics could not be calculated for all features, indicating some are
## non-numeric. 0 Kelvin in `temp` and 9831.3 mm in `rain_1h` are also possible
## recording errors so we will also perform cleaning on some numeric features.

## Cleaning numeric data ##

## Let's extract (i.e., separate from the rest of the data) any rows with the
## value 0 in `temp` or 9831.3 in `rain_1h` since they seem very unlikely to be
## accurate, then we can reevaluate the statistics without those values.

def badPointIdentifier(pt):
    return pt['temp'] == 0 or pt['rain_1h'] == 9831.3

extracted = traffic.points.extract(badPointIdentifier)

fixedReport = traffic.features.report(stats)
fixedReport[['temp', 'rain_1h'], :].show("Modified feature report")
print('Number of points with errors:', len(extracted.points))

## After extracting those values, our `features.report` statistics look much
## more reasonable for those features. Since the values for those "bad" data
## points were implausible, we can assume that the 11 extracted points contain
## recording errors so we will ignore `extracted` and continue with the 48,193
## points still remaining in `traffic`.

## Cleaning non-numeric data ##

## The values in the `date_time` feature are strings, so we will parse each
## string to generate five new numeric features ('year', 'month', 'day',
## 'hour', 'weekday') to replace this feature.
def dateTimeSplitter(value):
    dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    return [dt.year, dt.month, dt.day, dt.hour, dt.weekday()]

traffic.features.splitByParsing('date_time', dateTimeSplitter,
                                ['year', 'month', 'day', 'hour', 'weekday'])

## Now let's take a look at our data again after splitting a single feature
## of text into 5 numeric features.
traffic.show('New parsed features in traffic data', **keywordsForShow)

## Above, we also see that the `holiday` feature has many `nan` values. Let's
## take a look at a selection of points that include a holiday to get a better
## understanding of this feature.
pointsWithHoliday = slice(1368, 1372)
dateInfoFeatures = ['holiday', 'year', 'month', 'day', 'hour']
sample = traffic[pointsWithHoliday, dateInfoFeatures]
sample.show('Data sample with a holiday', **keywordsForShow)

## Now we can see that this feature records the holiday name for the first data
## point recorded on a holiday, otherwise the value is `nan`. This means most
## data points falling on a holiday still have `nan` as the holiday value. It
## would be more helpful if this feature identified **every** point that
## occurred on a holiday.

## We need a more complex custom function to differentiate between missing
## values that fall on a holiday and those that don't. The `holidayToBoolean`
## function relies on the fact that our data points are chronological. As
## `points.transform` iterates through each data point, each point with a
## string value in the `holiday` feature indicates the start of a holiday.
## `currentHoliday` stores the year, month and day of that holiday so that
## subsequent points occurring on the same date can also be identified as
## holidays. Ultimately, every value in `holiday` is replaced with a boolean
## value.
holidayIndex = traffic.features.getIndex('holiday')
currentHoliday = {'date': None}
def holidayToBoolean(point):
    newPt = list(point)
    dateTuple = (point['year'], point['month'], point['day'])
    if isinstance(point['holiday'], str):
        currentHoliday['date'] = dateTuple
    if currentHoliday['date'] == dateTuple:
        newPt[holidayIndex] = True
    else:
        newPt[holidayIndex] = False

    return newPt

traffic.points.transform(holidayToBoolean)
sample = traffic[pointsWithHoliday, dateInfoFeatures]
sample.show('Data sample with converted holiday feature', **keywordsForShow)

## We have two features related to categorizing the weather conditions. We saw
## in our first look at the data that the `weather_description` feature is more
## detailed than the `weather_main` feature. "Clouds" in the `weather_main`
## feature could be "scattered clouds", "broken clouds" or "overcast clouds" in
## `weather_description`. Since these features are very similar, we will use
## only one of them. The `weather_main` feature provides a good general idea of
## the current weather so let's delete `weather_description` from our data.
traffic.features.delete('weather_description')

## To make the string values in `weather_main` suitable for machine learning,
## we will represent each of the 11 unique values contained in this column as
## 11 new binary features.
newCols = traffic.replaceFeatureWithBinaryFeatures('weather_main')
sampleFts = ['weather_main=Clouds', 'weather_main=Clear', 'weather_main=Mist']
traffic[pointsWithHoliday, sampleFts].show('Sample of binary weather features',
                                           **keywordsForShow)

## Now that we have removed any bad points and transformed all of our data to
## numeric values, our dataset is ready for machine learning. We will be using
## this data to predict the `traffic_volume` feature from the other features.
traffic.show('Cleaned traffic data', **keywordsForShow)

## Writing to a file ##

## We'd like to be able to load the cleaned data for our
## [Supervised Learning example](supervised_learning.ipynb) any time we want,
## so we will write it to a new csv file.
traffic.writeFile('Metro_Interstate_Traffic_Volume_Cleaned.csv')

## **References:**

## Dua, D. and Graff, C. (2019).
## UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
## Irvine, CA: University of California, School of Information and Computer Science.

## Link:
## https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
