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

In this example we will learn about:

* [Loading and displaying data](#Getting-started)
* [Cleaning numeric data](#Cleaning-numeric-data)
* [Cleaning non-numeric data](#Cleaning-non-numeric-data)
* [Writing data to a file](#Writing-to-a-file)

[Open this example in Google Colab][colab]\
[Download this example as a script or notebook][files]\
[Download the dataset for this example][datasets]

[colab]: https://colab.research.google.com/drive/1TBeGjU7Jzt2XNzD9X_WFtmos4MvnrOje?usp=sharing
[files]: files.rst#cleaning-data
[datasets]: ../datasets.rst#cleaning-data
"""

## Getting started ##

## We first use `nimble.fetchFiles` to retrieve our dataset. This will return
## the path to our dataset, downloading it from the web if it is not already
## available locally. Nimble has built in a shorthand for datasets in the
## [UCI repository](https://archive.ics.uci.edu) that we use below. The
## second argument for `nimble.data` (`source`) can be a regular python data
## object, a path to a file, an open file, or URL pointing to the location on a
## website where the data resides. Then, since this dataset only has one file
## we use the single path returned by `nimble.fetchFile` to load the data.

from datetime import datetime

import nimble

paths = nimble.fetchFiles('uci::Metro Interstate Traffic Volume')
traffic = nimble.data(paths[0], name='Metro Interstate Traffic Volume',
                      returnType="Matrix")

traffic = traffic[2557:, :]

## The `show` method provides more flexibility for the printed output than
## using `print` or `repr`. It prints a description, the `name` and `shape` of
## the object and the object data (truncating if necessary) given the
## parameters. The `maxWidth` and `maxHeight` parameters control the number of
## characters printed horizontally and vertically, respectively. By default
## these are set dynamically based on the terminal size, but more or less of
## the data can be displayed by setting them manually. To preview our data, we
## will limit the output to 16 lines.
traffic.show("Raw traffic data", maxHeight=16)

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
traffic.show('New parsed features in traffic data', maxHeight=16)

## The `holiday` feature is extremely idiosyncratic, so it will require a
## complex function to transform. Since this case this unique to this dataset,
## we won't dig into the details but Nimble can definitely handle complex cases
## like this one. The purpose of the function is to create a binary feature
## that identifies the points in the data that occur on a holiday.
 
holidayIndex = traffic.features.getIndex('holiday')
currentHoliday = {'date': None}
def holidayToBinary(point):
    newPt = list(point)
    dateTuple = (point['year'], point['month'], point['day'])
    if isinstance(point['holiday'], str):
        currentHoliday['date'] = dateTuple
    if currentHoliday['date'] == dateTuple:
        newPt[holidayIndex] = True
    else:
        newPt[holidayIndex] = False

    return newPt

dateInfoFeatures = ['holiday', 'year', 'month', 'day', 'hour']
traffic.points.transform(holidayToBinary)
sample = traffic[:, dateInfoFeatures]
sample.show('Data sample with converted holiday feature', maxHeight=16)

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
traffic[:, sampleFts].show('Sample of binary weather features',
                                           maxHeight=16)

## Now that we have removed any bad points and transformed all of our data to
## numeric values, our dataset is ready for machine learning. We will be using
## this data to predict the `traffic_volume` feature from the other features.
traffic.show('Cleaned traffic data', maxHeight=16)

## Writing to a file ##

## We'd like to be able to load the cleaned data for our Supervised Learning
## example any time we want, so we will write it to a new csv file.
traffic.save('Metro_Interstate_Traffic_Volume_Cleaned.csv')

## **References:**

## Dua, D. and Graff, C. (2019).
## UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
## Irvine, CA: University of California, School of Information and Computer Science.

## Link:
## https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
