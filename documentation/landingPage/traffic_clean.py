"""
Cleaning Data

Using nimble to prepare interstate traffic volume

The dataset, `'Metro_Interstate_Traffic_Volume.csv'`, contains hourly data on
interstate traffic volumes and features that may contribute to changes in
traffic volume. Our goal here is to address several issues present in this
raw dataset to make it suitable for supervised learning.

Reference:
Dua, D. and Graff, C. (2019).
UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer Science.

Link:
https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
"""

from datetime import datetime

import nimble
from nimble import calculate

# Each point in this dataset represents a point in time where 9 features
# are recorded. The features are variables which may affect the traffic volume
# on an interstate highway and a new point is recorded each time a change
# occurs in one of the weather forecast features. Further details will be
# provided on any relevant features as the dataset is cleaned.

### Creating data
# We start by creating a data object using `nimble.createData`.  Nimble has
# four different data object types `Matrix`, `List`, `Sparse` and `DataFrame`
# that all have the same functionality and can interact with each other.
# `createData` can accept data in many forms, here we pass a string providing
# the path to a csv file.  This csv file contains headers for each column, so
# we set `featureNames` to True. After creating our object, we use some of it's
# functionality to learn a little bit about our dataset.

traffic = nimble.createData('Matrix', 'Metro_Interstate_Traffic_Volume.csv',
                            featureNames=True)
print(traffic.features.getNames())
print(traffic.summaryReport())
# Choosing this point range because shows a holiday
print(traffic[124:128, :])

### Cleaning data
# First, we are going to split the `date_time` feature into four features;
# `year`, `month`, `weekday`, and `hour`. This will give us numeric features
# from the non-numeric `date_time` feature.  To accomplish this, we will write
# a function to pass as the `rule` argument for `traffic.features.splitParsing`
# that extracts the information we need from each `datetime` value.

def dateTimeSplit(values):
    dt = datetime.strptime(values, "%Y-%m-%d %H:%M:%S")
    return [dt.year, dt.month, dt.day, dt.hour, dt.weekday()]

traffic.features.splitByParsing('date_time', dateTimeSplit,
                                ['year', 'month', 'day', 'hour', 'weekday'])

# Our initial look at the data also showed the `holiday` feature contains
# missing values. This feature contains a string of the holiday name if it
# is a holiday, but only for the first point of that day. Our goal is to make
# this a binary feature with 0 indicating a non-holiday and 1 indicating a
# holiday for each hour of every day.

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

# By default, nimble condenses the printed output of larger objects so we were
# unable to see the `weather_main` and `weather_description` features. A quick
# look indicates that they are very similar, but `weather_description` provides
# more detail for our predictions. We will remove `weather_main` and transform
# each of the 36 categories in `weather_description` to a binary feature so we
# have numeric data for our learners.

print(traffic[:4, ['weather_main', 'weather_description']])

traffic.features.delete('weather_main')
newCols = traffic.replaceFeatureWithBinaryFeatures('weather_description')

# Some values appear to be anomalies in the data, for example a small portion
# of values in the `temp` feature are recorded as 0 Kelvin. Since only a
# small number of points with anomalies exist, we will remove them. Our
# `anomalyRemover` function returns `True` for any value beyond 8 standard
# deviations from the mean indicating to `traffic.points.delete` to remove
# the point containing the anomaly from the data.

means = traffic.features.statistics('mean')
stDevs = traffic.features.statistics('standarddeviation')

def anomalyRemover(pt):
    for ft in ['temp', 'rain_1h', 'snow_1h']:
        if abs((pt[ft] - means[ft]) / stDevs[ft]) > 8:
            return True
    return False

traffic.points.delete(anomalyRemover)

### Writing to a file
# Now that our data is clean, we can write to a new csv file.
traffic.writeFile('Metro_Interstate_Traffic_Volume_Cleaned.csv')
