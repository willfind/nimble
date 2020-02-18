"""
Exploring Data

Using nimble to gain insight on the behaviors of online shoppers

Our dataset, `'online_shoppers_intention_clean.csv'`, is a collection of
behaviors for visitors to an online shopping website. In this example, we
highlight some of the methods nimble has available for digging deeper into a
dataset.

Reference:
Dua, D. and Graff, C. (2019).
UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer Science.

Link to original dataset:
https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
"""

import nimble

visits = nimble.createData('DataFrame', 'online_shoppers_intention_clean.csv',
                           featureNames=True)

# TODO loading True/False from CSV
def boolFill(ft, match):
    filled = []
    for val in ft:
        if val in match:
            filled.append(True if val =='True' else False)
        else:
            filled.append(val)
    return filled

visits.features.fillMatching(boolFill, ['True', 'False'])

print(visits[:5, :])
print(visits.features.getNames())

# A quick summary about the data
print(visits.summaryReport())

# Which pages are most people visiting?
for ft in ['Administrative', 'Informational', 'ProductRelated']:
    zeros = nimble.calculate.proportionZero(visits[:, ft])
    print(1 - zeros)

# How does Revenue correlate with other features?
correlations = visits.features.similarities('correlation')
print(correlations[:, 'Revenue'])

# Is there a relationship between time spent on informational pages and product related pages?
visits.plotFeatureAgainstFeature('Informational_Duration',
                                 'ProductRelated_Duration')

# Are visits consistent month-to-month?
visits.plotFeatureDistribution('Month')

# How long are most users spending on product related pages?
visits.plotFeatureDistribution('ProductRelated_Duration')

# How many product-related pages does the average user visit?
nimble.calculate.mean(visits.features['ProductRelated'])

# How many visits, on average, generate revenue?
sum(visits[:, 'Revenue']) / len(visits.points)
