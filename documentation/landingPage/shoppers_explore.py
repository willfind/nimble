"""
Exploring Data

Using nimble to gain insight on the behaviors of online shoppers

Our dataset, `'online_shoppers_intention_clean.csv'`, is a collection of
behaviors for visitors to an online shopping website. In this example, we
highlight some of the methods nimble has available for digging deeper into a
dataset.

References:
Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018).
[https://doi.org/10.1007/s00521-018-3523-0]

Dua, D. and Graff, C. (2019).
UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer Science.

Link to original dataset:
https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
"""

import nimble

# Each point in this dataset represents a single visit to an online shopping
# website where 18 features are recorded. All feature names are printed below
# and details will be provided on relevent features during this exploration.
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

### Exploring data through data object methods.
print(visits.summaryReport())
print(visits.features.getNames())
print(visits[:5, :])

# Revenue is a boolean feature indicating if the visitor made a purchase.
# How does Revenue correlate with other features?
correlations = visits.features.similarities('correlation')
print(correlations[:, 'Revenue'])

# What proportion of visits generated revenue?
revenueGen = sum(visits.features['Revenue']) / len(visits.points)
print('Proportion of visits that generated revenue:', revenueGen)

# The SpecialDay feature ranges from 0 to 1 indicating proximity to a special
# day (such as Mother's Day), where increased visits might be expected.
# How many visits occurred in proximity to a special day?
special = visits.points.count('SpecialDay>0')
print('Visits close to a special day:', special)

### Exploring data through nimble's calculate module

# This site categorizes their pages into the three types in the list below.
# Which types of pages are being visited the most?
for ft in ['Administrative', 'Informational', 'ProductRelated']:
    mean = nimble.calculate.mean(visits[:, ft])
    print(ft, 'average hits per visit', mean)

# What proportion of visitors reach a product-related page?
noProduct = nimble.calculate.proportionZero(visits.features['ProductRelated'])
print('Proportion of visitors that view a product page:', 1 - noProduct)

### Exploring data through plotting

# The amount of time spent on each page type is also recorded.
# Is there a relationship between time spent on informational pages and product related pages?
visits.plotFeatureAgainstFeature('Administrative_Duration',
                                 'ProductRelated_Duration')

# Each visitor's location is classified into one of nine regions in the Region feature.
# How is site traffic distributed by region?
visits.plotFeatureDistribution('Region')
