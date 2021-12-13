"""
# Exploring Data

### Analyzing the behaviors of online shoppers

Our dataset, `online_shoppers_intention_clean.csv`, stores data on the
behaviors of visitors to an online shopping website. Our goal is to
study this use behavior using a variety of Nimble features in order to
extract useful insights.

[Open this example in Google Colab][colab]

[Download this example as a script or notebook][files]

[Download the dataset for this example][datasets]

[colab]: https://colab.research.google.com/drive/1QwTiHODLKlZp94pOGJCcnvNKA903uov1?usp=sharing
[files]: files.rst#exploring-data
[datasets]: ../datasets.rst#exploring-data
"""

## Getting Started ##

import nimble

bucket = 'https://storage.googleapis.com/nimble/datasets/'
visits = nimble.data(bucket + 'online_shoppers_intention_explore.csv',
                     returnType="Matrix")
featureNames = visits.features.getNames()

## Data Overview ##

## This dataset has 18 features, too many to show at one time. Let's begin
## exploring our data by looking at groups of similar features.

## This online shopping website is composed of three different types of
## webpages (Admininstrative, Informational, and Product Related). Our first 6
## features record the counts and durations of time spent on each page type for
## each visit.
pageActivityFts = featureNames[:6]
visits[:, pageActivityFts].show('Page activity features', maxHeight=12)

## The next 3 features are website analytics collected during the visit.
analyticFts = featureNames[6:9]
visits[:, analyticFts].show('Website analytic features', maxHeight=12)

## The last 9 features are details about the visit or visitor.
visitDetailFts = featureNames[9:]
visits[:, visitDetailFts].show('Visit detail features', maxHeight=12)

## Now that we have a better understanding of our data, let's see what we can
## learn from it.

## Exploring data through Nimble's calculate module ##

## Reaching product-related pages is important for maximizing the chance that
## a purchase is made. This site categorizes their pages into three types
## ("Administrative", "Informational", and "ProductRelated"). Let's calculate
## the `mean` and `median` counts for each page type and find out if most
## visitors are reaching a product-related page.
for ft in ['Administrative', 'Informational', 'ProductRelated']:
    mean = nimble.calculate.mean(visits[:, ft])
    print('Mean', ft, 'hits per visit', mean)
    median = nimble.calculate.median(visits[:, ft])
    print('Median', ft, 'hits per visit', median)

noProduct = nimble.calculate.proportionZero(visits.features['ProductRelated'])
print('Proportion of visitors that view a product page:', 1 - noProduct)

## We see that the mean values are consistently higher than the median values.
## Since the mean is sensitive to outliers, this indicates that we have some
## visitors that view a very high number of pages. We are also happy to see
## that nearly every visitor interacts with at least one product related page
## during their visit.

## Exploring data through data object methods. ##

## Now that we know visitors are typically viewing product pages, let's focus
## on the Purchase feature. Purchase is a boolean feature indicating whether
## a purchase was made. Let's find the proportion of visits that result in a
## purchase.
purchases = visits.points.count(lambda pt: pt['Purchase'])
print('Proportion of visits with a purchase:', purchases / len(visits.points))

## Now let's check how Purchase correlates with our other features.
correlations = visits.features.similarities('correlation')
correlations[:, 'Purchase'].show('Feature correlations with Purchase')

## The SpecialDay feature ranges from 0 to 1 indicating proximity to a special
## day. Most days will have a value of 0 but, for example, a visit three days
## before Mother's Day could have a value of 0.4, the day before Mother's Day
## would have a (higher) value of 0.8, and visits on Mother's Day have a value
## of 1. We might think a special day would increase purchases made on the
## site, but we see above that SpecialDay has a negative correlation with
## Purchase. Let's investigate. First, we will find what percent of visits were
## near a special day.
special = visits.points.copy('SpecialDay > 0')
visitPercent = len(special.points) / len(visits.points) * 100
print(f'{visitPercent:.2f}% of all visits were near a special day')

## Now, let's see what percent of purchases occurred near a special day.
specialPurchases = special.points.count(lambda pt: pt['Purchase'])
purchasePercent = specialPurchases / purchases * 100
print(f'{purchasePercent:.2f}% of all purchases were near a special day')

## Visits near a special day represent over 10% of visits, but only about 4% of
## purchases. It appears that these days attract more visitors to the site, but
## these visitors are less likely to make a purchase.

## Exploring data through plotting ##

## We saw above that visits near a special day leads to less purchases, let's
## explore the impact of location on purchases. The location of each visit is
## classified into one of nine regions, let's see the distribution of visits by
## region.
visits.plotFeatureDistribution('Region')

## We see above that region 1 provides the most visits to the website and
## regions 1 and 3 combine for over 50% of website traffic. Now, let's check if
## some regions are more likely to make a purchase. We can use
## `plotFeatureGroupStatistics` to do this. Since this function is grouping by
## the Region feature, the regions on the x-axis will be in order of appearance
## in the data. To keep them in ascending numeric order, we will first sort our
## data by Region. Once sorted, `plotFeatureGroupStatistics` will find the
## `count` of values in the purchase column for each Region. Then, it will
## further subdivide each count bar based on the values in Purchase (True or
## False). Now we can see if any regions are particularly better or worse at
## providing visits with a purchase.
visits.points.sort('Region')
visits.plotFeatureGroupStatistics(nimble.calculate.count, 'Purchase', 'Region',
                                  subgroupFeature='Purchase',
                                  color=['red', 'blue'])

## It does not appear that any region is making disproportionately more or less
## purchases than the others. We have learned a lot about our website data
## through this exploration. Next, see how we can use Nimble to extract more
## insight from this dataset using machine learning in our Unsupervised
## Learning example.

## **References:**

## Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018).
## [https://doi.org/10.1007/s00521-018-3523-0]

## Dua, D. and Graff, C. (2019).
## UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
## Irvine, CA: University of California, School of Information and Computer Science.

## Link to original dataset:
## https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
