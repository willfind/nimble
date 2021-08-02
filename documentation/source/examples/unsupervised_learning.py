"""
# Unsupervised Learning

### Classifying online shoppers

In our Exploring Data example, we began exploring visitor behavior data
from an ecommerce website. We will use that same data for this example,
but a version that has been prepared for machine learning,
`online_shoppers_intention_clean.csv`. Our goal will be to use an
unsupervised machine learning algorithm to gain a better understanding
of visitors that made a purchase during their visit.

[Open this example in Google Colab][colab]

[Download this example as a script or notebook][files]

[Download the dataset for this example][datasets]

[colab]: https://colab.research.google.com/drive/10KZjKMnrrahHbOmDQa-dne44FmMMPePS?usp=sharing
[files]: files.rst#unsupervised-learning
[datasets]: ../datasets.rst#unsupervised-learning
"""
## Getting Started ##

import nimble
from nimble.calculate import meanStandardDeviationNormalize

bucket = 'https://storage.googleapis.com/nimble/datasets/'
visits = nimble.data('Matrix', bucket + 'online_shoppers_intention_clean.csv')

## We are going to focus on categorizing our visitors that made a purchase.
## First, we will copy any data points for visits that resulted in a purchase
## into a new object.
purchaseOnly = visits.points.copy(lambda pt: pt['Purchase'])

## The Purchase feature in `purchaseOnly` now only contains the value `True`.
## A feature with only one unique value does not provide any useful information
## and can cause failures for some machine learning algorithms, so it should be
## removed. This data also has several one-hot encoded features and some have
## all zero values in `purchaseOnly`. For example, visits occurred using
## Browser 9 but no purchases were ever made, so the feature Browser=9 needs to
## be removed. We can remove all of these at once by checking the unique
## element count in each feature and deleting any with only 1 unique value.
purchaseOnly.features.delete(lambda ft: len(ft.countUniqueElements()) == 1)
purchaseOnlyFtNames = purchaseOnly.features.getNames()

## For both the PCA (Principal Component Analysis) and k-means algorithms that
## we will be using, we want to normalize our data. We will make a copy of our
## data (we will still need our original data later), and then standardize each
## feature to have a mean of 0 and standard deviation of 1.
purchaseNormalized = purchaseOnly.copy()
purchaseNormalized.features.normalize(meanStandardDeviationNormalize)

## Train a learner ##

## We will use PCA to reduce the dimensionality of our normalized data from 67
## features to 2 features (by finding the first 2 principle components, and
## then projecting the data onto these two factors). Using data with reduced
## dimensionality may improve the results of the k-means clustering algorithm,
## and can help us visualize our clusters.
pca = nimble.train('skl.PCA', purchaseNormalized, n_components=2)
purchasePCA = pca.apply(purchaseNormalized)
purchasePCA.features.setNames(['component_1', 'component_2'])

## The reduced dimensionality data from PCA is well suited for the k-means
## clustering algorithm. When doing k-means clustering, a major question is how
## many clusters to use (i.e., how many groups to automatically divide our data
## points between). We will repeatedly train k-means on our PCA data, varying
## the number of clusters from 1 to 10. Each time we will record the number of
## clusters and the inertia (the sum of squared distances to the closest
## cluster center). Plotting these values will help us determine the best
## number of clusters to use.
kmeans = {}
withinClusterSumSquares = []
for i in range(1, 11):
    trainedLearner = nimble.train('skl.KMeans', purchasePCA, n_clusters=i)
    kmeans[i] = trainedLearner
    inertia = trainedLearner.getAttributes()['inertia_']
    withinClusterSumSquares.append([i, inertia])

wcss = nimble.data('List', withinClusterSumSquares,
                   featureNames=['clusters', 'wcss'])
wcss.plotFeatureAgainstFeature('clusters', 'wcss', linestyle='-')

## The elbow method suggests that the "elbow" of the plot above provides the
## best number of clusters to use. If we envision our plot above as an arm, we
## see our elbow at 3 because there are a steeper negative slopes from 1 to 3
## and flatter negative slopes from 3 to 10. So, we will use our
## `TrainedLearner` that was trained with 3 clusters.
numClusters = 3
kmeans = kmeans[numClusters]

## Applying k-means clustering to our principal component data, will provide a
## feature vector storing a cluster number (0, 1, or 2) for each point in the
## data.
clusters = kmeans.apply(purchasePCA)
clusters.features.setName(0, 'cluster')

## Cluster Visualization ##

## Now that we have our clusters, we can plot our clusters on the same figure
## (canvas containing our plots) to help visualize how the k-means learner is
## clustering our principal component data. First, we will append our
## `clusters` feature to our principal component data so that we can use
## `groupByFeature` to separate our principal component data based on cluster
## number. `pcaByCluster` is a dictionary mapping cluster numbers to cluster
## data, so we iterate through the cluster numbers to add a scatterplot of each
## cluster's data to our figure. We provide a `figureName` so that each
## scatterplot is added to the same figure and we add a `label` so each plot
## uses a unique color. By default, figures will show after each function call
## so we will set `show=False` until our figure is complete. We will show our
## completed figure in the next step when we add the cluster centers.
purchasePCA.features.append(clusters)
pcaByCluster = purchasePCA.groupByFeature('cluster')
for i in range(numClusters):
    cluster = pcaByCluster[i]
    label = 'cluster' + str(i)
    cluster.plotFeatureAgainstFeature('component_1', 'component_2',
                                      figureName='clusterAndCenters',
                                      label=label, show=False)

## Each cluster has a center that will help us see how close each data point
## lies to the center of the cluster in which it was placed. We use the cluster
## center values from our `TrainedLearner` to create a new object. Each point
## represents a cluster and contains the principal component centers for that
## cluster. For this plot, values will be marked with a black "X" so the
## centers are clearly visible and we will add it to our same
## 'clusterAndCenters' figure. Now that all of our plots have been added to our
## figure, the default `show=True` will display the figure.
centers = nimble.data('Matrix', kmeans.getAttributes()['cluster_centers_'],
                      featureNames=['component_1', 'component_2'])
centers.points.setNames(['cluster' + str(i) for i in range(numClusters)])

title = 'k-means clustering of principal components'
centers.plotFeatureAgainstFeature(0, 1, figureName='clusterAndCenters',
                                  title=title, marker='x', color='k')

## Cluster Analysis ##

## We used PCA and k-means clustering to identify cluster numbers for each
## point in our data. Now we can group our original data, `purchaseOnly`, by
## cluster number to analyze the data in each cluster. As we iterate through
## each cluster group, we will analyze how many data points fall into each
## cluster and calculate the feature means in each cluster for further
## analysis.
purchaseOnly.features.append(clusters)
# purchaseByCluster is a dictionary mapping cluster number to cluster data
purchaseByCluster = purchaseOnly.groupByFeature('cluster')

totalVisits = len(purchaseOnly.points)
means = []
meanPtNames = []
for i in range(numClusters):
    cluster = purchaseByCluster[i]
    clusterVisits =  len(cluster.points)
    perc = round(clusterVisits / totalVisits * 100, 2)
    print('cluster{} contains {} visits ({}%)'.format(i, clusterVisits, perc))

    means.append(cluster.features.statistics('mean'))
    meanPtNames.append('cluster' + str(i))

## We see some imbalance in cluster sizes, but each cluster contains at least
## 20% of our purchaser visits so we feel confident to continue. To analyze the
## feature means in each cluster, we will make a new Nimble object. This object
## has three points (one for each cluster) containing our calculated feature
## means.
clusterMeans = nimble.data('Matrix', means, featureNames=purchaseOnlyFtNames,
                           pointNames=meanPtNames)

## For this example, a good visual way to analyze differences in feature means
## between clusters is to use plots. We examined the mean of all visits by page
## type in [Exploring Data](exploring_data.ipynb), so now let's see how many
## times our average purchaser **in each cluster** visits the various page
## types.
pageTypes = ['Administrative', 'Informational', 'ProductRelated']
clusterMeans[:, pageTypes].features.plot()

## The plot above shows that the average purchaser in `cluster2` views many
## more pages than those in `cluster0` and `cluster1`. Product related pages
## are the most commonly visited on all clusters, but  `cluster2` averages
## about 3 times as many visits as `cluster0` and `cluster1`. Let's see if this
## pattern continues for the duration of time spent on these page types.
pageDurations = ['Admin_Duration', 'Info_Duration', 'Product_Duration']
clusterMeans[:, pageDurations].features.plot()

## In the plot above, we see this same pattern between clusters for the average
## duration of time spent on each page type during a visit. So, `cluster2`
## appears to enjoy browsing while the other two clusters spend less time on
## the site before making a purchase. Let's see if new users in each cluster
## could be contributing to this difference.
clusterMeans[:, ['NewVisitor']].features.plot()

## Above, the plot shows that `cluster2` is very unlikely to contain new
## visitors. This seems to imply that returning visitors are likely to visit
## more pages and spend a longer time on the website. We also see that
## `cluster1` contains the highest percentage of new visitors, followed closely
## by `cluster0`. However, since we know `cluster0` is much larger, it will
## contain more new visitors overall. Next we will look at the distribution of
## geographic regions in each cluster.
region = [ft for ft in purchaseOnlyFtNames if ft.startswith('Region')]
clusterMeans[:, region].features.plot()

## We see above that the majority of visitors in each cluster are from region
## 1. No cluster contains drastically more visitors from a certain region than
## the other clusters and all regions are represented in each cluster. This
## is important because it indicates that differences between clusters are not
## likely to be regional. Let's see if purchases on weekends or special days
## differ between clusters.
clusterMeans[:, ['Weekend', 'SpecialDay']].features.plot()

## The average purchaser in `cluster1` is more likely to visit on a weekend,
## relative to the other clusters. This is the first time that `cluster1` has
## been significantly different from `cluster0`, so maybe `cluster1` represents
## more "Weekend Shoppers".  We also see `cluster0` leads in special day
## shopping. This could suggest that `cluster0` has more "Holiday Shoppers". We
## will keep these ideas in mind, but it is also possible that these
## differences are due to random chance. We also see that `cluster2` rarely
## makes a special day purchase, so their long durations on the site are not
## driven by special occasions.

## A **PageValue** defines the estimated dollar amount for a visit to any given
## page. We know that each visit ended in a purchase, but not how much, so
## PageValues may provide some insight on each cluster’s spending habits.
clusterMeans[:,  'PageValues'].features.plot()

## It is interesting that `cluster2` has the lowest average PageValue. This
## means that, on average, they are looking at lower cost items on the site.
## So, despite spending a lot of time on the site, they are less likely to make
## a high value purchase. Conversely, `cluster0` and `cluster1` view more
## expensive items on the site, so their purchases are likely to be more
## costly. Unfortunately, we do not have more information, like purchase
## quantities, so it is still unclear which cluster generates the most revenue.

## We have data for 9 months of purchases, let's see if our clusters spend
## consistently each month.
months = [ft for ft in purchaseOnlyFtNames if ft.startswith('Month')]
clusterMeans[:, months].features.plot()

## Above, we see an overwhelming majority of `cluster2`’s purchases came in
## Month 7, as did a good percentage of `cluster1`’s. Since `cluster2` is
## primarily returning visitors, it is possible some marketing or other event
## took place in month 7 that led visitors to return and make a purchase at
## that time. Additionally, `cluster2` rarely buys for special days, so we can
## assume that the purchases in month 7 were not due to a special occasion. We
## see `cluster0` purchases more consistently, this seems to contradict our
## previous idea that `cluster0` could contain "Holiday Shoppers".

## Last, we will look at the operating systems and browsers that visitors are
## using within the clusters.
os = [ft for ft in purchaseOnlyFtNames
      if ft.startswith('OperatingSystem') or ft.startswith('Browser')]
clusterMeans[:, os].features.plot()

## Often operating systems provide a browser, so a high correlation between the
## features above is not surprising. Let's focus on browsers since the majority
## of visitors used browser 1 or 2. Up to this point, `cluster0` and `cluster1`
## appeared mostly similar, however we see above that visitors use very
## different browsers between these two clusters. We also see that visitors in
## `cluster0` and `cluster2` generally use similar browsers. It seems strange
## that visitors using browser 2 span two clusters with very different
## behaviors, yet visitors on browser 1 mostly fall into the same cluster.

## Cluster Assumptions ##

## We quickly saw that `cluster2` varied quite a bit from `cluster0` and
## `cluster1`. `cluster2` averages over an hour on the site and 100 individual
## page visits, but typically on pages with low page values. Visitors are
## almost always return visitors and 70% of the visits in `cluster2` were in
## month 7. It is possible that visitors in this cluster may have been
## motivated, possibly by a sale or coupon, to return to the site and spend
## time browsing the site's lower-cost items to find something to purchase.

## We found that `cluster0` and `cluster1` are very similar in terms of number
## of pages visited and duration spent on the site. The average visitor in both
## of these clusters visited around 30 pages and spent about 20 minutes on the
## site. They also both have similar average page values that are much higher
## than `cluster2`, with `cluster0` being slightly higher than `cluster1`. So,
## these visitors appear to be targeting specific, often more expensive, items
## when they visit the site, rather than browsing the site’s inventory.

## More purchases in `cluster1` were on a weekend but browser differentiated it
## most from the other clusters. Nearly all purchases that use browser 1 are
## found in `cluster1`. Most users of other browsers are found in either
## `cluster0` or `cluster2`. We might expect customer behavior to be mostly
## consistent across browsers, but purchasers using browser 1 are very rarely
## spend long durations on the site. The association between browser 1 and
## shorter durations spent on the site is likely something worth investigation.

## We can only speculate, but there are a couple of potential reasons that we
## might be seeing this inconsistent behavior across browsers. There could be a
## flaw or poorly optimized aspect of the site on browser 1 that is
## discouraging visitors from browsing. Or, given we have speculated that the
## behavior of visitors in `cluster2` could have been driven by marketing, it
## may be the case that this marketing failed to reach users of browser 1.

## **Reference:**

## Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018).
## [https://doi.org/10.1007/s00521-018-3523-0]

## Dua, D. and Graff, C. (2019).
## UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
## Irvine, CA: University of California, School of Information and Computer Science.

## Link to original dataset:
## https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
