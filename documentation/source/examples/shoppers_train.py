"""
# Unsupervised Learning

### Classifying online shoppers to maximize revenue

Our dataset, `online_shoppers_intention_clean.csv`, is a collection of
behaviors for visitors to an ecommerce website. Our goal will be to use
an unsupervised learner to classify and understand visitors to the
website.

[Download this example as a script or notebook][files]

[Download the dataset for this example][datasets]

[files]: files.rst#unsupervised-learning-example
[datasets]: ../datasets.rst#unsupervised-learning-example
"""
## Getting Started ##


import nimble

visits = nimble.data('Matrix', 'online_shoppers_intention_clean.csv',
                     featureNames=True)

## We are going to focus on categorizing our visitors that made a purchase,
## so first we will extract any points with no revenue. Without these points,
## `Revenue` and some other features only contain one unique value so we can
## remove them.
revenue = visits.points.extract("Revenue=1")
revenue.features.delete(lambda ft: len(ft.countUniqueElements()) < 2)

## For both the PCA and KMeans algorithms that we will be using, we want to
## normalize our data. We will standardize each feature to have 0 mean and
## unit-variance.
normalized = revenue.copy()
normalized.features.normalize(subtract='mean', divide='standard deviation')

## Train a learner ##

## We will use PCA to reduce the dimensionality of our data from 67 features to
## 2 principal component features. Using data with reduced dimensionality
## supports better results from our KMeans algorithm and helps us to visualize
## our clusters.
pca = nimble.train('skl.PCA', normalized, n_components=2)
principal = pca.apply(normalized)
principal.features.setNames(['component_1', 'component_2'])

## The elbow-method is useful to determine the number of clusters we should use
## for scikit-learn's KMeans learner.
kmeans = {}
withinClusterSumSquares = []
for i in range(1,11):
    trainedLearner = nimble.train('skl.KMeans', principal, n_clusters=i,
                                  random_state=6)
    kmeans[i] = trainedLearner
    inertia = trainedLearner.getAttributes()['inertia_']
    withinClusterSumSquares.append([i, inertia])

wcss = nimble.data('List', withinClusterSumSquares,
                   featureNames=['clusters', 'wcss'])
wcss.plotFeatureAgainstFeature('clusters', 'wcss')

## The elbow-method indicates 3 clusters would be optimal. We will apply our
## `TrainedLearner` with 3 clusters to our our data.
numClusters = 3

kmeans = kmeans[numClusters]
clusters = kmeans.apply(principal)
clusters.features.setName(0, 'cluster')

centers = nimble.data('Matrix', kmeans.getAttributes()['cluster_centers_'],
                      featureNames=principal.features.getNames())
centers.points.setNames(['cluster' + str(i) for i in range(numClusters)])

## Cluster Visualization ##

## Now we can group our components by the KMeans clusters and plot them on the
## same figure to visualize how the KMeans learner clustered our data. We will
## also add the cluster centers from our learner to the visualization.
principal.features.append(clusters)
groups = principal.groupByFeature('cluster')
for i in range(numClusters):
    group = groups[i]
    label = 'cluster' + str(i)
    groups[i].plotFeatureAgainstFeature('component_1', 'component_2',
                                        figureName='figure', label=label,
                                        show=False)

centers.plotFeatureAgainstFeature(0, 1, figureName='figure', marker='x',
                                  color='k', s=50)

## Cluster Analysis ##

## We can start by learning how many points fall into each cluster. We see some
## imbalance in cluster sizes, but each cluster contains at least 20% of our
## revenue-generating visits. We will generate cluster centers for our original
## data for deeper cluster analysis.
centers = []
centerPtNames = []
centerFtNames = revenue.features.getNames()

revenue.features.append(clusters)
groups = revenue.groupByFeature('cluster')
totalVisits = len(revenue.points)
for i in range(numClusters):
    clusterVisits =  groups[i].shape[0]
    perc = round(clusterVisits / totalVisits * 100, 2)
    print('cluster{} contains {} visits ({}%)'.format(i, clusterVisits, perc))

    centers.append(groups[i].features.statistics('mean'))
    centerPtNames.append('cluster' + str(i))

clusterCenters = nimble.data('Matrix', centers, featureNames=centerFtNames,
                             pointNames=centerPtNames)

## Product-related pages are the most commonly visited and visitors in
## `cluster0` view many more pages than `cluster1` and `cluster2`.  We see
## `cluster0` averaging over 100 product related pages visited each visit,
## while `cluster1` and `cluster2` average just under 30.
pageTypes = ['Administrative', 'Informational', 'ProductRelated']
clusterCenters[:, pageTypes].features.plot()

## We see this same pattern between clusters for the total duration of time
## spent on each page type during the visit.
pageDurations = ['Admin_Duration', 'Info_Duration', 'Product_Duration']
clusterCenters[:, pageDurations].features.plot()

## Here we see that `cluster0` is very unlikely to contain new visitors.
## `cluster1` contains the highest percentage of new visitors, followed closely
## by `cluster2`. However, since `cluster2` is much larger, it does contain
## more new visitors overall.
clusterCenters[:, ['NewVisitor']].features.plot()

## Visits in `cluster1` are more likely to be on a weekend, relative to the
## other clusters, but `cluster2` is more likely to visit near a special day.
clusterCenters[:, ['Weekend', 'SpecialDay']].features.plot()

## A **PageValue** defines the estimated dollar amount for a visit to any given
## page. We know that each visit generated revenue, but not how much, so
## `PageValues` may provide some insight on each cluster's spending habits. It
## is interesting that `cluster0` has the lowest page values. This means that
## on average they are looking at lower cost items on the site. So, despite
## spending a lot of time on the site, they are less likely to make a high
## value purchase. Whereas `cluster1` and `cluster2` view more expensive
## items on the site, so their purchases are likely to be more costly.
## Unfortunately, we do not have more information, like purchase quantities, so
## it is still unclear which cluster generates the most revenue.
clusterCenters[:,  'PageValues'].features.plot()

## Other features of interest are the months the purchases were made. We have
## data from 9 months, but we see an overwhelming majority of `cluster0`'s
## revenue-generating visits came in Month 7, as did a good percentage of
## `cluster1`'s. Since `cluster0` is mostly return visitors, it is possible
## some marketing or other event took place in month 7 that lead visitors to
## return and generate revenue at that time. `cluster2` looks to generate
## revenue most consistently.
months = [ft for ft in revenue.features.getNames() if ft.startswith('Month')]
clusterCenters[:, months].features.plot()

## Next, we will look at the operating systems and browsers that visitors are
## using within the clusters. Often operating systems provide a browser, so it
## is not surprising to see these are highly correlated. Up to this point,
## `cluster1` and `cluster2` appeared quite similar, however we see here that
## visitors use very different operating systems and browsers between these two
## clusters. We also see that visitors in `cluster0` and `cluster2` generally
## use similar operating systems and browsers.
os = [ft for ft in revenue.features.getNames()
      if ft.startswith('OperatingSystem') or ft.startswith('Browser')]
clusterCenters[:, os].features.plot()

## Last, let's look at the distribution of regions in each cluster. We see that
## the majority of visitors in each cluster are from region 1. No cluster
## contains drastically more visitors from a certain region than the other
## clusters. This is important to note as it indicates that any major
## differences we noticed above are less likely to be regional.
region = [ft for ft in revenue.features.getNames() if ft.startswith('Region')]
clusterCenters[:, region].features.plot()

## Cluster Assumptions ##

## We quickly saw that `cluster0` varied quite a bit from `cluster1` and
## `cluster2`. `cluster0` averages over an hour on the site and 100 individual
## page visits, but typically on pages with low page values. Visitors are
## almost always return visitors and 70% of the visits in `cluster0` were in
## month 7. So, the visitors in this cluster may have been motivated, possibly
## by a sale or coupon, to return to the site and spend time browsing the
## site's lower-cost items to find something to purchase.

## `cluster1` and `cluster2` are very similar in terms of number of pages
## visited and duration spent on the site. The average visitor in both of
## these clusters visited around 30 pages and spent about 20 minutes on the
## site. They also both have similar average page values that are much higher
## than `cluster0`, with `cluster2` being slightly higher than `cluster1`. So,
## these visitors appear to be targeting specific, often more expensive, items
## when they visit the site, rather than browsing the site's inventory.

## Browser looks to differentiate `cluster1` from the other clusters. Nearly
## all revenue-generating visits that use browser 1 are found in `cluster1`.
## Most users of other browsers are found in either `cluster0` or `cluster2`.
## We might expect customer behavior to be mostly consistent across browsers,
## so it is interesting that revenue-generating visitors using browser 1 very
## rarely spend long durations on the site. This indicates that something
## about this browser has caused visitors to, generally, act in a manner that
## is dissimilar to visitors in `cluster0`.

## We can only speculate, but there are a of couple potential reasons that we
## might be seeing this inconsistent behavior across browsers. There could be a
## flaw or poorly optimized aspect of the site on browser 1 that is
## discouraging visitors from browsing. However, given we expect that the
## behavior of visitors in `cluster0` may have been driven by marketing, it
## could be that this marketing failed to reach most users of browser 1. Either
## way, if this site wanted to generate more behavior similar to that of
## `cluster0`, they would want to ensure consistency for visitors using
## different browsers.

## **Reference:**

## Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018).
## [https://doi.org/10.1007/s00521-018-3523-0]

## Dua, D. and Graff, C. (2019).
## UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
## Irvine, CA: University of California, School of Information and Computer Science.

## Link to original dataset:
## https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
