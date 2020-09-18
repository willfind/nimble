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
## so first we will remove any points with no revenue. After deleting these
## points, `Revenue` and some other features only contain one unique value so
## we can remove them as well.
revenue = visits.points.extract("Revenue=1")
revenue.features.delete(lambda ft: nimble.calculate.standardDeviation(ft) == 0)

## For both the PCA and KMeans algorithms that we will use, we want to first
## normalize our data. We will stardardize each feature to have 0 mean and
## unit-variance.
normalized = revenue.copy()
normalized.features.normalize(subtract='mean', divide='standard deviation')
normalized.features.setNames(revenue.features.getNames())

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
## TrainedLearner with 3 clusters to our our data.
numClusters = 3

kmeans = kmeans[numClusters]
clusters = kmeans.apply(principal)
clusters.features.setName(0, 'cluster')

centers = nimble.data('Matrix', kmeans.getAttributes()['cluster_centers_'],
                      featureNames=principal.features.getNames())
centers.points.setNames(['cluster' + str(i) for i in range(numClusters)])

## Cluster Visualization ##

## Now we can group our `components` by the KMeans clusters and plot them
## on the same figure to visualize how the KMeans learner clustered our data.
## We will also add the cluster centers from our learner to the visualization.
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
## revenue-generating visits. We can also now generate cluster centers for our
## original data for deeper cluster analysis.
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

## Product related pages are the most commonly visited and `cluster0` view many
##  more pages than `cluster1` and `cluster2`.  We see `cluster0` averaging
## over 100 product related pages visited each visit, while `cluster1` and
## `cluster2` average just under 30.
pageTypes = ['Administrative', 'Informational', 'ProductRelated']
clusterCenters[:, pageTypes].features.plot()

## We see this same pattern between clusters for the total duration of time
## spent on each page type during the visit.
pageDurations = ['Admin_Duration', 'Info_Duration', 'Product_Duration']
clusterCenters[:, pageDurations].features.plot()

## Here we see that `cluster0` is very unlikely to contain new visitors.
## `cluster1` contains the highest percentage of new visitors, followed closely
## by `cluster2`. However, since `cluster2` is much larger, it does contain more
## new visitors overall.
clusterCenters[:, ['NewVisitor']].features.plot()

## Visits in `cluster1` are likely to be on a weekend, relative to the other
## clusters, but `cluster2` is more likely to visit near a special day.
clusterCenters[:, ['Weekend', 'SpecialDay']].features.plot()

## A **PageValue** defines the estimated dollar amount for a visit to any given
## page. We know that each visit generated revenue, but not how much, so
## PageValues may provide some insight on each cluster's spending habits. It
## is interesting that `cluster0` has the lowest PageValues. This means that
## on average they are looking at lower cost items on the site. So, despite
## spending a lot of time on the site, they are less likely to make a high
## value purchase. Whereas `cluster1` and `cluster2` view more expensive
## items on the site, so their purchases are likely to be more costly. However,
## we do not have more information, like purchase quantities, so we cannot
## learn anything more definitive about spending habits in our clusters.
clusterCenters[:,  'PageValues'].features.plot()

## The final features of interest are the months the purchases were made.
## We have data from 9 months, but we see an overwhelming majority of
## `cluster0`'s revenue-generating visits came in Month 7, as did a good
## percentage of `cluster1`'s. Since `cluster0` is mostly return visitors,
## it is possible some marketing or other event took place in month 7 that
## lead visitors to return and generate revenue at that time. `cluster2`
## looks to generate revenue most consistently.
months = [ft for ft in revenue.features.getNames() if ft.startswith('Month')]
clusterCenters[:, months].features.plot()

## Cluster Assumptions ##

## * `cluster0`: Contains almost 22% of all revenue-generating visits.
## Visitors are almost always return visitors and 70% of the visits in this
## cluster were in month 7. This cluster averages over an hour on the site and
## 100 individual page visits, but typically on pages with low page values. So,
## these visitors seem to enjoy browsing for a purchase from the site's lower
## cost items.
## * `cluster1`: Contains just over 20% of all revenue-generating visits.
## This cluster has the highest percentage of new visitors and visits are more
## likely to occur on a weekend. The average visitor in this cluster spend a
## bit under 21 minutes on the site and visits around 32 pages. This cluster
## visits pages with higher page values, though the average is not quite as
## high as `cluster2`.
## * `cluster2`: Contains nearly 58% of revenue-generating visits. This cluster
## tends to visit more often near a special day and is the most consistent for
## generating revenue month-over-month. These visits average the highest page
## values despite only averaging about 30 pages over a duration of about 20
## minutes. These visitors appear to be targeting specific, often more
## expensive, items when they visit the site.

## **Reference:**

## Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018).
## [https://doi.org/10.1007/s00521-018-3523-0]

## Dua, D. and Graff, C. (2019).
## UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
## Irvine, CA: University of California, School of Information and Computer Science.

## Link to original dataset:
## https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
