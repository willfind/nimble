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

visits = nimble.data('Matrix', 'online_shoppers_intention_train.csv',
                     featureNames=True)

## Train a learner ##

## We will use the elbow-method to determine the number of clusters we want to
## use for scikit-learn's KMeans learner.
kmeans = {}
withinClusterSumSquares = []
for i in range(1,11):
    trainedLearner = nimble.train('skl.KMeans', visits, n_clusters=i)
    kmeans[i] = trainedLearner
    inertia = trainedLearner.getAttributes()['inertia_']
    withinClusterSumSquares.append([i, inertia])

wcss = nimble.data('List', withinClusterSumSquares,
                   featureNames=['clusters', 'wcss'])
wcss.plotFeatureAgainstFeature('clusters', 'wcss')

## The elbow-methood indicates 4 clusters could be optimal. We will apply our
## TrainedLearner with 4 clusters to our our data. We will also be analyzing
## our cluster centers so we will create a nimble data object containing these.
kmeans = kmeans[4]
clusters = kmeans.apply(visits)
clusters.features.setName(0, 'cluster')

centers = nimble.data('Matrix', kmeans.getAttributes()['cluster_centers_'],
                      featureNames=visits.features.getNames())
centers.points.setNames(['cluster' + str(i) for i in range(4)])

## `visits` has too many features to visualize our clusters, but we can use
## scikit-learn's PCA to reduce our data to 2 features for visualization. We
## also need to apply this learner to our cluster centers.
decomp = visits.copy()
pca = nimble.train('skl.PCA', decomp, n_components=2)
components = pca.apply(decomp)
components.features.setNames(['component_1', 'component_2'])
components.features.append(clusters)
pcaCenters = pca.apply(centers.copy())

## Cluster Visualization ##

## Now we can group our `components` by the KMeans clusters and plot them
## on the same figure to visualize how the KMeans learner clustered our data.
groups = components.groupByFeature('cluster')

for i in range(4):
    label = 'cluster' + str(i)
    groups[i].plotFeatureAgainstFeature('component_1', 'component_2',
                                        figureName='figure',
                                        label=label, show=False)
pcaCenters.plotFeatureAgainstFeature(0, 1, figureName='figure', marker='x',
                                     color='k', s=60)

## Cluster Analysis ##

## We can start by learning how many points fall into each cluster. We see an
## imbalance in cluster sizes, but it is possible this could be typical for
## ecommerce traffic data.
totalVisits = len(visits.points)
for i, cluster in groups.items():
    clusterVisits = cluster.shape[0]
    proportion = round(clusterVisits / totalVisits * 100, 2)
    print('cluster{} contains {} visits ({}%)'.format(int(i), clusterVisits,
                                                      proportion))

## Let's begin investigating differences between clusters by analyzing the
## cluster centers.

## `cluster0` visits the least number of pages, on average, followed by
## `cluster2`, then `cluster1`, and `cluster3` visits the most. We see this
## mirrors the PCA visualization of the clusters as we move left to right along
## the x-axis. We also notice that the majority of pages hit are on
## `ProductRelated` pages.
pageTypes = ['Administrative', 'Informational', 'ProductRelated']
centers[:, pageTypes].features.plot()

## We see the pattern is the same for duration.
pageDurations = ['Admin_Duration', 'Info_Duration', 'Product_Duration']
centers[:, pageDurations].features.plot()

## The pattern is consistent for `Revenue` as well. It is important to note
## that `Revenue` is a binary feature indicating whether a purchase was made.
## So, `cluster0` generates revenue just over 10% of the time and `cluster3`
## generates revenue almost 35% of the time, but we do not know the amount
## of these transactions.
centers[:, 'Revenue'].features.plot()

## Here we see an inverse of the pattern for `BounceRates`, `ExitRates`, and
## `NewVisitor`. `BounceRates` and `ExitRates` indicate users leaving the site,
## so we would expect the inverse pattern. It also appears that `cluster0` and
## `cluster2` contain most of the `NewVisitors`.
centers[:, ['BounceRates', 'ExitRates', 'NewVisitor']].features.plot()

# We do not see much variation across clusters for `Weekend`, however we see
# `SpecialDay` varies between clusters.
centers[:, ['Weekend', 'SpecialDay']].features.plot()

## Interestingly, `PageValues` does not align with any of the patterns that we
## have seen. A **PageValue** defines the estimated dollar amount for a visit
## to a given page. Since the `Revenue` feature only indicates whether a
## purchase was made it appears that it is not a good indication of which
## cluster brings in the most money. This indicates that cluster2 could be
## spending more money than cluster3 and cluster1, despite being less likely to
## generate revenue and spending less time on the site during the visit. We see
## the center for `cluster3`'s `PageValues` is about half of `cluster2`. So
## despite spending a lot of time on the site and often making a purchase, they
## are typically viewing or purchasing high value items.
centers[:,  'PageValues'].features.plot()

## Cluster Assumptions ##

## * `cluster0`: Over two-thirds of visits fall into this cluster. Visitors are
## likely to be a new visitor and do not spend much time on the site. Visits
## are more likely to occur near a special day and are unlikely to hit pages
## with high page values. This cluster is the least likely to generate revenue.
## * `cluster1`: Contains only 7% of visitors. These are often return visitors.
## They spend a good amount of time browsing the site and will visit pages with
## higher page values. They are reasonably likely to generate revenue.
## * `cluster2`: Represents about one quarter of all visits. They are somewhat
## more likely to be a new visitor. They do not spend as much time browsing,
## but often visit pages with high value pages. They appear to be targeting
## only specific products but are less likely to decide to buy.
## * `cluster3`: Mostly return visitors and only represent about 1% of all
## visitors. They spend longer durations browsing the site, but typically visit
## pages with low page values. They appear to enjoy browsing and are the most
## likely to generate revenue, but their purchases are likely lower cost items.

## **Reference:**

## Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018).
## [https://doi.org/10.1007/s00521-018-3523-0]

## Dua, D. and Graff, C. (2019).
## UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
## Irvine, CA: University of California, School of Information and Computer Science.

## Link to original dataset:
## https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
