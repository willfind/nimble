"""
Unsupervised Learning

Using nimble to classify online shoppers

Our dataset, `'online_shoppers_intention_clean.csv'`, is a collection of
behaviors for visitors to an online shopping website. Our goal will be to use
an unsupervised learner to classify visitors of the website into groups, then
explore how differences between groups could help increase the site's revenue.

Reference:
Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018).
[https://doi.org/10.1007/s00521-018-3523-0]

Dua, D. and Graff, C. (2019).
UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer Science.

Link to original dataset:
https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
"""
## Getting Started

import nimble

visits = nimble.createData('DataFrame', 'online_shoppers_intention_clean.csv',
                           featureNames=True)

## Train our learner
# We will use the elbow-method to determine the number of clusters we want to
# use for scikit-learn's KMeans learner.

withinClusterSumSquares = []
for i in range(1,11):
    trainedLearner = nimble.train('skl.KMeans', visits, n_clusters=i)
    inertia = trainedLearner.getAttributes()['inertia_']
    withinClusterSumSquares.append([i, inertia])

wcss = nimble.createData('List', withinClusterSumSquares,
                         featureNames=['clusters', 'wcss'])
wcss.plotFeatureAgainstFeature('clusters', 'wcss')

# 5 clusters is a reasonable choice according to the plot. We will train
# with 5 clusters then add the generated clusters as a new `'cluster'` feature
# in our object.

tl = nimble.train('skl.KMeans', visits, n_clusters=5)

clusters = tl.apply(visits)
clusters.features.setName(0, 'cluster')
visits.features.append(clusters)

## Analyzing the clusters
# Let's group our data by cluster so we can analyze differences between the
# groups identified by the algorithm. Then, we'll examine some difference in
# features that we could target marketing toward to increase revenue.

byCluster = visits.groupByFeature('cluster')

targetFts = ['Revenue', 'SpecialDay', 'Weekend', 'NewVisitor']

for cluster, data in byCluster.items():
    print('cluster {} ({} points):'.format(cluster, len(data.points)))
    print(data[:, targetFts].features.statistics('mean'))

# We see that weekends and special days do not seem to have much effect on
# revenue, so marketing based on the day may not be effective. However,
# clusters with more new visitors are clearly less likely to buy, so we
# may want to focus on ways to bring visitors back to the site.

## Improving revenue
# Let's examine some additional visitor characteristic differences between
# our worst revenue cluster (0) and our best revenue cluster (4).
# The visitors location is classified into one of 9 regions in the Region
# feature. The source that directed the visitor to the website is classified
# into one of 20 sources in the TrafficType feature. Targeted marketing by
# region and/or traffic type are practical ways to improve revenue so let's
# investigate if the distributions of these vary between the two clusters.

worstRevenue = byCluster[0]
bestRevenue = byCluster[4]

def featureDistributionProportions(data, feature):
    """
    Helper function to examine a features distribution.
    """
    distribution = data.features[feature].countUniqueElements()
    for elem in distribution:
        distribution[elem] /= len(data.points)
    return distribution

# Print distribution differences greater than 5 percent
for ft in ['Region', 'TrafficType']:
    distBest = featureDistributionProportions(bestRevenue, ft)
    distWorst = featureDistributionProportions(worstRevenue, ft)
    for key in distBest:
        diff = distBest[key] - distWorst[key]
        if abs(diff) > 0.05:
            msg = "In {ft} {key}, there is a difference of {diff} between "
            msg += "the best and worst revenue clusters."
            print(msg.format(ft=ft, key=key, diff=round(diff, 4)))

# The two clusters are within 5 percent for each region, but the best revenue
# cluster gets much more traffic via TrafficType 2 and much less via
# TrafficType 3. We already know most visitors in `bestRevenue` are not new.
# But let's check how different visitor types in `worstRevenue` are being
# directed to our website via these two traffic types.
worstRevNewVisitCount = sum(worstRevenue[:, 'NewVisitor'])
worstRevReturnVisitCount = len(worstRevenue.points) - worstRevNewVisitCount
for trafficType in [2, 3]:
    def trafficCountNewVisit(pt):
        return pt['TrafficType'] == trafficType and pt['NewVisitor']

    def trafficCountReturnVisit(pt):
        return pt['TrafficType'] == trafficType and not pt['NewVisitor']

    newTraffic = worstRevenue.points.count(trafficCountNewVisit)
    returnTraffic = worstRevenue.points.count(trafficCountReturnVisit)
    byNew = newTraffic / worstRevNewVisitCount * 100
    byReturn = returnTraffic / worstRevReturnVisitCount * 100
    msg = '{vType} Visitors for trafficType {num}: {perc}%'
    print(msg.format(vType='New', num=trafficType, perc=round(byNew, 1)))
    print(msg.format(vType='Return', num=trafficType, perc=round(byReturn, 1)))

# We saw that our best revenue cluster generates more traffic from TrafficType
# 2 and less traffic from TrafficType 3. In our worst revenue group, new
# visitors are primarily coming from TrafficType 2, but many return visitors
# arrive via TrafficType 3. As a plan to generate more revenue, we would
# suggest that this website reevaluates any investment in TrafficType 3 and
# focuses on bringing more return visitors back via TrafficType 2.
