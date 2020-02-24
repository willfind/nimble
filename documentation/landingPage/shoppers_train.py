"""
Unsupervised Learning

Using nimble to classify online shoppers

Our dataset, `'online_shoppers_intention_clean.csv'`, is a collection of
behaviors for visitors to an online shopping website. Our goal will be to use
an unsupervised learner to classify visitors of the website into groups, then
explore the differences between some of these groups.

Reference:
Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018).
[https://doi.org/10.1007/s00521-018-3523-0]

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

### Train our learner
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

# 5 clusters seems to be reasonable choice according to the plot. We will train
# with 5 clusters then add the generated clusters as a new `'clusters'` feature
# in our object.

tl = nimble.train('skl.KMeans', visits, n_clusters=5)

clusters = tl.apply(visits)
clusters.features.setName(0, 'cluster')
visits.features.append(clusters)

### Analyzing the clusters
# Let's group our data by cluster so we can begin to examine characteristics of
# each cluster. `groupByFeature` will use the unique values in the `cluster`
# feature to group each point in `visits`. There are 5 clusters so the returned
# dictionary will have 5 keys paired with 5 new data objects that each contain
# only the points for the cluster associated with that key.

byCluster = visits.groupByFeature('cluster')

# Let's take a look at how some of the features change between clusters
for cluster, data in byCluster.items():
    numPoints = len(data.points)
    print('cluster {} ({} points):'.format(cluster, numPoints))
    boolFts = ['Weekend', 'SpecialDay', 'NewVisitor', 'Revenue']
    for ft in boolFts:
        ftPercentage = (sum(data[:, ft]) / numPoints) * 100
        print("  {}: {}%".format(ft, round(ftPercentage, 1)))
    durationFts = ['Administrative_Duration', 'Informational_Duration',
                   'ProductRelated_Duration']
    avgPageDurations = sum(data[:, durationFts].points) / numPoints
    for ft in durationFts:
        # convert from seconds to minutes
        avgDurationInMinutes = avgPageDurations[ft] / 60
        msg = "  Average {}: {} minutes"
        print(msg.format(ft, round(avgDurationInMinutes, 1)))

# We can see that the biggest differences between clusters are time spent on
# ProductRelated pages, percentage of new visitors and revenue percentage.
# We want to develop a strategy to maximize revenue so we will compare some
# visitor characteristics of our best revenue cluster (4) and our worst revenue
# cluster (0).
worstRevenue = byCluster[0]
bestRevenue = byCluster[4]

# A visitors location and the source that brought them to our website
# might vary between these clusters. The visitors location is classified
# into one of 9 regions in the Region Feature. The source that directed
# the visitor to the website is classified into one of 20 sources in the
# TrafficType feature. Let's see if there are any significant differences
# in these featrues between the best and worst revenue clusters.
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
# TrafficType 3. Let's check our worst revenue cluster for the types of
# visitors that are being directed to our website via these two traffic types.
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

# We saw from our best revenue cluster that we generate more revenue when more
# traffic is from TrafficType 2 and less traffic is from TrafficType 3. In our
# worst revenue group, new visitors are primarily coming from TrafficType 2,
# but many return visitors arrive via TrafficType 3. Our data provides no
# further details regarding TrafficType, so we will stop here. As a plan to
# generate more revenue, we would suggest that this website reevaluates its
# investments in TrafficType 3 and focus on bringing return visitors back via
# TrafficType 2.
