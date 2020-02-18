"""
Unsupervised Learning

Using nimble to classify online shoppers

Our dataset, `'online_shoppers_intention_clean.csv'`, is a collection of
behaviors for visitors to an online shopping website. Our goal will be to use
an unsupervised learner to classify visitors of the website into groups, then
explore the differences between some of these groups.

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

# 5 clusters seem to be reasonable choice according to the plot. We will train
# with 5 clusters then add the generated labels as a new `'labels'` feature in
# our object.

tl = nimble.train('skl.KMeans', visits, n_clusters=5)

labels = tl.apply(visits)
labels.features.setName(0, 'labels')

### Analyzing the labels
# Let's group our data by label so we can begin to examine characteristics of
# the different groups.

visits.features.append(labels)
byLabel = visits.groupByFeature('labels')

# Our goal is to maximize site visits that generate revenue, so we will
# investigate the proportion of visits that generated revenue for each label.
# Then we will focus on differences between the worst revenue group and the
# best revenue group.
for label, data in byLabel.items():
    msg = "label {} contains {} with an average Revenue of {}"
    avgRev = sum(data[:, 'Revenue']) / len(data.points)
    print(msg.format(label, len(data.points), avgRev))

# We see a much higher percentage of new visitors in the lowest revenue group.
worstRevenue = byLabel[0]
bestRevenue = byLabel[4]
newVisitWorst = sum(worstRevenue[:, 'NewVisitor']) / len(worstRevenue.points)
newVisitBest = sum(bestRevenue[:, 'NewVisitor']) / len(bestRevenue.points)
print("Proportion of new visitors for WORST revenue label:", newVisitWorst)
print("Proportion of new visitors for BEST revenue label:", newVisitBest)

# These groups are also spending different proportions of time on different
# page types. Both groups spend most of their time on product-related pages,
# but the low revenue group spends less time on product-related pages and
# more time on administrative pages.
durationFts = ['Administrative_Duration', 'Informational_Duration',
               'ProductRelated_Duration']
worstRevDurations = sum(worstRevenue[:, durationFts].points)
worstTotalDuration = sum(worstRevDurations)
worstDurationProportions = worstRevDurations / worstTotalDuration
worstDurationProportions.show('Duration Proportion for WORST label', False)
bestRevDurations = sum(bestRevenue[:, durationFts].points)
bestTotalDuration = sum(bestRevDurations)
bestDurationProportions = bestRevDurations / bestTotalDuration
bestDurationProportions.show('Duration Proportion for BEST label', False)

# This might indicate that we should investigate if the lower revenue is due to
# administative pages interfering with visitors (especially new visitors)
# ability to navigate to product pages or make a purchase.
