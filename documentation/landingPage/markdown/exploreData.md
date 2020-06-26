# Exploring Data
### Using nimble to gain insight on the behaviors of online shoppers
Our dataset, `'online_shoppers_intention_clean.csv'`, is a collection of behaviors for visitors to an online shopping website. Our goal is to explore this dataset to gather information about the visitors to this site using a variety of the function and methods nimble has available for digging deeper into a dataset.

## Getting Started

```python
import nimble

visits = nimble.createData('DataFrame', 'online_shoppers_intention_clean.csv',
                           featureNames=True)
```

## Data Overview
Each point in this dataset represents a single visit to an online shopping website where 18 features are recorded.

```python
featureNames = visits.features.getNames()
```

The first 6 features record the counts and durations of time spent on the site, based on the types of pages the visitor viewed.

```python
webActivityFts = featureNames[:6]
print(visits[:4, webActivityFts].toString(maxWidth=120))
```

![activity](images/activity.png)

The next 3 features are website analytics collected during the visit.

```python
webAnalyticsFts = featureNames[6:9]
print(visits[:4, webAnalyticsFts].toString(maxWidth=120))
```

![analytics](images/analytics.png)

The last 9 features are details on the visit or visitor.

```python
visitDetailFts = featureNames[9:]
print(visits[:4, visitDetailFts].toString(maxWidth=120))
```

![details](images/details.png)

Now that we have a better understanding of our data, let's see what we can learn from it.

## Exploring data through data object methods.
Revenue is a boolean feature indicating if the visitor made a purchase.  
How does Revenue correlate with other features?

```python
correlations = visits.features.similarities('correlation')
print(correlations[:, 'Revenue'])
```

![correlations](images/correlations.png)

What proportion of visits generated revenue?

```python
revenueGen = sum(visits.features['Revenue']) / len(visits.points)
print('Proportion of visits that generated revenue:', revenueGen)
```

![revenue](images/revenue.png)

The SpecialDay feature ranges from 0 to 1 indicating proximity to a special day (such as Mother's Day), where increased visits might be expected.  
What proportion of visits occur in proximity to a special day?
```python
special = visits.points.count('SpecialDay>0') / len(visits.points)
print('Proportion of visits near a special day:', special)
```

![specialDay](images/specialDay.png)

## Exploring data through nimble's calculate module

This site categorizes their pages into the three types in the list below.  
Which types of pages are being visited the most?

```python
for ft in ['Administrative', 'Informational', 'ProductRelated']:
    mean = nimble.calculate.mean(visits[:, ft])
    print(ft, 'average hits per visit', mean)
```

![pageHits](images/pageHits.png)

What proportion of visitors reach a product-related page?

```python
noProduct = nimble.calculate.proportionZero(visits.features['ProductRelated'])
print('Proportion of visitors that view a product page:', 1 - noProduct)
```

![product](images/product.png)

## Exploring data through plotting
Is there a relationship between time spent on administrative pages and product related pages?

```python
visits.plotFeatureAgainstFeature('Administrative_Duration',
                                 'ProductRelated_Duration')
```

![duration](images/duration.png)

Each visitor's location is classified into one of nine regions in the Region feature.  
How is site traffic distributed by region?

```python
visits.plotFeatureDistribution('Region')
```
![regions](images/regions.png)

We have learned a lot about website visitors on our own. Next see how applying an [Unsupervised Learning](unsupervisedLearning.md) model can help us learn even more.

**Link to original dataset:**  
https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset

**References:**  
Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018).
[https://doi.org/10.1007/s00521-018-3523-0]

Dua, D. and Graff, C. (2019).
UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer Science.