"""
# Neural Networks

### Using neural networks to identify handwritten digits.

Our dataset contains 1593 flattened 16x16 black and white images (each
pixel is represented by a 1 or 0) of handwritten digits (0-9).
Approximately half of the digits were written neatly and the other half
were written as quickly as possible. This provides some images that are
very difficult for even the human eye to decipher correctly. Each point
in our dataset will contain 266 features. The first 256 features
represent pixel values for the flattened image. The last 10 features
identify the known label for the image using one-hot encoding. For
example, [0,0,0,1,0,0,0,0,0,0] is a 3 and [0,0,0,0,0,0,0,0,0,1] is a 9.

[Open this example in Google Colab][colab]

[Download this example as a script or notebook][files]

[Download the dataset for this example][datasets]

[colab]: https://colab.research.google.com/drive/1o-1M9MWiqPdgXeAZ0RUDBN-ilx4J_rh1?usp=sharing
[files]: files.rst#neural-networks
[datasets]: ../datasets.rst#neural-networks
"""

## Getting Started ##

import nimble

path = nimble.fetchFile('uci::Semeion Handwritten Digit')
images = nimble.data('Matrix', path)

## Preparing the data ##

## We need to separate the features identifying the labels (the last 10
## features) from the features containing our image data. Using
## `features.extract` performs this separation. New labels are placed in the
## `labels` object and our `images` object now only contains our image data.
labels = images.features.extract(range(256, len(images.features)))
labels.show('one-hot encoded labels', maxHeight=9)

## Rather than 10 one-hot encoded features, we need our labels to be a single
## feature with the values 0-9 for our neural network. We can perform this
## conversion by matrix multiplying (with Python's `@` matrix multiplication
## operator) our `labels` object (1593 x 10) by a feature vector with the
## sequential values 0-9 (10 x 1). Since each label contains nine `0` values
## and a single `1`, the only non-zero product is between the `1` value in the
## label and the value in the feature vector that corresponds with the index of
## the label's `1` value. So, we quickly create a 1593 x 1 object with our
## labels as the integers 0 through 9.
intLabels = labels @ nimble.data('Matrix', list(range(10))).T
intLabels.show('integer labels', maxHeight=9)

## Now that we have a single feature of labels, we can randomly partition our
## data into training and testing sets.
trainX, trainY, testX, testY = images.trainAndTestSets(testFraction=0.25,
                                                       labels=intLabels)
## Simple neural network ##

## To start, we can build a simple Sequential model using the
## [Keras](https://keras.io) neural network package, directly from within
## Nimble. The `layers` argument for a Sequential object requires a list of
## Keras `Layer` objects. However, there is no need to import those directly
## from Keras. As long as Keras is installed, `nimble.Init` can search the
## interfaced package for the desired class and instantiate it with any keyword
## arguments. So we can avoid extra imports (i.e., `from keras.layers import
## Dense, Dropout`) and there is no need to recall the package's module names
## that contain the objects we want to use.
layer0 = nimble.Init('Dense', units=64, activation='relu', input_dim=256)
layer1 = nimble.Init('Dropout', rate=0.5)
layer2 = nimble.Init('Dense', units=10, activation='softmax')
layers = [layer0, layer1, layer2]

## Now that we’ve taken advantage of `nimble.Init` to define our layers, we can
## train and apply our model in one step. `nimble.trainAndApply` will first
## train the model on our trainX data to predict our trainY data, then apply
## the resulting model to our testX data.
digitProbability = nimble.trainAndApply(
    'keras.Sequential', trainX=trainX, trainY=trainY, testX=testX,
    layers=layers, optimizer='adam', loss='sparse_categorical_crossentropy',
    metrics=['accuracy'], epochs=10)

## The returned `digitProbability` object has 10 features where each feature
## denotes the probability for the label at that index.  For our prediction,
## we will use the index with the maximum probability (that is, the digit that
## the model thinks is most likely for that data point). Then we can see how
## our simple neural net performed on our test set.
def maximumProbability(pt):
    maximum = max(pt)
    return list(pt).index(maximum)

predictions = digitProbability.points.calculate(maximumProbability)
accuracy = nimble.calculate.fractionCorrect(testY, predictions)
print('Accuracy of simple neural network:', accuracy)

## Some older Keras versions rely on sources of randomness outside of Nimble's
## control, so your exact result could vary from ours, but should be around 90%
## accurate. This is pretty good for a simple model only trained for 10 epochs.

## Convolutional Neural Network ##

## Let's try to do better by increasing the complexity and creating a 2D
## convolutional neural network. This algorithm requires that our data be
## formatted so that it knows that each image is a 16 x 16 single channel
## (i.e., grayscale) image, so our flattened image data will not work. This
## will require each point to be a 3D (16 x 16 x 1) object. Fortunately, Nimble
## supports multi-dimensional data. We can reshape each point in our `trainX`
## and `testX` data using `unflatten`. Ultimately, this allows Nimble to
## identify each object as a four-dimensional object (a container of 3D objects
## representing 2D grayscale images). It is worth noting that Nimble will treat
## the object as if it has more than two dimensions, but the underlying data
## object is always two-dimensional. For this reason, the `shape` attribute
## will always provide the two-dimensional shape and the `dimensions` attribute
## will provide the dimensions that Nimble considers the object to have
## (`shape` and `dimensions` are the same for 2D data).
def reshapePoint(pt):
    ret = pt.copy()
    ret.unflatten((16, 16, 1))
    return ret

trainX = trainX.points.calculate(reshapePoint)
testX = testX.points.calculate(reshapePoint)
print('trainX.shape', trainX.shape, 'trainX.dimensions', trainX.dimensions)
print('testX.shape', testX.shape, 'testX.dimensions', testX.dimensions)

## For our 2D convolutional neural network, we will need five different types
## of Keras `Layers` objects. Just as we did with our simple neural network
## above, we can use `nimble.Init` to instantiate these objects without
## directly importing them from Keras.
layersCNN = []
layersCNN.append(nimble.Init('Conv2D', filters=64, kernel_size=3,
                             activation='relu', input_shape=(16, 16, 1)))
layersCNN.append(nimble.Init('Conv2D', filters=32, kernel_size=3,
                             activation='relu'))
layersCNN.append(nimble.Init('Dropout', rate=0.2))
layersCNN.append(nimble.Init('MaxPooling2D', pool_size=2))
layersCNN.append(nimble.Init('Flatten'))
layersCNN.append(nimble.Init('Dense', units=128, activation='relu'))
layersCNN.append(nimble.Init('Dense', units=10, activation='softmax'))

probabilityCNN = nimble.trainAndApply(
    'keras.Sequential', trainX=trainX, trainY=trainY, testX=testX,
    layers=layersCNN, optimizer='adam', loss='sparse_categorical_crossentropy',
    metrics=['accuracy'], epochs=10)

## We see that the loss and accuracy of this model improved much faster than
## our previous model. Let's check how it performed on our test set.
predictionsCNN = probabilityCNN.points.calculate(maximumProbability)
accuracyCNN = nimble.calculate.fractionCorrect(testY, predictionsCNN)
print('Accuracy of 2D convolutional neural network:', accuracyCNN)

## With the same amount of training, our convolutional neural network is about
## 6% more accurate than our simple neural network. Considering some images are
## very difficult to correctly identify because they were drawn as quickly as
## possible, nearly 97% accuracy is a significant improvement and a very good
## result.

## **References:**

## Semeion Research Center of Sciences of Communication, via Sersale 117,
## 00128 Rome, Italy
## Tattile Via Gaetano Donizetti, 1-3-5,25030 Mairano (Brescia), Italy

## Dua, D. and Graff, C. (2019).
## UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
## Irvine, CA: University of California, School of Information and Computer Science.

## Link to original dataset:
## https://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit
