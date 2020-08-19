"""
# Neural Networks

### Using neural networks to identify handwritten digits.

Each point in our dataset contains 267 features. The first 256 features
represent pixel values for a flattened 16x16 grayscale image. The final
10 features are a binary representation of the known label for the
image. Our goal is to correctly identify handwritten digits using neural
networks.
"""

## Getting Started ##

import nimble

images = nimble.data('Matrix', 'semeion.data')

## Preparing the data ##

## We need to separate the labels (the last 10 features) from the image data
labels = images.features.extract(range(256, len(images.features)))

## We want our neural network to choose digits from 0-9. This can be
## accomplished by matrix multiplying the `labels` object by a feature vector
## with the range 0-9.
intLabels = labels @ nimble.data('Matrix', list(range(10))).T

## We'll append our new labels to the end of our images object, then
## randomly partition that data into train and test sets.
images.features.append(intLabels)
trainX, trainY, testX, testY = images.trainAndTestSets(testFraction=0.25,
                                                       labels=-1)

## Simple neural network ##

## To start, we can build a simple Sequential model using Keras. The `layers`
## argument for a Sequential object requires a list of Keras Layer objects.
## However, there is no need to import those directly from Keras. `nimble.Init`
## can instead be used to trigger instantiation of a matching class in the
## interfaced package. Below, we create Dense and Dropout objects with varying
## keyword arguments.
layer0 = nimble.Init('Dense', units=64, activation='relu', input_dim=256)
layer1 = nimble.Init('Dropout', rate=0.5)
layer2 = nimble.Init('Dense', units=10, activation='softmax')
layers = [layer0, layer1, layer2]

## Now that we've taken advantage of `nimble.Init` to define our layers, we can
## train and apply our model in one step. `nimble.trainAndApply` will first
## train the model on our `trainX` and `trainY` data, then apply the model to
## our testX data.
digitProbability = nimble.trainAndApply(
    'keras.Sequential', trainX=trainX, trainY=trainY, testX=testX,
    layers=layers, optimizer='adam', loss='sparse_categorical_crossentropy',
    metrics=['accuracy'], epochs=10)

## Similar to our `labels` object, `digitProbability` has 10 features but this
## time each feature represents a probability for the label at that index.  For
## our prediction, we will use the index with the maximum probability. Then we
## can see how our simple neural net performed on our test set.
def maximumProbabilty(pt):
    maximum = max(pt)
    return list(pt).index(maximum)

predictions = digitProbability.points.calculate(maximumProbabilty)
print(nimble.calculate.fractionCorrect(testY, predictions))

## Some older Keras versions rely on sources of randomness outside of Nimble's
## control, so your exact result could vary from ours, but should be around 90%
## accurate. This is pretty good for a simple model only trained for 10 epochs.

## Convolutional Neural Network ##

## Let's try to do better by increasing the complexity and create a 2D
## convolutional neural network this time. For this we will need to reshape our
## trainX and testX data so that Keras knows that each point is a 16 x 16
## single channel image.
def reshapePoint(pt):
    ret = pt.copy()
    ret.unflatten((16, 16, 1))
    return ret

trainX = trainX.points.calculate(reshapePoint)
testX = testX.points.calculate(reshapePoint)

## We need even more Keras objects as layers for our 2D convolutional neural
## network, so again we will use `nimble.Init`.
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
predictionsCNN = probabilityCNN.points.calculate(maximumProbabilty)
print(nimble.calculate.fractionCorrect(testY, predictionsCNN))

## **References:**

## Semeion Research Center of Sciences of Communication, via Sersale 117,
## 00128 Rome, Italy
## Tattile Via Gaetano Donizetti, 1-3-5,25030 Mairano (Brescia), Italy

## Dua, D. and Graff, C. (2019).
## UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
## Irvine, CA: University of California, School of Information and Computer Science.

## Link to original dataset:
## https://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit
