

"""
Select the best questions to keep in a survey from a larger set of questions
"""
import numpy

from allowImports import boilerplate
boilerplate()

import os.path
import UML
from UML import trainAndTest
from UML import train
from UML import trainAndApply
from UML import configuration
from UML import createData
from UML.calculate import fractionIncorrect
from UML import calculate
import tableString


from UML.customLearners import CustomLearner


class MajorityVote(CustomLearner):
	learnerType = 'classification'

	def train(self, trainX, trainY):
		counts = {}
		def counter(v):
			if v in counts:
				counts[v] += 1
			else:
				counts[v] = 0
		trainY.applyToElements(counter)
		mostCommonKey = None
		highestCount = 0
		for key, count in counts.iteritems():
			if count > highestCount:
				count = highestCount
				mostCommonKey = key
		self.mode = mostCommonKey

	def apply(self, testX):
		raw = numpy.zeros(testX.pointCount)
		numpy.ndarray.fill(raw, self.mode)

		ret = UML.createData("Matrix", raw)
		ret.transpose()
		return ret



def per(x):
	return str(int(round(x*100,0))) + "%"

def functionName(function):
	return str(function).split("at")[0].split("function")[1].strip()

if __name__ == "__main__":
	#columns are: Deliberation,Stimulation,Self-identification of introversion
	fileName = "data from second study - introversion - stimulation - deliberation.csv"
	
	yLabel = "to predict"
	data = createData("List", fileName, featureNames=0)
	data.setFeatureName("Self-identification of introversion", yLabel)


	#I can't get this to work. Not only does calling train for this algorithm after not seem to work,
	#but calling this seems to add the following line Custom.MajorityVote = __main__.MajorityVote
	#to configuration.ini which means that I can no longer run this program without getting a crash during import
	#UML.registerCustomLearner("Custom", MajorityVote)

	#settings = UML.configuration.loadSettings()
	#settings.set(section="RegisteredLearners", option="MajorityVote", value=MajorityVote)
	#sectionSettings = settings.get(section="RegisteredLearners", option="MajorityVote")
	#print "sectionSettings:", sectionSettings

	#get rid of blank rows
	data.extractPoints(lambda x: len(str(x["Deliberation"]).strip()) == 0)
	
	#make the data numerical
	data = data.copyAs("Matrix")

	#make the "Self-indentification of introversion" into a binary feature
	def toBinaryDefinitionOfIntrovert(v):
		#scale is 3 to 15
		#if v >= 12: return 1 #they said 4 or 5 (on a 1=strongly disagree, 3=neither agree nor disagree, 5=strongly agree) scale on all three self-identification ?'s
		if v > 9: return 1 #they gave a value higher than the middle value (answering 3 on all three questions)
		return 0
	data.applyToElements(lambda x: toBinaryDefinitionOfIntrovert(x), features=yLabel)

	#get the training and testing data
	trainX, trainY, testX, testY = data.trainAndTestSets(testFraction=.2, labels=yLabel)

	CValue = 10**8 #we need this really large to turn off regulization

	def introvertsOnly(x):
		return x[yLabel] == 1 #this filters out extraverts
	def extravertsOnly(x):
		return x[yLabel] == 0 #this filters out intraverts

	featuresToKeep = ["none", "Stimulation", "Deliberation", "all"]
	filterFunctionsForTestPoints = [None, introvertsOnly, extravertsOnly]

	headers = ["", "Misclassified %", "In sample misclassified %", "Intercept", "Coefs"]
	output = []
	output.append(headers)

	trainYCopy = trainY.copy()

	for filterFunctionForTestPoints in filterFunctionsForTestPoints:
		subjectsStr = ""
		if filterFunctionForTestPoints == None: 
			subjectsStr += "allSubjects"
		else:
			subjectsStr += functionName(filterFunctionForTestPoints)

		output.append(["",""])
		output.append(["--"+subjectsStr+"--",""])

		for featureToKeep in featuresToKeep:

			trainXTemp = trainX.copy()
			testXTemp = testX.copy()

			#remove some of the features
			if featureToKeep != "all" and featureToKeep != "none":
				trainXTemp = trainXTemp.extractFeatures(featureToKeep)	
				testXTemp = testXTemp.extractFeatures(featureToKeep)

			#get rid of some of the points in the test set X and Y based on a filter
			Z = testXTemp.copy()
			Z.appendFeatures(testY)
			ZInSample = trainXTemp.copy()
			ZInSample.appendFeatures(trainYCopy.copy())

			if filterFunctionForTestPoints != None:
				Z = Z.extractPoints(filterFunctionForTestPoints)
				#print "ZInSample 3\n", ZInSample
				ZInSample = ZInSample.extractPoints(filterFunctionForTestPoints)

			testYTemp = Z.extractFeatures(yLabel)
			testXTemp = Z.copy()
			inSampleY = ZInSample.extractFeatures(yLabel)
			inSampleX = ZInSample

			name = ""
			if featureToKeep == "all": 
				name += "All Features"
			elif featureToKeep == "none":
				name += "Majority Vote"
			else:
				name += "Only " + str(featureToKeep)

			#misclassified = trainAndTest("SciKitLearn.LogisticRegression", trainX=trainXTemp, trainY=trainY, testX=testXTemp, testY=testYTemp, performanceFunction=fractionIncorrect, C=CValue)
			#print "trainX", trainX
			if featureToKeep != "none":
				classifier = train("SciKitLearn.LogisticRegression", trainX=trainXTemp, trainY=trainY, C=CValue)
			
				attributes = classifier.getAttributes()
				intercept = classifier.backend.intercept_
				coefs = classifier.backend.coef_

				misclassifiedInSample = classifier.test(testX=inSampleX, testY=inSampleY, performanceFunction=fractionIncorrect)
				#predictions = classifier.apply(testX=testXTemp, testY=testYTemp, performanceFunction=fractionIncorrect)
				misclassified = classifier.test(testX=testXTemp, testY=testYTemp, performanceFunction=fractionIncorrect)

			else:
				#classifier = train("Custom.MajorityVote", trainX=trainXTemp, trainY=trainY)
				#UML.deregisterCustomLearner("Custom", "MajorityVote")
				coefs = None
				intercept = None
				#predictions = None
				misclassifiedInSample = inSampleY.featureStatistics("mean")[0,0]
				misclassified = testYTemp.featureStatistics("mean")[0,0]


			#print "\nROUND"
			#print subjectsStr
			#print name
			#print "misclassified %:", misclassified
			#if predictions != None:
			#	print "predictions:\n", predictions
			#	print predictions.featureStatistics("mean")

			output.append([name, per(misclassified), per(misclassifiedInSample), intercept, coefs])



	print "\nResult:\n"
	print tableString.tableString(output)
	print "\n\n"



