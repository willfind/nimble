"""
Contains functions that generalize those in the individual interfaces 


"""


from mahout_interface import mahout
from regressors_interface import regressor
from scikit_learn_interface import sciKitLearn
from mlpy_interface import mlpy


# TODO can we do this in an automated way??

def run(package, algorithm, trainData, testData, output=None, dependentVar=None, arguments={}):
	if package == 'mahout':
		return mahout(algorithm, trainData, testData, output, dependentVar, arguments)
	if package == 'regressor':
		return regressors(algorithm, trainData, testData, output, dependentVar, arguments)
	if package == 'sciKitLearn':
		return sciKitLearn(algorithm, trainData, testData, output, dependentVar, arguments)
	if package == 'mlpy':
		return mlpy(algorithm, trainData, testData, output, dependentVar, arguments)



#list all (includeParams)



