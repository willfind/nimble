"""
Contains the functions to be used for in-script calls to the external Mahout
package

"""


#TODO redo output capture
#TODO setup default temporary files/directories

import subprocess
import os
import os.path

import UML
from UML.exceptions import ArgumentException


patchDir = os.path.dirname(__file__) + "/mahout_patches"
mahoutDir = os.environ["MAHOUT_HOME"] if "MAHOUT_HOME" in os.environ else None

def setMahoutLocation(path):
	""" Sets the location of the root directory of the mahout installation to be used """
	global mahoutDir	
	mahoutDir = path

def getMahoutLocation():
	return mahoutDir

def mahout(algorithm, trainData, testData, dependentVar=None, arguments={}, output=None, timer=None):
	"""

	"""

	if not mahoutPresent():
		print('Cannot find mahout, please call setMahoutLocation() with the path of the Mahout root directory')
		return

	raise NotImplementedError("Mahout interface is currently not in a usable state")

#	if redirectOutputTarget == '':
#		dumpFile = tempfile.NamedTemporaryFile()
#		redirectOutputTarget = dumpFile.name

	if algorithm == 'tasteBasedEstimation':
		mahoutTasteRecommenderEstimation(arguments) #,redirectOutputTarget)
		return

	if isinstance(trainData, UML.data.BaseData):
		raise ArgumentException("Must call mahout with paths to input files, not data objects")
	if isinstance(testData, UML.data.BaseData):
		raise ArgumentException("Must call mahout with paths to input files, not data objects")
	if isinstance(dependentVar, UML.data.BaseData):
		raise ArgumentException("dependentVar must be the name of the dependent var in trainData, a vector is not allowed")

	cmds = []

	cmd1 = mahoutDir + '/bin/mahout -core ' + algorithm
	
	# comand line argument packing	
	if trainData is not None:
		cmd1 += ' --input ' + trainData
	if dependentVar is not None:
		cmd1 += ' --target ' + dependentVar
	if output is not None:
		cmd1 += ' --output ' + output 

	for key in arguments.keys():
		cmd1 += ' --' + key + ' ' + arguments[key]
#	if redirectOutput is not None:
#		print "TRYING TO REDIRECT"
#		cmd1 += ' > ' + redirectOutputTarget  + ' 2>&1'
	cmds.append(cmd1)

	for cmd in cmds:
		subprocess.call(cmd, shell=True)


def mahoutTasteRecommenderEstimation(trainData, testData, output, arguments): #,redirectOutputTarget=None):
	"""
	Function to call an extension to Mahout to use Taste (Sequential) Recommenders
	to predict ratings. Extension is achieved by temporarily patching Mahout.
	"""	
	if "recommender" not in argsDict:
		raise ArgumentException("When calling the taste based recomenders, must include the " +
								"argument 'recommender' specifiying which one to use")

	recommenderType = argsDict["recommender"]
	del argsDict["recommender"]

	#mv driver props to temp
	cmd = 'mv '
	cmd += mahoutDir + '/src/conf/driver.classes.props '
	cmd += mahoutDir + '/src/conf/driver.classes.props.temp'
	subprocess.call(cmd,shell=True)

	# mv our driver.props in
	cmd = 'cp '
	cmd += patchDir + '/driver.classes.props.taste '
	cmd += mahoutDir + '/src/conf/driver.classes.props'
	subprocess.call(cmd,shell=True)

	# backup batch estimation if it exists. Will do nothing if the file isn't present
	path = mahoutDir + '/core/target/classes/org/apache/mahout/cf/taste/common/TasteRecommenderBatchEstimation.class'
	try:	
		batchOrig = open(path)
	except IOError:
		path = ''
	cmd = 'mv '
	cmd += path + ' '
	cmd += mahoutDir + '/core/target/classes/org/apache/mahout/cf/taste/common/TasteRecommenderBatchEstimationTEMP.class'
	if path:
		subprocess.call(cmd,shell=True)

	#mv our batch est in
	cmd = 'cp '
	cmd += patchDir + '/TasteRecommenderBatchEstimation.class '
	cmd += mahoutDir + '/core/target/classes/org/apache/mahout/cf/taste/common/TasteRecommenderBatchEstimation.class'
	subprocess.call(cmd,shell=True)

	#call mahout
	cmd = mahoutDir + '/bin/mahout -core tasteBasedEstimation'
	cmd += ' --recommender ' + recommenderType
	for key in argsDict.keys():
		cmd += ' --' + str(key) + ' ' + str(argsDict[key])
	if redirectOutputTarget is not None:
		print "TRYING TO REDIRECT"
		cmd += ' > ' + redirectOutputTarget + ' 2>&1'
	print cmd
	subprocess.call(cmd,shell=True)

	#undo batch
	cmd = 'rm '
	cmd += mahoutDir + '/core/target/classes/org/apache/mahout/cf/taste/common/TasteRecommenderBatchEstimation.class'
	subprocess.call(cmd,shell=True)

	# restore batch est backup if it exists. Will do nothing if the file isn't present
	path = mahoutDir + '/core/target/classes/org/apache/mahout/cf/taste/common/TasteRecommenderBatchEstimationTEMP.class'	
	try:	
		batchOrig = open(path)
	except IOError:
		path = ''
	cmd = 'mv '
	cmd += path + ' '
	cmd += mahoutDir + '/core/target/classes/org/apache/mahout/cf/taste/common/TasteRecommenderBatchEstimation.class'
	if path:	
		subprocess.call(cmd,shell=True)

	#undo driver prosp
	cmd = 'mv '
	cmd += mahoutDir + '/src/conf/driver.classes.props.temp '
	cmd += mahoutDir + '/src/conf/driver.classes.props'
	subprocess.call(cmd,shell=True)


def mahoutPresent():
	""" Return true if we can establish that mahout is present """

	# check for nonesense values
	if mahoutDir is None or mahoutDir == '':
		return False

	# check that the path is to a valid directory
	if not os.path.isdir(mahoutDir):
		return False

	# check whether the mahout bash script is in the right place
	return os.path.isfile(mahoutDir + '/bin/mahout')


def listAlgorithms():
	"""
	Function to return a list of all algorithms callable through our interface, if mahout is present
	
	"""
	if not mahoutPresent():
		return []

	ret = []
	# look in /src/conf/ for a file for each algorithm.
	contents = os.listdir(mahoutDir + '/src/conf')
	for name in contents:
		if name == 'driver.classes.props':
			continue
		# split 
		nameList = name.split('.')
		ret.append(nameList[0])

	return ret

	

	
