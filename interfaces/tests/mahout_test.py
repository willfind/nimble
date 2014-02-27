"""
Unit tests for mahout_interface.py

"""

import UML
import os

#from UML.interfaces.mahout_interface import setMahoutLocation
#from UML.interfaces.mahout_interface import getMahoutLocation
import tempfile

# TODO re enable 
def MahoutHandmadeOutput():
	""" Test mahout() by running kmeans with known output """
	
	setMahoutLocation('/home/tpburns/Dropbox/ML_intern_tpb/workspace/mahout-distribution-0.7')
	
	trainingIn = tempfile.NamedTemporaryFile()
	trialIn = tempfile.NamedTemporaryFile()
	actualOut = tempfile.NamedTemporaryFile()

	trainingIn.write("1,1,1\n")
	trainingIn.write("1,2,1\n")
	trainingIn.write("3,3,2\n")
	trainingIn.flush()
	trialIn.write("3,4\n")
	trialIn.flush()
	args = { 'numClusters':'2' }
	UML.run("mahout.Kmeans", trainingIn.name, trainY=None, testX=trialIn.name, output=actualOut.name, arguments=args)
	
	actualOut.seek(0)
	line = actualOut.readline()
	print line
	assert line.strip() == "3,2,5.0"


# TODO re enable 
def MahoutTasteHandmadeOutput():
	""" Test mahout() by running a command against the taste patch with known output """

	setMahoutLocation('/home/tpburns/Dropbox/ML_intern_tpb/workspace/mahout-distribution-0.7')

	trainingIn = tempfile.NamedTemporaryFile()
	trialIn = tempfile.NamedTemporaryFile()
	actualOut = tempfile.NamedTemporaryFile()

	trainingIn.write("1,2,5.0\n")
	trainingIn.write("3,4,1.0\n")
	trainingIn.flush()
	trialIn.write("3,2\n")
	trialIn.flush()
	args = {"trainingInput":trainingIn.name, "trialInput":trialIn.name, "output":actualOut.name, "recommender":'ia'}
	UML.run("mahout.tasteBasedEstimation",args)
	
	actualOut.seek(0)
	line = actualOut.readline()
	print line
	assert line.strip() == "3,2,5.0"


# TODO re enable 
def MahoutTasteBasedPatchIntegrity():
	"""
	Test mahoutTasteRecommenderEstimation() for maintaining the integrety of
	the files it patches over.

	"""

	setMahoutLocation('/home/tpburns/Dropbox/ML_intern_tpb/workspace/mahout-distribution-0.7')

	trainingIn = tempfile.NamedTemporaryFile()
	trialIn = tempfile.NamedTemporaryFile()
	actualOut = tempfile.NamedTemporaryFile()
	trainingIn.write("1,2,5.0\n")
	trainingIn.write("3,4,1.0\n")
	trainingIn.flush()
	trialIn.write("3,2\n")
	trialIn.flush()

	propsOrig = open(getMahoutLocation() + '/src/conf/driver.classes.props')
	tempProps = tempfile.TemporaryFile()
	for line in propsOrig:
		tempProps.write(line)	

	batchPresent = True	
	try:	
		batchOrig = open(getMahoutLocation() + '/core/src/main/java/org/apache/mahout/cf/taste/common/TasteRecommenderBatchEstimation.java')
		tempBatch = tempfile.TemporaryFile()
		for line in batchOrig:
			tempBatch.write(line)
	except IOError:
		batchPresent = False	
	
	args = {"trainingInput":trainingIn.name, "trialInput":trialIn.name, "output":actualOut.name,
			"recommender":'ia'}
	UML.run('mahout.tasteBasedEstimation',args,'')

	tempProps.seek(0)
	propsPost = open(getMahoutLocation() + '/src/conf/driver.classes.props')

	for lineExpected in tempProps:
		linePost = propsPost.readline()
		assert lineExpected == linePost

	if batchPresent:
		tempBatch.seek(0)
		batchPost = open(getMahoutLocation() + '/core/src/main/java/org/apache/mahout/cf/taste/common/TasteRecommenderBatchEstimation.java')

		for lineExpected in tempBatch:
			linePost = batchPost.readline()
			assert lineExpected == linePost


# TODO re enable 
def MahoutLocation():
	""" Test setMahoutLocation() and the default mahoutDir value """

	if "MAHOUT_HOME" in os.environ:
		assert getMahoutLocation() == os.environ["MAHOUT_HOME"]

	path = '/test/path'
	setMahoutLocation(path)

	assert getMahoutLocation() == path


# TODO re enable 
def MahoutPresent():
	""" Test mahoutPresent() will return false for obviously wrong path values """

	# default is none - should be false
	setMahoutLocation(None)
	assert not mahoutPresent()

	# pathes which are not directories - should be false
	setMahoutLocation('')
	assert not mahoutPresent()
	setMahoutLocation('/bin/bash')
	assert not mahoutPresent()
	
	# non mahout directory
	setMahoutLocation('/home')
	assert not mahoutPresent()


# TODO re enable 
def MahoutListLearners():
	""" Test Mahout's listMahoutLearners() by checking the output for those learners we unit test """
	
	setMahoutLocation('/home/tpburns/Dropbox/ML_intern_tpb/workspace/mahout-distribution-0.7')
	
	ret = UML.listLearners('Mahout')


	assert 'kmeans' in ret
	assert 'parallelALS' in ret


