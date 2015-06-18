"""
Script demonstrating the various options available for plotting UML
data objects. By default, outputs plots to the screen. If a directory
is passed as an argument to the script, then plots will instead
be written as files in that directory

"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
	import UML
	import numpy
	import sys
	import os

	outDir = None
	if len(sys.argv) > 1:
		outDir = sys.argv[1]

	rawNorm = numpy.random.normal(loc=0, scale=1, size=(1000,1))
	objNorm = UML.createData("Matrix", rawNorm, featureNames=["N(0,1)"])

	# 1000 samples of N(0,1)
	def plotDistributionNormal():
		outPath = None
		if outDir is not None:
			print "hello"
			outPath = os.path.join(outDir, "NormalDistribution")

		objNorm.plotFeatureDistribution(0, outPath=outPath)

	# 1000 samples of N(0,1), squared
	def plotDistributionNormalSquared():
		objNorm.elementwisePower(2)
		objNorm.setFeatureName(0,"N(0,1) squared")

		outPath = None
		if outDir is not None:
			outPath = os.path.join(outDir, "NormalSquared")
		objNorm.plotFeatureDistribution(0, outPath=outPath)


	#compare two columns: col1= rand noise, col2 =3* col1 + noise
	def plotComparisonNoiseVsScaled():
		raw1 = numpy.random.rand(50,1)
		raw2 = numpy.random.rand(50,1)
		scaled = (raw1 * 3) + raw2
		obj1 = UML.createData("Matrix", raw1)
		obj2 = UML.createData("Matrix", scaled)
		obj1.appendFeatures(obj2)

		#obj1.setFeatureName(0, "[0, 1) random noise")
		obj1.setFeatureName(1, "(Feature 0 * 3) + noise")
		obj1.name = "Noise"

		outPath = None
		if outDir is not None:
			outPath = os.path.join(outDir, "NoiseVsScaled")
		obj1.plotFeatureAgainstFeature(0,1, outPath=outPath)



	checkObj = UML.createData("Matrix", numpy.zeros((15,12)), name="Checkerboard")
	
	def makeCheckered(val, p, f):
		pParity = p % 2
		fParity = f % 2
		if pParity == fParity:
			return 0
		else:
			return 1
	checkObj.applyToElements(makeCheckered)

	def addGradient(vector):
		n = len(vector)
		ret = []
		for i in range(n):
			val = vector[i]
			ret.append(val + i/2.)
		return ret

	checkGradObj = checkObj.applyToPoints(addGradient, inPlace=False)
	checkGradObj = checkGradObj.applyToFeatures(addGradient, inPlace=False)
	checkGradObj.name = "Checkerboard with linear gradient"

	# plot
	#heatMap: checkboard pattern, even columns 0s, odds = 1's, offset every other point
	def plotCheckerboad():
		outPath = None
		if outDir is not None:
			outPath = os.path.join(outDir, "checkerboard")
		checkObj.plot(outPath=outPath)

	# heatmap: another one to show a gradient from a given formula (linear?)
	def plotCheckeredGradient():
		outPath = None
		if outDir is not None:
			outPath = os.path.join(outDir, "checkerboardWithBar")
		checkGradObj.plot(includeColorbar=True, outPath=outPath)

	# one function for each plot being drawn.
	plotDistributionNormal()
	plotDistributionNormalSquared()
	plotComparisonNoiseVsScaled()
	plotCheckerboad()
	plotCheckeredGradient()
