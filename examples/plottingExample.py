"""
Script demonstrating the various options available for plotting UML
data objects. By default, outputs plots to the screen. If a directory
is passed as an argument to the script, then plots will instead
be written as files in that directory

"""

from __future__ import absolute_import
from __future__ import print_function
try:
    from allowImports import boilerplate
except:
    from .allowImports import boilerplate
from six.moves import range

boilerplate()

import numpy
import os
import sys


if __name__ == "__main__":
    import UML

    # if a directory is given, plots will be output to file in that location.
    givenOutDir = None
    if len(sys.argv) > 1:
        givenOutDir = sys.argv[1]
        print(givenOutDir)

    rawNorm = numpy.random.normal(loc=0, scale=1, size=(1000, 1))
    objNorm = UML.createData("Matrix", rawNorm, featureNames=["N(0,1)"])

    # 1000 samples of N(0,1)
    def plotDistributionNormal(plotObj, outDir):
        outPath = None
        if outDir is not None:
            print("hello")
            outPath = os.path.join(outDir, "NormalDistribution")

        plotObj.plotFeatureDistribution(0, outPath=outPath)

    # 1000 samples of N(0,1), squared
    def plotDistributionNormalSquared(plotObj, outDir):
        plotObj.elementwisePower(2)
        plotObj.setFeatureName(0, "N(0,1) squared")

        outPath = None
        if outDir is not None:
            outPath = os.path.join(outDir, "NormalSquared")
        plotObj.plotFeatureDistribution(0, outPath=outPath)


    #compare two columns: col1= rand noise, col2 =3* col1 + noise
    def plotComparisonNoiseVsScaled(outDir):
        import numpy
        raw1 = numpy.random.rand(50, 1)
        raw2 = numpy.random.rand(50, 1)
        scaled = (raw1 * 3) + raw2
        obj1 = UML.createData("Matrix", raw1)
        obj2 = UML.createData("Matrix", scaled)
        obj1.addFeatures(obj2)

        #obj1.setFeatureName(0, "[0, 1) random noise")
        obj1.setFeatureName(1, "(Feature 0 * 3) + noise")
        obj1.name = "Noise"

        outPath = None
        if outDir is not None:
            outPath = os.path.join(outDir, "NoiseVsScaled")
        obj1.plotFeatureAgainstFeature(0, 1, outPath=outPath)


    checkObj = UML.createData("Matrix", numpy.zeros((15, 12)), name="Checkerboard")

    def makeCheckered(val, p, f):
        pParity = p % 2
        fParity = f % 2
        if pParity == fParity:
            return 0
        else:
            return 1

    checkObj.transformEachElement(makeCheckered)

    def addGradient(vector):
        n = len(vector)
        ret = []
        for i in range(n):
            val = vector[i]
            ret.append(val + i / 2.)
        return ret

    checkGradObj = checkObj.calculateForEachPoint(addGradient)
    checkGradObj = checkGradObj.calculateForEachFeature(addGradient)
    checkGradObj.name = "Checkerboard with linear gradient"

    # plot
    #heatMap: checkboard pattern, even columns 0s, odds = 1's, offset every other point
    def plotCheckerboad(plotObj, outDir):
        outPath = None
        if outDir is not None:
            outPath = os.path.join(outDir, "checkerboard")
        plotObj.plot(outPath=outPath)

    # heatmap: another one to show a gradient from a given formula (linear?)
    def plotCheckeredGradient(plotObj, outDir):
        outPath = None
        if outDir is not None:
            outPath = os.path.join(outDir, "checkerboardWithBar")
        plotObj.plot(includeColorbar=True, outPath=outPath)

    # one function for each plot being drawn.
    plotDistributionNormal(objNorm, givenOutDir)
    plotDistributionNormalSquared(objNorm, givenOutDir)
    plotComparisonNoiseVsScaled(givenOutDir)
    plotCheckerboad(checkObj, givenOutDir)
    plotCheckeredGradient(checkGradObj,givenOutDir)
