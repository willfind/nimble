"""
Script demonstrating the various options available for plotting nimble
data objects. By default, outputs plots to the screen. If a directory
is passed as an argument to the script, then plots will instead
be written as files in that directory
"""

import numpy
import os
import sys

import nimble

if __name__ == "__main__":

    # if a directory is given, plots will be output to file in that location.
    givenOutDir = None
    if len(sys.argv) > 1:
        givenOutDir = sys.argv[1]

    rawNorm = numpy.random.normal(loc=0, scale=1, size=(1000, 1))
    objNorm = nimble.data("Matrix", rawNorm, featureNames=["N(0,1)"])

    # 1000 samples of N(0,1)
    def plotDistributionNormal(plotObj, outDir):
        outPath = None
        if outDir is not None:
            outPath = os.path.join(outDir, "NormalDistribution")

        plotObj.plotFeatureDistribution(0, outPath=outPath)

    # 1000 samples of N(0,1)
    def plotDistributionCompare(plotObj, outDir):
        outPath = None
        if outDir is not None:
            outPath = os.path.join(outDir, "DistributionCompare")

        title = "Shifted Normal Distributions"
        alpha = 0.5
        shiftLeft = plotObj - 1
        shiftRight = plotObj + 1
        shiftLeft.plotFeatureDistribution(0, show=False, figureName='dists',
                                          alpha=alpha, label='shiftLeft')
        shiftRight.plotFeatureDistribution(0, outPath=outPath, title=title,
                                           show=False, figureName='dists',
                                           label='shiftRight', alpha=alpha,
                                           color='g')

    # 1000 samples of N(0,1), squared
    def plotDistributionNormalSquared(plotObj, outDir):
        plotObj **= 2
        plotObj.features.setName(0, "N(0,1) squared")

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
        obj1 = nimble.data("Matrix", raw1)
        obj2 = nimble.data("Matrix", scaled)
        obj1.features.append(obj2)

        #obj1.features.setName(0, "[0, 1) random noise")
        obj1.features.setName(1, "(Feature 0 * 3) + noise")
        obj1.name = "Noise"

        outPath = None
        if outDir is not None:
            outPath = os.path.join(outDir, "NoiseVsScaled")
        obj1.plotFeatureAgainstFeature(0, 1, outPath=outPath)

    def plotComparisonMultipleNoiseVsScaled(outDir):
        import numpy
        raw1 = numpy.random.rand(50, 3)
        raw1[:, 1] = (raw1[:, 0] * 3) + raw1[:, 1]
        raw1[:, 2] = (raw1[:, 0] * 3) + raw1[:, 2]
        obj1 = nimble.data("Matrix", raw1)

        outPath = None
        if outDir is not None:
            outPath = os.path.join(outDir, "NoiseVsMultipleScaled")
        obj1.plotFeatureAgainstFeature(0, 1, outPath=None, show=False,
                                       figureName='noise', label='1')
        obj1.plotFeatureAgainstFeature(0, 2, outPath=outPath,
                                       figureName='noise', label='2',
                                       title='Feature 0 Vs 1 and 2',
                                       yAxisLabel='Features 1 and 2')

    checkObj = nimble.data("Matrix", numpy.zeros((15, 12)), name="Checkerboard")

    def makeCheckered(val, p, f):
        pParity = p % 2
        fParity = f % 2
        if pParity == fParity:
            return 0
        else:
            return 1

    checkObj.transformElements(makeCheckered)

    def addGradient(vector):
        n = len(vector)
        ret = []
        for i in range(n):
            val = vector[i]
            ret.append(val + i / 2.)
        return ret

    checkGradObj = checkObj.points.calculate(addGradient)
    checkGradObj = checkGradObj.features.calculate(addGradient)
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
    plotDistributionCompare(objNorm, givenOutDir)
    plotDistributionNormalSquared(objNorm, givenOutDir)
    plotComparisonNoiseVsScaled(givenOutDir)
    plotComparisonMultipleNoiseVsScaled(givenOutDir)
    plotCheckerboad(checkObj, givenOutDir)
    plotCheckeredGradient(checkGradObj,givenOutDir)
