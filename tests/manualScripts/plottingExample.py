"""
Script demonstrating the various options available for plotting nimble
data objects. By default, outputs plots to the screen. If a directory
is passed as an argument to the script, then plots will instead
be written as files in that directory
"""

import os
import sys

import numpy as np

import nimble

def getOutPath(outDir, outName):
    outPath = None
    if outDir is not None:
        outPath = os.path.join(outDir, outName)
    return outPath

if __name__ == "__main__":

    # if a directory is given, plots will be output to file in that location.
    givenOutDir = None
    if len(sys.argv) > 1:
        givenOutDir = sys.argv[1]

    # plots will not show when show value set to False
    givenShow = True
    if len(sys.argv) > 2:
        givenShow = True if sys.argv[2].lower() != "false" else False

    rawNorm = np.random.normal(loc=0, scale=1, size=(1000, 1))
    objNorm = nimble.data("Matrix", rawNorm, featureNames=["N(0,1)"])

    # 1000 samples of N(0,1)
    def plotDistributionNormal(plotObj, outDir, givenShow):
        outPath = getOutPath(outDir, "NormalDistribution")

        plotObj.plotFeatureDistribution(0, outPath=outPath, show=givenShow)

    # 1000 samples of N(0,1) shifted left and right
    def plotDistributionCompare(plotObj, outDir):
        outPath = getOutPath(outDir, "DistributionCompare")

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
    def plotDistributionNormalSquared(plotObj, outDir, givenShow):
        plotObj **= 2
        plotObj.features.setName(0, "N(0,1) squared")

        outPath = getOutPath(outDir, "NormalSquared")
        plotObj.plotFeatureDistribution(0, outPath=outPath, show=givenShow)

    #compare two columns: col1= rand noise, col2 =3* col1 + noise
    def plotComparisonNoiseVsScaled(outDir, givenShow):
        raw1 = np.random.rand(50, 1)
        raw2 = np.random.rand(50, 1)
        scaled = (raw1 * 3) + raw2
        obj1 = nimble.data("Matrix", raw1)
        obj2 = nimble.data("Matrix", scaled)
        obj1.features.append(obj2)

        #obj1.features.setName(0, "[0, 1) random noise")
        obj1.features.setName(1, "(Feature 0 * 3) + noise")
        obj1.name = "Noise"

        outPath = getOutPath(outDir, "NoiseVsScaled")
        obj1.plotFeatureAgainstFeature(0, 1, trend='linear', outPath=outPath,
                                       show=givenShow, xMin=-0.2, xMax=1.2,
                                       yMin=-0.2, yMax=5.2)

    def plotComparisonMultipleNoiseVsScaled(outDir, givenShow):
        raw1 = np.random.rand(50, 3)
        raw1[:, 1] = raw1[:, 0] + raw1[:, 1]
        raw1[:, 2] = raw1[:, 0] + raw1[:, 2]
        obj1 = nimble.data("Matrix", raw1)

        obj1.plotFeatureAgainstFeatureRollingAverage(
            0, 1, 10, outPath=None, show=False, figureName='noise', label='1')

        title = 'Feature 0 Vs Feature 0 + Noises 1 and 2'
        outPath = getOutPath(outDir, "NoiseVsMultipleScaled")
        obj1.plotFeatureAgainstFeatureRollingAverage(
            0, 2, 10, outPath=outPath, show=givenShow, figureName='noise',
            label='2', title=title, yAxisLabel='')

    checkObj = nimble.data("Matrix", np.zeros((15, 12)), name="Checkerboard")

    def plotClusters(outDir, givenShow):
        centers = nimble.data('Matrix', [[0, 0], [1, 1], [2, 2], [3, 3]])

        obj = nimble.data("Matrix", np.random.rand(25, 2))

        label = 'cluster0'
        # by setting yMin=0.5, it will trigger yMax to be set to ~1 for
        # this first plot and for the entire figure. We want to test that
        # our helpers allow the yMax value to remain dynamic (expanding to
        # ~4) when adding the plots below while keeping our defined yMin.
        obj.plotFeatureAgainstFeature(0, 1, yMin=0.5, show=False,
                                      figureName='fig', label=label)

        for i in range(1, 4):
            if i < 3:
                show = False
                outPath = None
            else:
                show = givenShow
                outPath = getOutPath(outDir, "Clusters")

            obj = nimble.data("Matrix", np.random.rand(25, 2))
            scaled = obj + centers.points[i].stretch

            label = 'cluster' + str(i)
            scaled.plotFeatureAgainstFeature(0, 1, show=show, outPath=outPath,
                                             figureName='fig', label=label)

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
    def plotCheckerboad(plotObj, outDir, givenShow):
        outPath = getOutPath(outDir, "checkerboard")
        plotObj.plotHeatMap(outPath=outPath, show=givenShow)

    # heatmap: another one to show a gradient from a given formula (linear?)
    def plotCheckeredGradient(plotObj, outDir, givenShow):
        outPath = getOutPath(outDir, "checkerboardWithBar")
        plotObj.plotHeatMap(includeColorbar=True, outPath=outPath,
                            show=givenShow)

    groups = [['G1', 'W', 25], ['G1', 'M', 20], ['G1', 'W', 32],
              ['G1', 'M', 29], ['G1', 'M', 26], ['G1', 'M', 31],
              ['G2', 'M', 33], ['G2', 'W', 32], ['G2', 'M', 25],
              ['G2', 'W', 27], ['G3', 'M', 30], ['G3', 'W', 34],
              ['G3', 'W', 29], ['G3', 'W', 28], ['G3', 'W', 28]]

    features = ['GroupID', 'Gender', 'Score']
    groupObj = nimble.data('Matrix', groups, featureNames=features)

    def plotGroupMeans(plotObj, outDir, givenShow):
        outPath = getOutPath(outDir, "groupMeansCI")

        plotObj.plotFeatureGroupMeans(feature=2, groupFeature=0, ecolor='g',
                                      xAxisLabel=False,  yAxisLabel=False,
                                      outPath=outPath, show=givenShow)

    def plotGroupStats(plotObj, outDir, givenShow):
        outPath = getOutPath(outDir, "groupCounts")

        plotObj.plotFeatureGroupStatistics(
            nimble.calculate.sum, feature=2, groupFeature=0, show=False,
            figureName='count', color='y', alpha=0.6)
        plotObj.plotFeatureGroupStatistics(
            nimble.calculate.sum, feature=2, groupFeature=0,
            subgroupFeature=1, outPath=outPath, show=givenShow,
            figureName='count', edgecolor='k')

        plotObj.plotFeatureGroupStatistics(
            nimble.calculate.count, feature=1, groupFeature=0,
            subgroupFeature=1, outPath=outPath, show=givenShow,
            figureName='count', color=['black', 'yellow'])
        # Strange example with given data but used to highlight two things:
        # 1) Proper bar spacing when the subgroup does not contain all of the
        # unique values in the in the subgroup feature
        # 2) The color spectrum does not repeat with more than 10 subgroup bars
        plotObj.plotFeatureGroupStatistics(
            nimble.calculate.count, feature=2, groupFeature=0, show=False,
            figureName='count', color='y', alpha=0.6)
        plotObj.plotFeatureGroupStatistics(
            nimble.calculate.count, feature=2, groupFeature=0,
            subgroupFeature=2, outPath=outPath, show=givenShow,
            figureName='count', edgecolor='k')

    def plotFeatureSumsManual(plotObj, outDir, givenShow):
        outPath = getOutPath(outDir, "groupSumsManual")

        groups = plotObj.groupByFeature(0)
        counts = {}
        for name, group in groups.items():
            count = [None, None]
            subgroups = group.groupByFeature(0)
            for subname, subgroup in subgroups.items():
                if subname == 'W':
                    count[0] = sum(subgroup)
                else:
                    count[1] = sum(subgroup)
            counts[name] = count

        countObj = nimble.data('Matrix', counts, rowsArePoints=False)
        countObj.points.setNames(['W', 'M'])
        countObj.features.plot(outPath=outPath, show=givenShow,
                               yAxisLabel='sum', title='Score sum by GroupID',
                               legendTitle='Gender')

    def plotFeatureComparisons(outDir, givenShow):
        d = np.random.rand(500, 6) * 2
        d[:, 1] /= 2
        d[:, 2] *= 2
        d[:, 3] += 0.5
        d[:, 4] = 1
        d[:250, 5] = -2
        d[250:, 5] = 4
        plotObj = nimble.data("Matrix", d)
        plotObj.name = 'MAD'

        pathCI = getOutPath(outDir, "featureMeansCI")
        plotObj.features.plotMeans(horizontal=True, show=False, outPath=pathCI)

        pathBar = getOutPath(outDir, "featureMAD")
        # show x tick label rotation when labels too long
        plotObj.features.plotStatistics(
            nimble.calculate.medianAbsoluteDeviation, outPath=pathBar,
            show=givenShow, yAxisLabel='MAD', color='orange')

    def plotPointComparison(outDir, givenShow):
        d = np.random.rand(5, 10)
        d[d < 0.5] = d[d < 0.5] + 0.5 # value range (0.5 - 1)
        d[0, 0] = 0.1
        d[1, :] = 0.75
        d[2, :] /= 2
        d[3, :] = [0.8 if v < 0.8 else v for v in d[3, :]]
        pnames = ['Patient #' + str(i) for i in range(1, 6)]
        fnames = ['Trial #' + str(i) for i in range(1, 11)]
        plotObj = nimble.data("DataFrame", d, pointNames=pnames,
                              featureNames=fnames)
        plotObj.name = 'Patient Trials'

        pathBar = getOutPath(outDir, "pointMinMax")
        # stat function returning multiple values
        def minmax(pt):
            return nimble.data('DataFrame', [min(pt), max(pt)],
                               featureNames=['min', 'max'])

        plotObj.points.plotStatistics(minmax, outPath=pathBar,
            show=givenShow, yAxisLabel='Effectiveness', color=['red', 'blue'])


    # # one function for each plot being drawn.
    plotDistributionNormal(objNorm, givenOutDir, givenShow)
    plotDistributionCompare(objNorm, givenOutDir)
    plotDistributionNormalSquared(objNorm, givenOutDir, givenShow)
    plotComparisonNoiseVsScaled(givenOutDir, givenShow)
    plotComparisonMultipleNoiseVsScaled(givenOutDir, givenShow)
    plotClusters(givenOutDir, givenShow)
    plotCheckerboad(checkObj, givenOutDir, givenShow)
    plotCheckeredGradient(checkGradObj, givenOutDir, givenShow)
    plotGroupMeans(groupObj, givenOutDir, givenShow)
    plotGroupStats(groupObj, givenOutDir, givenShow)
    plotFeatureSumsManual(groupObj, givenOutDir, givenShow)
    plotFeatureComparisons(givenOutDir, givenShow)
    plotPointComparison(givenOutDir, givenShow)
