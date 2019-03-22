
from __future__ import absolute_import

from six.moves import range

import UML
import sys
import os.path
import numpy
import matplotlib
from matplotlib import pyplot as plt

import PIL
from PIL import Image

NO_Y_LABEL = False

rootFolder = "/home/tpburns/sparkwave/data/gender/3rd_round/user_arrow"
fileName = os.path.join(rootFolder, 'arrow')


#matplotlib.rc('font', size=12)

# setup the figure that we will be using later
W = 960.0
H = 240
DPI = 96.0
DPISAVE = 144.0
DEMONSTRATE = True
DEMONSTRATE_PATH = "/home/tpburns/sparkwave/data/gender/3rd_round/plots/Aesthetic"
DEMONSTRATE_FILE = "Aesthetic"

#W = 960.0
#H = 432.0
#DPI = 96.0
#DPISAVE = 144.0

#wanted = [-9.99]
#wanted = [.5 * i for i in xrange(-20,21)]
#wanted += [.5 * i for i in range(-19,20)]
#wanted += [9.99]
#wanted = [-9.99]

#wanted = [-3]
#wanted += [.25 * i for i in range(-11,12)]
wanted = [.5 * i  for i in range(-6,7)]
print wanted
#wanted += [3]

dummyData = [9] * 13
xVals = numpy.array([i for i in range(-6, 7)])

for arrowX in wanted:
    fig = plt.figure(facecolor='white', figsize=(W/DPI, H/DPI), tight_layout=True)
    ax = plt.axes()
    ax.set_frame_on(False)
#    ax.grid(True)

    plt.plot(xVals, dummyData, marker='.', linestyle='-', markersize=10)

    plt.xlim(-6.6, 6.6)
    plt.ylim(0, 10)

    tickNames = [-3, "", -2, "", -1, "", 0, "", 1, "", 2, "", 3]
    loc, labels = plt.xticks(xVals, tickNames)
    ytickVals = [0,9]
    ytickStr = ['0%', '90%']
    plt.yticks(ytickVals, ytickStr)
    plt.ylabel("% of resp", fontweight='bold', fontsize=12)

    # For Uparrow. Yes, really, you can check with grid lines turned on
#    arrowLeftShift = 0.104
    # for uparrow.
    arrowLeftShift = 0.06

    loc = ((arrowX*2)-arrowLeftShift, 5)
#    print loc
#    xOffset = 0.8
    xOffset = 0.25
    yOffset = 1
    textLoc = (loc[0]+xOffset, loc[1]) if loc[0] < 0 else (loc[0]-xOffset, loc[1])
    textAlignment = "left" if loc[0] < 0 else "right"

    plt.plot(loc[0], loc[1], marker='$\uparrow$', color="Black", markersize=20)
    plt.annotate("Your Score", xy=(loc[0],loc[1]), xytext=(textLoc[0], textLoc[1]), color="Black", horizontalalignment=textAlignment, verticalalignment='top')

#    print (str(plt.xlim()))

    arrowX = round(arrowX) if abs(arrowX) == 9.99 else arrowX
    _name = "_".join(str(arrowX).split("."))
    currName = os.path.join(rootFolder, _name + ".png")
    plt.savefig(currName, dpi=DPISAVE)
    plt.close()
#    continue

    # use pillow to crop the useless top and bottom white space / X axis frame
    toCrop = PIL.Image.open(currName)
    box = (0, 140, 1440, 190)
    tight = toCrop.crop(box)

    # grab a vertical white bar to cover over the Y axis frame
    if NO_Y_LABEL:
        whiteVertBox = (0,0,5,50)
    else:
        whiteVertBox = (1410,0,1440,50)
    whiteVertBar = tight.crop(whiteVertBox)

    # paste over the left and right portions of the frame
    if NO_Y_LABEL:
        targetBoxL1 = (45,0,50,50)
        targetBoxL2 = (45,0,50,50)
        targetBoxR = (1400,0,1405,50)
    else:
        targetBoxL1 = (20,0,50,50)
        targetBoxL2 = (31,0,61,50)
        targetBoxR = (1400,0,1430,50)
    tight.paste(whiteVertBar, targetBoxL1)
    tight.paste(whiteVertBar, targetBoxL2)
    tight.paste(whiteVertBar, targetBoxR)

    if arrowX == 10:
        flipBox = (1375,0,1400,50)
        flip = tight.crop(flipBox)
        flipT = flip.transpose(Image.FLIP_LEFT_RIGHT)
        flipTBox = (1400,0,1425,50)
        tight.paste(flipT, flipTBox)

    if arrowX == -10:
        if NO_Y_LABEL:
            flipBox = (50,0,75,50)
            flipTBox = (25,0,50,50)
        else:
            flipBox = (61,0,86,50)
            flipTBox = (36,0,61,50)
        flip = tight.crop(flipBox)
        flipT = flip.transpose(Image.FLIP_LEFT_RIGHT)
        tight.paste(flipT, flipTBox)

    tight.save(currName, "png")


    if DEMONSTRATE:
        toExtend = toCrop = PIL.Image.open(DEMONSTRATE_PATH + ".png")
        extended = toExtend.resize((1440,708))
        extended.paste(toExtend)
        pasteBox = (0, 648, 1440, 698)
        extended.paste(tight, pasteBox)
        saveLoc = os.path.join(rootFolder, DEMONSTRATE_FILE +'_' + _name + ".png")
#        print (saveLoc)
        extended.save(saveLoc, "png")
