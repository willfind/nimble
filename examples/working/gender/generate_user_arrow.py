
from __future__ import absolute_import
from .allowImports import boilerplate
from six.moves import range
boilerplate()

import UML
import os.path

matplotlib = UML.importModule("matplotlib")
if matplotlib:
    mlines = UML.importModule("matplotlib.mlines")
    mpatches = UML.importModule("matplotlib.mpatches")
    plt = UML.importModule("matplotlib.pyplot")

PIL = UML.importModule("PIL")
if PIL:
    Image = UML.importModule("PIL.Image")

NO_Y_LABEL = False

rootFolder = "/home/tpburns/gimbel_tech/data/gender/2nd_round_trial/user_arrow"
fileName = os.path.join(rootFolder, 'arrow')


matplotlib.rc('font', size=12)

# setup the figure that we will be using later
W = 960.0
#W = 1440
#H = 27.0
H = 120
DPI = 96.0
DPISAVE = 144.0

wanted = [-9.99]
#wanted = [.5 * i for i in xrange(-20,21)]
wanted += [.5 * i for i in range(-19,20)]
wanted += [9.99]
#wanted = [-9.99]
for arrowX in wanted:
    fig = plt.figure(facecolor='white', figsize=(W/DPI, H/DPI), tight_layout=True)
    ax = plt.axes()

#   ax.set_frame_on(False)

#   plt.xticks([])
    plt.yticks([])

    plt.xlim(-10, 10)
    plt.ylim(0, 10)

    loc = (arrowX, 5)
    xOffset = 0.8
    yOffset = 1
    textLoc = (loc[0]+xOffset, loc[1]) if loc[0] < 0 else (loc[0]-xOffset, loc[1])
    textAlignment = "left" if loc[0] < 0 else "right"

    plt.ylabel("of")
    plt.plot(loc[0], loc[1], marker='$\Downarrow$', color="Black", markersize=35)
    plt.annotate("Your Score", xy=(loc[0],loc[1]), xytext=(textLoc[0], textLoc[1]), color="Black", horizontalalignment=textAlignment, verticalalignment='center')

    arrowX = round(arrowX) if abs(arrowX) == 9.99 else arrowX
    _name = "_".join(str(arrowX).split("."))
    currName = os.path.join(rootFolder, _name + ".png")
    plt.savefig(currName, dpi=DPISAVE)
    plt.close()

    # use pillow to crop the useless top and bottom white space / X axis frame
    toCrop = PIL.Image.open(currName)
    box = (0, 40, 1440, 105)
    tight = toCrop.crop(box)

    # grab a vertical white bar to cover over the Y axis frame
    if NO_Y_LABEL:
        whiteVertBox = (0,0,5,65)
    else:
        whiteVertBox = (1410,0,1440,65)
    whiteVertBar = tight.crop(whiteVertBox)

    # paste over the left and right portions of the frame
    if NO_Y_LABEL:
        targetBoxL1 = (45,0,50,65)
        targetBoxL2 = (45,0,50,65)
        targetBoxR = (1400,0,1405,65)
    else:
        targetBoxL1 = (20,0,50,65)
        targetBoxL2 = (31,0,61,65)
        targetBoxR = (1400,0,1430,65)
    tight.paste(whiteVertBar, targetBoxL1)
    tight.paste(whiteVertBar, targetBoxL2)
    tight.paste(whiteVertBar, targetBoxR)

    if arrowX == 10:
        flipBox = (1375,0,1400,65)
        flip = tight.crop(flipBox)
        flipT = flip.transpose(Image.FLIP_LEFT_RIGHT)
        flipTBox = (1400,0,1425,65)
        tight.paste(flipT, flipTBox)

    if arrowX == -10:
        if NO_Y_LABEL:
            flipBox = (50,0,75,65)
            flipTBox = (25,0,50,65)
        else:
            flipBox = (61,0,86,65)
            flipTBox = (36,0,61,65)
        flip = tight.crop(flipBox)
        flipT = flip.transpose(Image.FLIP_LEFT_RIGHT)
        tight.paste(flipT, flipTBox)


    tight.save(currName, "png")
