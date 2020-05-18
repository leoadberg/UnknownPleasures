import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider, Button, TextBox
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.draw import line
from skimage.transform import resize

if len(sys.argv) != 2:
    print("Usage: python3 main.py image.png")
    exit(1)

##############
# PARAMETERS #
##############

SUPERSAMPLE = 2

horizontalsize = 500
verticalsize = 500
horizontalpad = 50
verticalpad = 50
thickness = 1
lines = 40
noise = 5
offsetscale = 5
sourceblur = 5
randomfrequency = 0.3

# Load source image as greyscale (just red channel)
originalsource = mpimg.imread(sys.argv[1])
if len(originalsource.shape) > 2:
    originalsource = originalsource[:,:,0]

fig = plt.figure()
ax = plt.subplot()
fig.subplots_adjust(bottom=0.5)

im = np.zeros((verticalsize + verticalpad * 2, horizontalsize + horizontalpad * 2))
im_ax = ax.imshow(im, cmap='gray', interpolation='bicubic', vmin=0, vmax=1.0)

######################################################
#                NOISE FUNCTION:                     #
# I tuned this by eye to resemble the original album #
# cover, might not be the best for other data        #
######################################################
t1 = 0
t2 = 0
def getnoise():
    global t1, t2
    t1 += (random.random() - 0.3) * randomfrequency
    t2 += (random.random() - 0.3) * randomfrequency
    return (1 + np.sin(t1)) * (1 + np.sin(1.5 * t2))

# For some reason np.clip() is slow on scalars
def clip(val, vmin, vmax):
    return int(vmin if val < vmin else vmax if val > vmax else val)

######################################################
#                 DRAW FUNCTION:                     #
# Very unoptimized, basically draws pixel-by-pixel.  #
######################################################

def draw():
    global im

    # Blur a bit so the lines are more contiguous
    source = gaussian_filter(originalsource, sigma = sourceblur)

    resultshape = (verticalsize + verticalpad * 2, horizontalsize + horizontalpad * 2)
    imshape = (resultshape[0] * SUPERSAMPLE, resultshape[1] * SUPERSAMPLE)

    im = np.zeros(imshape)
    minheights = [float("inf")] * imshape[1]

    for i in range(lines):
        height = int(verticalpad + (verticalsize - thickness) / (lines - 1) * (lines - i - 1))
        lastoffset = None

        curvecoords = ([],[])

        source_y = int(source.shape[0] * (lines - i - 1) / lines)

        for x in range(horizontalsize):
            source_x = int(source.shape[1] * x / horizontalsize)

            # Get color at source image (aka intensity)
            intensity = source[source_y][source_x]
            offset = -(noise * getnoise() * intensity + offsetscale * intensity)
            if lastoffset is None:
                lastoffset = offset

            # Get starting and ending points of line segment
            oldlineheight = (height + lastoffset) * SUPERSAMPLE
            lineheight = (height + offset) * SUPERSAMPLE
            oldlineheight = clip(oldlineheight, 0, im.shape[0] - 1)
            lineheight = clip(lineheight, 0, im.shape[0] - 1)
            linestart = int((x + horizontalpad) * SUPERSAMPLE)
            lineend = int((x + horizontalpad + 1) * SUPERSAMPLE)

            # Get points of curve
            rr, cc = line(oldlineheight, linestart, lineheight, lineend)
            curvecoords[0].extend(rr)
            curvecoords[1].extend(cc)

            lastoffset = offset

        # Ensure that we don't draw below the previous peaks
        newminheights = list(minheights)
        for i in range(len(curvecoords[0]))[::-1]:
            col = curvecoords[1][i]
            curheight = curvecoords[0][i]
            if curheight > minheights[col]:
                del curvecoords[0][i]
                del curvecoords[1][i]
            else:
                newminheights[col] = min(newminheights[col], curheight)
        minheights = newminheights
        im[curvecoords[0], curvecoords[1]] = 1

        # Seed noise by random amount
        for _ in range(random.randrange(horizontalsize)):
            getnoise()

    # Add thickness to the lines
    im = morphology.dilation(im, morphology.disk(radius=thickness * SUPERSAMPLE))

    # Shrink to result size (in case of supersampling)
    im = resize(im, resultshape, anti_aliasing=True)

    # Blurring can help the final look
    im = gaussian_filter(im, sigma = 1)

    global im_ax
    im_ax = ax.imshow(im, cmap='gray', interpolation='bicubic', vmin=0, vmax=1.0)

draw()

###################
# Boring UI stuff #
###################

uiheight = 0
def addheight(val):
    global uiheight
    uiheight += val
    return uiheight

savefilename = ".".join(sys.argv[1].split(".")[:-1])
savefilename += "_out.png"
savefilenamebox = TextBox(fig.add_axes([0.25, addheight(0.01), 0.35, 0.03]), "Filename", initial=savefilename)
savebutton = Button(fig.add_axes([0.65, addheight(0.0), 0.1, 0.03]), 'Save')
thicknessslider = Slider(fig.add_axes([0.25, addheight(0.04), 0.5, 0.03]), 'Thickness', 1, 10, valinit=thickness, valfmt='%d')
linesslider = Slider(fig.add_axes([0.25, addheight(0.04), 0.5, 0.03]), 'Lines', 2, 100, valinit=lines, valfmt='%d')
noiseslider = Slider(fig.add_axes([0.25, addheight(0.04), 0.5, 0.03]), 'Noise', 0, 10, valinit=np.sqrt(noise), valfmt='%f')
offsetscaleslider = Slider(fig.add_axes([0.25, addheight(0.04), 0.5, 0.03]), 'Offset', 0, 10, valinit=np.sqrt(offsetscale), valfmt='%f')
hsizeslider = Slider(fig.add_axes([0.25, addheight(0.04), 0.5, 0.03]), 'Horizontal Size', 0, 2000, valinit=horizontalsize, valfmt='%d')
vsizeslider = Slider(fig.add_axes([0.25, addheight(0.04), 0.5, 0.03]), 'Vertical Size', 0, 2000, valinit=verticalsize, valfmt='%d')
hpadslider = Slider(fig.add_axes([0.25, addheight(0.04), 0.5, 0.03]), 'Horizontal Padding', 0, 1000, valinit=horizontalpad, valfmt='%d')
vpadslider = Slider(fig.add_axes([0.25, addheight(0.04), 0.5, 0.03]), 'Vertical Padding', 0, 1000, valinit=verticalpad, valfmt='%d')
sourceblurslider = Slider(fig.add_axes([0.25, addheight(0.04), 0.5, 0.03]), 'Source Blur', 0, 100, valinit=sourceblur, valfmt='%d')
randomfrequencyslider = Slider(fig.add_axes([0.25, addheight(0.04), 0.5, 0.03]), 'Random Frequency', 0, 1, valinit=randomfrequency, valfmt='%f')

def updatesavefilename(newname):
    global savefilename
    savefilename = newname

def save(event):
    plt.imsave(savefilename, im, cmap='gray', vmin=0, vmax=1.0)

def updatethickness(val):
    global thickness
    thickness = int(val)
    draw()

def updatelines(val):
    global lines
    lines = int(val)
    draw()

def updatenoise(val):
    global noise
    noise = val ** 2
    draw()

def updateoffset(val):
    global offsetscale
    offsetscale = val ** 2
    draw()

def updatehsize(val):
    global horizontalsize
    horizontalsize = int(val)
    draw()

def updatevsize(val):
    global verticalsize
    verticalsize = int(val)
    draw()

def updatehpad(val):
    global horizontalpad
    horizontalpad = int(val)
    draw()

def updatevpad(val):
    global verticalpad
    verticalpad = int(val)
    draw()

def updatesourceblur(val):
    global sourceblur
    sourceblur = int(val)
    draw()

def updaterandomfrequency(val):
    global randomfrequency
    randomfrequency = val
    draw()

savefilenamebox.on_text_change(updatesavefilename)
savebutton.on_clicked(save)
thicknessslider.on_changed(updatethickness)
linesslider.on_changed(updatelines)
noiseslider.on_changed(updatenoise)
offsetscaleslider.on_changed(updateoffset)
hsizeslider.on_changed(updatehsize)
vsizeslider.on_changed(updatevsize)
hpadslider.on_changed(updatehpad)
vpadslider.on_changed(updatevpad)
sourceblurslider.on_changed(updatesourceblur)
randomfrequencyslider.on_changed(updaterandomfrequency)

plt.show()
