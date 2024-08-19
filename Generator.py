from PIL import Image, ImageDraw
import numpy as np
import scipy
import scipy.sparse
import scipy.ndimage

# Chessboard Image generator


def chess_generator(imsize, contrast):
    im = Image.new("RGB", (imsize, imsize), "black")
    draw = ImageDraw.Draw(im)
    imsize = imsize/4
    for i in range(0,4):
        by = i * imsize
        ey = (i + 1) * imsize
        for j in range(0,4):
            bx = (2 * j + (i % 2)) * imsize
            ex = (2 * j + 1 + (i % 2)) * imsize
            draw.rectangle(xy=(bx, by, ex, ey),
                            fill=(contrast, contrast, contrast))
            im.paste(im)
    im.save("test.png", "PNG")
    return "test.png"

# Simple Image Generator

def image_generator(shape, imsizex, imsizey, fill, width, outline=None, backcolor=None, ):
    im = Image.new("RGB", (imsizex, imsizey), backcolor)
    draw = ImageDraw.Draw(im)
    x = imsizex / 2
    rx = imsizey / 4
    y = imsizex / 2
    ry = imsizey / 4
    if shape == "circle":
        draw.ellipse(xy=(rx, ry, x + rx, y + ry),
                     fill=fill,
                     outline=outline,
                     width=width)
        im.save("test.png", "PNG")
        return im

    if shape == "line":
        draw.line(xy=(0, im.size[1], im.size[0], 0),
                  fill=fill,
                  width=width)
        im.save("linienvergleich.png", "PNG")
        return im

    if shape == "rectangle":
        draw.rectangle(xy=(rx, ry, x + rx, y + ry),
                       outline=outline,
                       fill=fill,
                       width=width)
        im.save("test.png", "PNG")
        return im

    else:
        print("shape = {rectangle, circle, line}, size = R, backcolor = {color} benutzen ")
        exit()


    #Für Bilder plotten
    """fig, ax = plt.subplots(1, iteration, figsize=(10, 10))

    for i in range(len(ax)):
        ax[i].axis('off')"""
    # Generating simple Images
    """#shape : "rectangle", "line", "cirlce"
    shape = "rectangle"
    #xsize, ysize of image
    xsize = 500
    ysize = 500
    #Backgroundcolor: "{color}"
    bgcolor = "blue"
    #fillcolor: ( {0-255} (red) , {0-255} (green), {0-255} (blue))
    fillcolor = (255, 0, 0)
    #outline: ( {0-255} (red) , {0-255} (green), {0-255} (blue))
    outlinecolor = (255, 255, 255)
    #width: width
    width = 3
    im = image_generator(shape, xsize, ysize, bgcolor, fillcolor, outlinecolor, width)
    im.save("test.png", "PNG")
    #lwidth: line width
    lwidth = 10
    line = image_generator("line", 100, 100, "black", fill=(255, 255, 255),outline=(255,255,255), width = lwidth)
    line.save("linienvergleich.png", "PNG")
    """


# http://news.zahlt.info/en/optimization/image-deconvolution-using-tikhonov-regularization/
# TODO: Noch besser verstehen und warum muss man gerade und ungerade unterscheiden????

# Sparse matrix for Convolutionkernel convoluted with Image

def make_kernel_2D(convolutionkernel, dims):
    """
        PSF is the 2D kernel
        dims are is the side size of the image in order (r,c)
    """
    d = len(convolutionkernel) ## assmuming square convolutionkernel (but not necessarily square image)
    N = dims[0]*dims[1]
    ## pre-fill a 2D matrix for the diagonals
    diags = np.zeros((d*d, N))
    offsets = np.zeros(d*d)
    heads = np.zeros(d*d) ## for this a list is OK
    i = 0
    if d % 2 == 1:
        for y in range(len(convolutionkernel)):
            for x in range(len(convolutionkernel[y])):
                diags[i,:] += convolutionkernel[y, x]
                heads[i] = convolutionkernel[y, x]
                xdist = d/2 - x
                ydist = d/2 - y ## y direction pointing down
                offsets[i] = (ydist*dims[1] + xdist - dims[0]/2 - 0.5)
                i+=1
    else:
        for y in range(len(convolutionkernel)):
            for x in range(len(convolutionkernel[y])):
                diags[i,:] += convolutionkernel[y, x]
                heads[i] = convolutionkernel[y, x]
                xdist = d/2 - x
                ydist = d/2 - y ## y direction pointing down
                offsets[i] = ydist*dims[1] + xdist
                i+=1
    ## create linear operator
    H = scipy.sparse.dia_matrix((diags,offsets),shape=(N,N))
    return H

# Gaussian blur matrix to convolve with
def make_blur_matrix(img, kernel_size=5):
    n = kernel_size
    k2 = np.zeros(shape=(n, n))
    k2[int(n / 2), int(n / 2)] = 1
    sigma = kernel_size / 5.0  ## 2.5 sigma
    testk = scipy.ndimage.gaussian_filter(k2, sigma)  ## already normalized
    blurmat = make_kernel_2D(testk, img.shape)

    return blurmat



