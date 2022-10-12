import numpy as np
#import cunumeric as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import argparse

##
## The 'smooth' funciton implements Gaussian smoothing, which is a convolution
## between the image and the following 5x5 filter:
##
##        |  2  4  5  4  2 |
##   1    |  4  9 12  9  4 |
##  --- * |  5 12 15 12  5 |
##  159   |  4  9 12  9  4 |
##        |  2  4  5  4  2 |
##
## Note that the upper left corner of the filter is applied to the
## pixel that is off from the center by (-2, -2).  NumPy does not seem to
## have a 2D convolution primitive, so the implementation below uses a sequence
## matrix-matrix additions, each addition handling one element of the convolution for
## all elements of the image.
##
def smooth(image):
    x,y = image.shape
    x -= 2
    y -= 2
    smooth = 15.0 * image
    for polarity in [1, -1]:
        smooth[2:-2,2:-2] += 12.0 * image[2 + 1 * polarity:x + 1 * polarity, 2 + 0 * polarity:y + 0 * polarity] + \
                              9.0 * image[2 + 1 * polarity:x + 1 * polarity, 2 + 1 * polarity:y + 1 * polarity] + \
                              4.0 * image[2 + 1 * polarity:x + 1 * polarity, 2 + 2 * polarity:y + 2 * polarity] + \
                              5.0 * image[2 + 2 * polarity:x + 2 * polarity, 2 + 0 * polarity:y + 0 * polarity] + \
                              4.0 * image[2 + 2 * polarity:x + 2 * polarity, 2 + 1 * polarity:y + 1 * polarity] + \
                              2.0 * image[2 + 2 * polarity:x + 2 * polarity, 2 + 2 * polarity:y + 2 * polarity]

        smooth[2:-2,2:-2] += 12.0 * image[2 + 0 * polarity:x + 0 * polarity, 2 + 1 * -polarity:y + 1 * -polarity] + \
                              9.0 * image[2 + 1 * polarity:x + 1 * polarity, 2 + 1 * -polarity:y + 1 * -polarity] + \
                              4.0 * image[2 + 2 * polarity:x + 2 * polarity, 2 + 1 * -polarity:y + 1 * -polarity] + \
                              5.0 * image[2 + 0 * polarity:x + 0 * polarity, 2 + 2 * -polarity:y + 2 * -polarity] + \
                              4.0 * image[2 + 1 * polarity:x + 1 * polarity, 2 + 2 * -polarity:y + 2 * -polarity] + \
                              2.0 * image[2 + 2 * polarity:x + 2 * polarity, 2 + 2 * -polarity:y + 2 * -polarity]                           
    return smooth / 159.0                         
                         
#    smooth_filter = np.array([[2,4,5,4,2],
#                              [4,9,12,9,4],
#                              [5,12,15,12,5],
#                              [4,9,12,9,4],
#                              [2,4,5,4,2]])
#    return apply_filter(image, smooth_filter)


##
## 'sobelX' finds the x component of the gradient vector at each pixel.
## Use the following 3x3 filter to implement this function:
##
##  | -1  0  1 |
##  | -2  0  2 |
##  | -1  0  1 |
##
def sobelX(image):
    x_grad = np.zeros_like(image)
    # TODO: implement this function

    return x_grad
##
## 'sobelY' finds the y component of the gradient vector at each pixel.
## Use the following 3x3 filter to implement this function:
##
##  | -1 -2 -1 |
##  |  0  0  0 |
##  |  1  2  1 |
##
def sobelY(image):
    y_grad = np.zeros_like(image)
    # TODO: implement this function

    return y_grad


##
## The 'suppressNonmax' task filters only the gradients that are local maximum.
## Each gradient is compared with two neighbors along its positive
## and negative direction. The gradient direction is rounded to nearest 45°
## to work on a discrete image. The following diagram will be useful to
## determine which neighbors to pick for the comparison:
##
##           j - 1    j      j + 1    x-axis
##         -------------------------
##         |  45°  |  90°  |  135° |
##  i - 1  |  or   |  or   |  or   |
##         |  225° |  270° |  315° |
##         |------------------------
##         |  0°   |       |  0°   |
##    i    |  or   | center|  or   |
##         |  180° |       |  180° |
##         |------------------------
##         |  135° |  90°  |  45°  |
##  i + 1  |  or   |  or   |  or   |
##         |  315° |  270° |  225° |
##         -------------------------
##  y-axis
##

## suppressNonmax returns a boolean array of the same shape as image. If a pixel
## is the local maximum the corresponding entry in the result is True, otherwise
## it is False
def suppressNonmax(image, gradient_x, gradient_y):
    # your code should set entries in localmax to true that are local gradient maxima
    localmax = np.zeros_like(image).astype(bool)
    # TODO implement this function
    
    return localmax


# Find the places where the gradient is > threshold and the point is the local
# maxima in gradient of the region.
def findEdge(gradient_x, gradient_y, localmax, threshold):
    edge_matrix = np.zeros_like(gradient_x)
    gradient = np.sqrt(gradient_x**2 + gradient_y ** 2)
    mask = np.logical_and(gradient > threshold, localmax)
    edge_matrix[mask] = 255
    return edge_matrix


#
# The main code.
#

#Usage: python3 edge.py [OPTIONS]
#OPTIONS
#  -h            : Print the usage and exit.
#  -i {file}     : Use {file} as input.
#  -o {file}     : Save the final edge to {file}. Will use 'edge.png' by default.
#  -s {file}     : Save the image after Gaussian smoothing to {file}.
#  -t {value}    : Set {value} to the threshold.
#  --no-smooth   : Skip Gaussian smoothing.
#  --no-suppress : Skip non-maximum suppression.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect edges in an image')
    parser.add_argument('-i', type=str, help='name of file to load',
                        required = True)
    parser.add_argument('-o', type=str, help='name of result file',
                        default= 'edges.png')
    parser.add_argument('-s', type=str, help='save image after Gaussian smoothing to {file}')
    parser.add_argument('-t', type = float, default=80,
                        help='The threshold over which gradients will be considered.')
    parser.add_argument('--no_smooth',action="store_true",help="skip Gaussian smoothing")
    parser.add_argument('--no_suppress',action="store_true",help="skip non-maximum suppression" )

    args = parser.parse_args()

    image = Image.open(args.i, 'r')
    w, h = image.size
    data = np.array(image.getdata()).reshape(h, w)
    if args.no_smooth:
        smooth_data = data
    else:
        smooth_data = smooth(data)
        if args.s is not None:
           plt.imsave(args.s,smooth_data,cmap="gray")
    gradient_x = sobelX(smooth_data)
    gradient_y = sobelY(smooth_data)
    if args.no_suppress:
        localmax = np.ones_like(smooth_data)
    else:
        localmax = suppressNonmax(smooth_data, gradient_x, gradient_y)
    edge_data = findEdge(gradient_x, gradient_y, localmax, args.t)
    plt.imsave(args.o, edge_data,cmap="gray")
