import numpy as np
import pylab

import scipy.ndimage as nd

import png


from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors

import krahenbuhl2013

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--image", dest="image_file", type='string',
                  help="Path to image")
parser.add_option("--model", dest="model", type='string',
                  help="Model weights")
parser.add_option("--output", dest="output", type='string',
                  help="Output png file name", default='')
parser.add_option("--smooth", dest="smooth", action='store_true',
                  help="Apply postprocessing")

options, _ = parser.parse_args()

smooth = options.smooth
image_file = options.image_file
model = options.model
output = options.output

mean_pixel = np.array([104.0, 117.0, 123.0])

palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
           (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
           (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
           (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
           (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
           (0.0, 0.25, 0.5)]
my_cmap = mpl_colors.LinearSegmentedColormap.from_list('Custom cmap', palette, 21)


def write_to_png_file(im, f):
    global palette
    palette_int = map(lambda x: map(lambda xx: int(255 * xx), x), palette)
    w = png.Writer(size=(im.shape[1], im.shape[0]), bitdepth=8, palette=palette_int)
    with open(f, "w") as ff:
        w.write(ff, im)


def preprocess(image, size, mean_pixel=mean_pixel):

    image = np.array(image)

    image = nd.zoom(image.astype('float32'),
                    (size / float(image.shape[0]),
                    size / float(image.shape[1]), 1.0),
                    order=1)

    image = image[:, :, [2, 1, 0]]
    image = image - mean_pixel

    image = image.transpose([2, 0, 1])
    return image


def predict_mask(image_file, smooth=True):

    im = pylab.imread(image_file)

    net.blobs['images'].data[0] = preprocess(im, 321)
    net.forward()

    scores = np.transpose(net.blobs['fc8-SEC'].data[0], [1, 2, 0])
    d1, d2 = float(im.shape[0]), float(im.shape[1])

    scores_exp = np.exp(scores - np.max(scores, axis=2, keepdims=True))
    probs = scores_exp / np.sum(scores_exp, axis=2, keepdims=True)
    probs = nd.zoom(probs, (d1 / probs.shape[0], d2 / probs.shape[1], 1.0), order=1)

    eps = 0.00001
    probs[probs < eps] = eps

    if smooth:
        result = np.argmax(krahenbuhl2013.CRF(im, np.log(probs), scale_factor=1.0), axis=2)
    else:
        result = np.argmax(probs, axis=2)

    return result


if __name__ == "__main__":

    net = caffe.Net('deploy.prototxt', model, caffe.TEST)

    mask = predict_mask(image_file, smooth=smooth)

    if output:
        write_to_png_file(mask, output)
    else:
        fig = plt.figure()

        ax = fig.add_subplot('121')
        ax.imshow(pylab.imread(image_file))

        ax = fig.add_subplot('122')
        ax.matshow(mask, vmin=0, vmax=21, cmap=my_cmap)

        plt.show()
