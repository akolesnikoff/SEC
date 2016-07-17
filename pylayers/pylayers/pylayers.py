import caffe

import numpy as np
from scipy.ndimage import zoom

import theano
import theano.tensor as T

import cPickle

from krahenbuhl2013 import CRF

min_prob = 0.0001


class SoftmaxLayer(caffe.Layer):

    def setup(self, bottom, top):

        if len(bottom) != 1:
            raise Exception("Need two inputs to compute distance.")

        preds = T.ftensor4()
        top_diff = T.ftensor4()

        preds_max = T.addbroadcast(T.max(preds, axis=1, keepdims=True), 1)
        preds_exp = np.exp(preds - preds_max)
        probs = preds_exp / T.addbroadcast(T.sum(preds_exp, axis=1, keepdims=True), 1) + min_prob
        probs = probs / T.sum(probs, axis=1, keepdims=True)

        probs_sum = T.sum(probs * top_diff)

        self.forward_theano = theano.function([preds], probs)
        self.backward_theano = theano.function([preds, top_diff], T.grad(probs_sum, preds))

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.forward_theano(bottom[0].data[...])

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], top[0].diff[...])
        bottom[0].diff[...] = grad


class CRFLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):

        unary = np.transpose(np.array(bottom[0].data[...]), [0, 2, 3, 1])

        mean_pixel = np.array([104.0, 117.0, 123.0])
        im = bottom[1].data[...]
        im = zoom(im, (1.0, 1.0, 41.0 / im.shape[2], 41.0 / im.shape[3]), order=1)
        im = im + mean_pixel[None, :, None, None]
        im = np.transpose(np.round(im), [0, 2, 3, 1])

        N = unary.shape[0]

        self.result = np.zeros(unary.shape)

        for i in range(N):
            self.result[i] = CRF(im[i], unary[i], scale_factor=12.0)

        self.result = np.transpose(self.result, [0, 3, 1, 2])
        self.result[self.result < min_prob] = min_prob
        self.result = self.result / np.sum(self.result, axis=1, keepdims=True)

        top[0].data[...] = np.log(self.result)

    def backward(self, top, prop_down, bottom):
        grad = (1 - self.result) * top[0].diff[...]
        bottom[0].diff[...] = grad


class SeedLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

        probs = T.ftensor4()
        labels = T.ftensor4()

        count = T.sum(labels, axis=(1, 2, 3), keepdims=True)
        loss_balanced = -T.mean(T.sum(labels * T.log(probs), axis=(1, 2, 3), keepdims=True) / count)

        self.forward_theano = theano.function([probs, labels], loss_balanced)
        self.backward_theano = theano.function([probs, labels], T.grad(loss_balanced, probs))

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        top[0].data[...] = self.forward_theano(bottom[0].data[...], bottom[1].data[...])

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...])
        bottom[0].diff[...] = grad


class ConstrainLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

        probs = T.ftensor4()
        probs_smooth_log = T.ftensor4()

        probs_smooth = T.exp(probs_smooth_log)

        loss = T.mean(T.sum(probs_smooth * T.log(probs_smooth / probs), axis=1))

        self.forward_theano = theano.function([probs, probs_smooth_log], loss)
        self.backward_theano = theano.function([probs, probs_smooth_log], T.grad(loss, [probs, probs_smooth_log]))

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        top[0].data[...] = self.forward_theano(bottom[0].data[...], bottom[1].data[...])

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...])[0]
        bottom[0].diff[...] = grad
        grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...])[1]
        bottom[1].diff[...] = grad


class ExpandLossLayer(caffe.Layer):

    def setup(self, bottom, top):

        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

        probs_tmp = T.ftensor4()
        stat_inp = T.ftensor4()

        stat = stat_inp[:, :, :, 1:]

        probs_bg = probs_tmp[:, 0, :, :]
        probs = probs_tmp[:, 1:, :, :]

        probs_max = T.max(probs, axis=3).max(axis=2)

        q_fg = 0.996
        probs_sort = T.sort(probs.reshape((-1, 20, 41 * 41)), axis=2)
        weights = np.array([q_fg ** i for i in range(41 * 41 - 1, -1, -1)])[None, None, :]
        Z_fg = np.sum(weights)
        weights = T.addbroadcast(theano.shared(weights), 0, 1)
        probs_mean = T.sum((probs_sort * weights) / Z_fg, axis=2)

        q_bg = 0.999
        probs_bg_sort = T.sort(probs_bg.reshape((-1, 41 * 41)), axis=1)
        weights_bg = np.array([q_bg ** i for i in range(41 * 41 - 1, -1, -1)])[None, :]
        Z_bg = np.sum(weights_bg)
        weights_bg = T.addbroadcast(theano.shared(weights_bg), 0)
        probs_bg_mean = T.sum((probs_bg_sort * weights_bg) / Z_bg, axis=1)

        stat_2d = stat[:, 0, 0, :] > 0.5

        loss_1 = -T.mean(T.sum((stat_2d * T.log(probs_mean) / T.sum(stat_2d, axis=1, keepdims=True)), axis=1))
        loss_2 = -T.mean(T.sum(((1 - stat_2d) * T.log(1 - probs_max) / T.sum(1 - stat_2d, axis=1, keepdims=True)), axis=1))
        loss_3 = -T.mean(T.log(probs_bg_mean))

        loss = loss_1 + loss_2 + loss_3

        self.forward_theano = theano.function([probs_tmp, stat_inp], loss)
        self.backward_theano = theano.function([probs_tmp, stat_inp], T.grad(loss, probs_tmp))

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        top[0].data[...] = self.forward_theano(bottom[0].data[...], bottom[1].data[...])

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...])
        bottom[0].diff[...] = grad


class AnnotationLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("The layer needs two inputs!")

        self.data_file = cPickle.load(open('localization_cues/localization_cues.pickle'))

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0], 1, 1, 21)
        top[1].reshape(bottom[0].data.shape[0], 21, 41, 41)

    def forward(self, bottom, top):

        top[0].data[...] = 0.0
        top[1].data[...] = 0.0

        for i, image_id in enumerate(bottom[0].data[...]):

            labels_i = self.data_file['%i_labels' % image_id]
            top[0].data[i, 0, 0, labels_i] = 1.0

            cues_i = self.data_file['%i_cues' % image_id]
            top[1].data[i, cues_i[0], cues_i[1], cues_i[2]] = 1.0
