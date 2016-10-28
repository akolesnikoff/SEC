# Seed, Expand, Constrain: Three Principles for Weakly-Supervised Image Segmentation
Created by Alexander Kolesnikov and Christoph Lampert at IST Austria.

## Introduction

![Overview of SEC](https://cloud.githubusercontent.com/assets/460828/19805112/cb7e053c-9d12-11e6-912f-24e2dbdc6699.png)

We propose a new composite loss function for training convolutional neural
networks for the task of weakly-supervised image segmentation. Our approach
relies on the following three insights:

1. Image classification neural networks can
be used to generate reliable object localization cues (seeds), but fail to
predict the exact spatial extent of the objects. We incorporate this aspect
by using a seeding loss that encourages a segmentation network to match
localization cues but that is agnostic about the rest of the image.

2. To train a segmentation network from per-image annotation, a global pooling
layer can be used that aggregates segmentation masks into image-level label
scores. The choice of this layer has large impact on the quality of segmenta-
tions. For example, max-pooling tends to underestimate the size of objects
while average-pooling tends to overestimate it. We propose a global
weighted rank pooling that is leveraged by expansion loss to expand
the object seeds to regions of a reasonable size. It generalizes max-pooling
and average pooling and outperforms them in our empirical study.

3. Networks trained from image-level labels rarely capture the precise bound-
aries of objects in an image. Postprocessing by fully-connected conditional
random fields (CRF) at test time is often insucient to overcome this ect,
because once the networks have been trained they tend to be confident even
about misclassified regions. We propose a new constrain-to-boundary
loss that alleviates the problem of imprecise boundaries already at train-
ing time. It strives to constrain predicted segmentation masks to respect
low-level image information, in particular object boundaries.

## Citing this repository

If you find this code useful in your research, please consider citing us:

        @inproceedings{kolesnikov2016seed,
          title={Seed, Expand and Constrain: Three Principles for Weakly-Supervised Image Segmentation},
          author={Kolesnikov, Alexander and Lampert, Christoph H.},
          booktitle={European Conference on Computer Vision ({ECCV})},
          year={2016},
          organization={Springer}
        }

## Installing dependencies

* Python packages:
```bash
      $ pip install -r python-dependencies.txt
```
* **caffe**: installation instructions are available at `http://caffe.berkeleyvision.org/installation.html`.
   Note, you need to compile **caffe** with python wrapper and support for python layers.

* Fully connected CRF wrapper (requires the **Eigen3** package).
```bash
      $ pip install CRF/
```

## Adding new loss layers required by SEC 

In order to add new loss layers to caffe install the `pylayers` package, provided in this repository:
```bash
      $ pip install pylayers/
```

## Training the SEC model

* Go into the training directory: 

```bash
      $ cd training
```

* Download the initial model (~80 MB) pretrained on Imagenet:

```bash
      $ wget http://ccvl.stat.ucla.edu/ccvl/init_models/vgg16_20M.caffemodel
```

* Decompress localization cues:

```bash
      $ gzip -kd localization_cues/localization_cues.pickle.gz
```

* Set *root_folder* parameter in **train.prototxt** to the directory with **PASCAL VOC 2012** images 

* Run caffe:

```bash
      $ caffe train --solver solver.prototxt --weights vgg16_20M.caffemodel --gpu <gpu_id>
```
   The trained model will be created in `training/models`

## Deploying the SEC model

* Go into the deploy directory: 

```bash
      $ cd deploy
```

* Download pretrained SEC model (~80 MB):

```bash
      $ wget http://pub.ist.ac.at/~akolesnikov/SEC-data/SEC.caffemodel
```

* Run the model on any image (*smooth* options switches on CRF postprocessing):

```bash
      $ python demo.py --model SEC.caffemodel --image <PATH TO IMAGE> --smooth
```

* Some segmentation examples:

![example](https://cloud.githubusercontent.com/assets/460828/19045485/57c72416-8999-11e6-8089-27b00c5c4712.png)
