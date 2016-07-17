# Seed, Expand, Constrain: Three Principles for Weakly-Supervised Image Segmentation
Created by Alexander Kolesnikov and Christoph Lampert at IST Austria.

### Disclaimer: the code was tested on **Ubuntu 14.04 LTS**.

## Dependencies

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

## Adding new loss layers to caffe

In order to add new loss layers to caffe install the `pylayers` package, provided in this repository:
```bash
      $ pip install pylayers/
```

## How to train SEC model

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

## How to deploy the SEC model

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
