# extremeearth-polar


## Overview

Sentinel-1 extra-wideswath SAR images provide 20mx100m spatial resolution HH and HV bands, with a swath size of 400km. The short repeat period of these images gives us the opportunity to observe how the coastline of Greenland is changing with a greater temporal resolution than ever before. These images can tell us where the coastline is, where the calving fronts of glaciers are, how much melt/freezing has there been, how many ice bergs there are, and how all of these things are changing. In order to generate a timeseries based on one of these observations, the images must be segmented into areas of land, sea and ice. Traditionally this was done manually; a long and laborious task. However, the rise of neural networks as a tool for semantic image segmentation has given scientists a way of achieving this in a fraction of the time, with similar accuracy. 

The goal of this challenge is to build and train a model which is able to automatically segment Sentinel-1 satellite images from a given region into the three classes of sea, ice, and land. For this task, a training dataset of annotated Sentinel-1 images is provided. 

The following is a naive sample solution to the challenge, taking the approach of (1) sampling sub-images, or patches, from the large SAR images, and then labelling them using the ground truth segmentations; then (2) training a convolutional neural network to perform classification on a large training set of those patches; and (3) creating the final segmentation prediction by re-piecing together the patches and their predictions on test images.

## Dataset

The full dataset can be downloaded from the following two links: 

- Shapefile data with all ground truth segmentation data and preview PNG images, and documentation: https://zenodo.org/record/3695276#.X37M7JNKgp9  (30 MB) 

- GeoTIFF Sentinel images: ftp://ftp.met.no/users/nicholsh/EE_S1_Training/  (2.11 GB) 
 
There is also a [GitHub Repo polar-patch](https://github.com/ysbecca/polar-patch) containing a set of approximately 10,000 image patches and corresponding labels, to be used for development or testing.

## Contents

1. code
2. docs -- PDF documentation for dataset
3. icebergs -- Shapefiles defining icebergs (not used in this challenge)
4. pngs -- PNG downsized renderings of the SAR images with segmentations overlaid
5. sea_ice -- Shapefile images (.dbf, .prj, .shp, .shx) defining ground truth segmentations


## Running

Ensure all data is in locations specified by ```local_config.py``` and define ```ROOT_DIR``` in ```local_config.py```. Choose between ```sub_image_sample.py``` and ```sub_image_sample_random.py```, which perform either consecutive sampling from every labelled location (producing a very large dataset), or random sub-sampling with a max number of samples per label and a timeout on sampling attempts.

Train CNN model on dataset specifying arguments.

```bash
python train.py --run_number 0 --epochs 3 --save_interval 1 --batch_size 28
```





