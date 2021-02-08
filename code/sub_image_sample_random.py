"""

Saves image patches and their labels, in directories

Randomly selects positions until quota filled (meant for non-comprehensive sampling)

"""


import os
import matplotlib.pyplot as plt
import geopandas as gpd
import cv2
import rasterio
from shapely.geometry import Point

from os import listdir
from os.path import isfile, join

import numpy as np

from local_config import *
from global_config import *

# timeout for some images might not have enough of that class
MAX_TRIES = 1200


# one by one read in Shapefiles
shp_files = [f for f in listdir(SHAPEFILE_DIR) if isfile(join(SHAPEFILE_DIR, f)) and ".shp" in f]
tiff_files = [f for f in listdir(TIFF_DIR) if isfile(join(TIFF_DIR, f)) and ".tif" in f]

print("Shapefiles found:       ", len(shp_files))
print("GeoTIFF files found:    ", len(tiff_files))

# per image
samples_per_label = 100

# function to determine whether a spatial coordinate is in any of the listed shapes
def get_label(shape_data, spatial_coords):

	spa_x, spa_y = spatial_coords

	for i, poly in enumerate(shape_data['geometry']):
		if poly.contains(Point(spa_x, spa_y)):
			return shape_data['poly_type'][i]

	return None


# each row corresponds to a sub_image - image ix, (x, y) for top left corner, label
meta = []

idx = 7520  # unique sub-sample image idx


def is_quota_met(count_dict):
	for val in count_dict.values():
		if val < samples_per_label:
			return False
	return True



# Local testing constraint
# shp_files = shp_files[:2]



for shpfile in shp_files:
	tries = 0
	shp_id = shpfile.split("_")[-1][:-4].upper()
	shape_data = gpd.read_file(SHAPEFILE_DIR + shpfile)

	# reset the sample count dict per image
	sample_count = {
		"L": 0,
		"W": 0,
		"I": 0,
	}

	# read in associated GeoTIFF file
	tiff_file = [g for g in tiff_files if shp_id in g]
	print(tiff_file[0])

	if len(tiff_file):
		src = rasterio.open(TIFF_DIR + tiff_file[0])

		# get numpy version for sampling sub_image
		np_tiff = np.rollaxis(src.read(), 0, 3)

		while not is_quota_met(sample_count) and tries < MAX_TRIES:

			tries += 1

			# select random x, y in src.width, src.height
			x_pos = np.random.randint(0, src.width)
			y_pos = np.random.randint(0, src.height)

			spatial_coords = src.transform * (x_pos + int(K/2), y_pos + int(K/2))

			# is this position in a shape?
			label = get_label(shape_data, spatial_coords)
			if label and sample_count[label] < samples_per_label:
				sub_im = np_tiff[x_pos:(x_pos + K), y_pos:(y_pos + K)]

				if sub_im.shape == (K, K, 3):
					cv2.imwrite(SAMPLING_DIR + str(idx) + ".png", sub_im)
					meta.append([idx, x_pos, y_pos, label])
					idx += 1

					sample_count[label] += 1



	print("Number of tries:", tries)


print("")
print("Final idx:              ", idx)
print("Meta length:            ", len(meta))
np.save(META_DIR + "meta3.npy", np.array(meta))
