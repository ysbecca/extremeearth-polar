"""

Saves image patches and their labels, in directories

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


# One by one read in Shapefiles

shp_files = [f for f in listdir(SHAPEFILE_DIR) if isfile(join(SHAPEFILE_DIR, f)) and ".shp" in f]
tiff_files = [f for f in listdir(TIFF_DIR) if isfile(join(TIFF_DIR, f)) and ".tif" in f]

print("Shapefiles found:       ", len(shp_files))
print("GeoTIFF files found:    ", len(tiff_files))



# Function to determine whether a spatial coordinate is in any of the listed shapes
def get_label(shape_data, spatial_coords):

	spa_x, spa_y = spatial_coords

	for i, poly in enumerate(shape_data['geometry']):
		if poly.contains(Point(spa_x, spa_y)):
			return shape_data['poly_type'][i]

	return None



# each row corresponds to a sub_image - image ix, (x, y) for top left corner, label
meta = []

# Local limitation
shp_files = [shp_files[0]]

for shpfile in shp_files:

	shp_id = shpfile.split("_")[-1][:-4].upper()
	shape_data = gpd.read_file(SHAPEFILE_DIR + shpfile)

	# Read in associated GeoTIFF file
	tiff_file = [g for g in tiff_files if shp_id in g]
	print(tiff_file[0])

	if len(tiff_file):
		src = rasterio.open(TIFF_DIR + tiff_file[0])

		# Get numpy version for sampling sub_image
		np_tiff = np.rollaxis(src.read(), 0, 3)

		# Travel along the src dimensions
		x_pos = 0
		idx = 0  # unique sub-sample image idx
		while x_pos < src.width:

			y_pos = 0
			while y_pos < src.height:

				spatial_coords = src.transform * (x_pos + int(K/2), y_pos + int(K/2))

				# is this position in a shape?
				label = get_label(shape_data, spatial_coords)
				if label:
					sub_im = np_tiff[x_pos:(x_pos + K), y_pos:(y_pos + K)]

					if sub_im.shape == (K, K, 3):
						cv2.imwrite(SAMPLING_DIR + str(idx) + ".png", sub_im)
						meta.append([idx, (x_pos, y_pos), label])
						idx += 1

				y_pos += K
			x_pos += K


print("")
print("Final idx:              ", idx)
print("Meta length:            ", len(meta))
np.save(META_DIR + "meta.npy", np.array(meta))
