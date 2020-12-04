
"""

Global configuration variables.


"""

from local_config import *



####################################################################
# 
# 			Parameters
# 
####################################################################


# Sub-image sampling size K x K
K = 50


LABELS = {
	"L": 0,
	"W": 1,
	"I": 2,
}

TRAIN_SIZE = 0.8


####################################################################
# 
# 			Directories
# 
####################################################################

CKPT_DIR = ROOT_DIR + "checkpoints/"

SAMPLING_DIR = ROOT_DIR + "samples/"

META_DIR = ROOT_DIR + "sample_meta/"

SHAPEFILE_DIR = ROOT_DIR + "sea_ice/"

TIFF_DIR = ROOT_DIR + "tifs/"