"""

One-time use to save meta to GitHub repo

"""


import numpy as np
import csv

from local_config import *
from global_config import *


NUM_FILES = 4



meta_files = []
total_count = 0

for i in range(NUM_FILES):
	data = np.load(META_DIR + "meta" + str(i) + ".npy")
	
	total_count += len(data)
	meta_files.append(data)


meta = []

loc = 0
for file in meta_files:
	for row in file:
		meta.append(row)

print(len(meta))
print(meta[:5])


with open(META_DIR + 'meta_all.csv', 'w', newline='') as csvfile:
    writer = csv.writer(
    	csvfile,
    	delimiter=' ',
    	quotechar='|',
    	quoting=csv.QUOTE_MINIMAL
    )

    for row in meta:
	    writer.writerow(row)






