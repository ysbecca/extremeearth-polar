"""

Custom sub-image patch dataset for Polar dataset



"""


from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
from PIL import Image


from local_config import *
from global_config import *

class PolarPatch(Dataset):
    def __init__(self, transform=None, split="train"):
        super(PolarPatch, self).__init__()

        meta = np.load(META_DIR + "meta.npy", allow_pickle=True)


        s = int(TRAIN_SIZE * len(meta))
        if split == "train":
            meta = meta[:s]
        else:
            meta = meta[s:]
                   

        self.images = range(len(meta))
        self.coords = [row[1] for row in meta]

        # Targets in integer form
        self.targets = [LABELS[row[2]] for row in meta]
        self.transform = transform


    def __len__(self):
        return len(self.targets)


    def __getitem__(self, index):

        x = Image.open(SAMPLING_DIR + str(self.images[index]) + ".png")
        y = self.targets[index]
        coord = self.coords[index]

        if self.transform:
        	x = self.transform(x)

        return x, y, coord








