"""

Train a model on Polar patch data

python train.py --run_number 0 --epochs 2 --save_interval 1

"""




import sys
import os

import argparse
import torch
from torch import optim, nn
from torchvision import transforms


from polarpatch import PolarPatch
from alexnet import AlexNet

from local_config import *
from global_config import *

from custom_save_load import *

# Set up training arguments and parse
parser = argparse.ArgumentParser(description='Training network')


parser.add_argument(
    '--run_number', type=int,
    help='Training run number')
parser.add_argument(
    '--epochs', type=int, default=14, # 1e7
    help='Number of epochs to train for')
parser.add_argument(
    '--batch_size', type=int, default=128,
    help='Training batch size')
parser.add_argument(
	'--save_interval', type=int, default=10,
    help='How often to checkpoint - epochs')


# parser.add_argument(
#     '--test_only', type=bool, default=False,
#     help='Flag to test model only')


args = parser.parse_args()


# Device configuration - defaults to CPU unless GPU is available on device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("=======================================")
print("         TRAINING PARAMS               ")
print("=======================================")
for arg in vars(args):
    print(F"{arg:>20} {getattr(args, arg)}")



####################################################################
# 
# 			Load the dataset
# 
####################################################################


data_transform = transforms.Compose([
	transforms.ToTensor(),
])

train_set = PolarPatch(
	split='train',
	transform=data_transform
)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2
)


####################################################################
# 
# 			Load the model
# 
####################################################################


model = AlexNet()
criterion = nn.CrossEntropyLoss()

# Stochastic gradient descent
optimizer = optim.SGD(
	model.parameters(),
	lr=0.01,
	weight_decay=0.0005,
	momentum=0.9,
)


####################################################################
# 
# 			Train the model
# 
####################################################################


for epoch in range(args.epochs):
	for i, data in enumerate(train_loader, 0):
		images, labels, _ = data
		images, labels = images.to(device), labels.to(device)

		optimizer.zero_grad()

		outputs = model(images)
		loss = criterion(outputs, labels)

		loss.backward()
		optimizer.step()

		running_loss = loss.item()
		running_error = (outputs.max(dim=1)[1] != labels).sum().item()

		if i % 100 == 99:    # Print every 100 mini-batches
			print('Epoch / Batch [%d / %d] - Loss: %.3f - Error: %.3f' %
				(epoch + 1, i + 1, running_loss / 100, running_error / 100))


	if epoch % args.save_interval == (args.save_interval - 1):
		model_name = get_model_name(args.run_number, epoch)
		torch.save(model.state_dict(), CKPT_DIR + model_name + ".ckpt")
		text = "SAVED"
		print(F"{text:>20} {model_name}.ckpt")








