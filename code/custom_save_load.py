"""

Unified place to put save and load model code so that model nomenclature is the same everywhere.

"""
import torch
import numpy as np


def get_model_name(run_number, epoch):

	return "model_e" + str(epoch) + "_r" + str(run_number)

def load_checkpoint(model_name, path):
	"""
	Args:

		model_name        as formatted by __get_model_name()
		path              directory to load checkpoint from

	Returns:

		checkpoint

	"""

	text = "LOADED"
	print(F"{text:>20} {model_name}.ckpt")
	return torch.load(path + model_name + ".ckpt")

def save_model(model, model_name, path):
	"""
	Args:

		model_name        as formatted by __get_model_name()
		path              directory to load checkpoint from


	Returns:

		model_name

	"""
	torch.save(model.state_dict(), path + model_name + ".ckpt")
	text = "SAVED"
	print(F"{text:>20} {model_name}.ckpt")
	return model_name
