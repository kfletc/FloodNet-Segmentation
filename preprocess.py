from dataset import FloodDataset
import config
from torch.utils.data import DataLoader
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import torch
import pickle

# load the image and mask filepaths
train_imagePaths = sorted(list(paths.list_images(config.IMAGE_TRAIN_DATASET_PATH)))
train_maskPaths = sorted(list(paths.list_images(config.MASK_TRAIN_DATASET_PATH)))
test_imagePaths = sorted(list(paths.list_images(config.IMAGE_TEST_DATASET_PATH)))
test_maskPaths = sorted(list(paths.list_images(config.MASK_TEST_DATASET_PATH)))
val_imagePaths = sorted(list(paths.list_images(config.IMAGE_VAL_DATASET_PATH)))
val_maskPaths = sorted(list(paths.list_images(config.MASK_VAL_DATASET_PATH)))

# define transformations
transforms = transforms.Compose([transforms.ToPILImage(),
			       transforms.Resize((config.INPUT_IMAGE_HEIGHT,
						  config.INPUT_IMAGE_WIDTH)),
			       transforms.ToTensor()])

# create the datasets
trainDS = FloodDataset(imagePaths=train_imagePaths, maskPaths=train_maskPaths,
		       transforms=transforms)
testDS = FloodDataset(imagePaths=test_imagePaths, maskPaths=test_maskPaths,
		      transforms=transforms)
valDS = FloodDataset(imagePaths=val_imagePaths, maskPaths=val_maskPaths,
		     transforms=transforms)

with open('PickleDumps/train_pickle.pickle', 'wb') as train_file:
	pickle.dump(trainDS, train_file)
with open('PickleDumps/val_pickle.pickle', 'wb') as val_file:
	pickle.dump(valDS, val_file)
with open('PickleDumps/test_pickle.pickle', 'wb') as test_file:
	pickle.dump(testDS, test_file)
