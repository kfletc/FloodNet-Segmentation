# import the necessary pachages
from dataset import FloodDataset
import config
from model import FloodNet
# loss function
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
import pickle

# load the image and mask filepaths
train_imagePaths = sorted(list(paths.list_images(config.IMAGE_TRAIN_DATASET_PATH)))
train_maskPaths = sorted(list(paths.list_images(config.MASK_TRAIN_DATASET_PATH)))
test_imagePaths = sorted(list(paths.list_images(config.IMAGE_TEST_DATASET_PATH)))
test_maskPaths = sorted(list(paths.list_images(config.MASK_TEST_DATASET_PATH)))
val_imagePaths = sorted(list(paths.list_images(config.IMAGE_VAL_DATASET_PATH)))
val_maskPaths = sorted(list(paths.list_images(config.MASK_VAL_DATASET_PATH)))

# define transformations
transforms_image = transforms.Compose([transforms.ToPILImage(),
			       transforms.Resize((config.INPUT_IMAGE_HEIGHT,
						  config.INPUT_IMAGE_WIDTH)),
			       transforms.ToTensor()])

transforms_mask = transforms.Compose([transforms.ToPILImage(),
			       transforms.Resize((config.INPUT_IMAGE_HEIGHT,
						  config.INPUT_IMAGE_WIDTH)),
			       transforms.PILToTensor()])

# create the datasets
if os.path.exists('PickleDumps/train_pickle'):
	pickle_file = open('PickleDumps/train_pickle', 'rb')
	trainDS = pickle.load(pickle_file)
	pickle_file.close()
else:
	trainDS = FloodDataset(imagePaths=train_imagePaths, maskPaths=train_maskPaths,
				   		transforms=transforms)
if os.path.exists('PickleDumps/test_pickle'):
	pickle_file = open('PickleDumps/test_pickle', 'rb')
	testDS = pickle.load(pickle_file)
	pickle_file.close()
else:
	testDS = FloodDataset(imagePaths=test_imagePaths, maskPaths=test_maskPaths,
		      			transforms=transforms)
if os.path.exists('PickleDumps/val_pickle'):
	pickle_file = open('PickleDumps/val_pickle', 'rb')
	valDS = pickle.load(pickle_file)
	pickle_file.close()
else:
	valDS = FloodDataset(imagePaths=val_imagePaths, maskPaths=val_maskPaths,
		     			transforms=transforms)

print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")
print(f"[INFO] found {len(valDS)} examples in the val set...")


# create the training, test, and val data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
			 batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
			 num_workers=0)
testLoader = DataLoader(testDS, shuffle=False,
			 batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
			 num_workers=0)
valLoader = DataLoader(valDS, shuffle=False,
			 batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
			 num_workers=0)

# initialize UNet model
unet = FloodNet().to(config.DEVICE)

# initialize loss function and optimizer
lossFunc = CrossEntropyLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)

# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE
valSteps = len(valDS) // config.BATCH_SIZE

# initialize a dictionary to store training history
H = {"train_loss": [], "val_loss":[], "train_acc": [], "val_acc": []}

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
	print(f"[INFO] starting epochs {e}")

	# set the model in training mode
	unet.train()

	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0
	totalTrainAccuracy = 0
	totalValAccuracy = 0
	
	# loop over the training set
	for (x, y) in trainLoader:
		# send the input to the device
		(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

		# preform a forward pass and calculate the training loss
		pred = unet(x)
		y_squeeze = torch.squeeze(y)
		loss = lossFunc(pred, y_squeeze)
		# calculate training pixel accuracy
		_, class_preds = torch.max(pred, 1)
		correct_tensor = class_preds.eq(y_squeeze.data.view_as(class_preds))
		total_correct = np.sum(correct_tensor.cpu().numpy())
		totalTrainAccuracy += total_correct / (config.INPUT_IMAGE_WIDTH * config.INPUT_IMAGE_HEIGHT * config.BATCH_SIZE)

		# first, zero out any previously accumulated gradients, then
		# preform backproagation, and then update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()

		# add teh loss to the total training loss so far
		totalTrainLoss += loss


	# switch off autograd
	with torch.no_grad():
		# set model in evaluation mode
		unet.eval()

		# loop over the validation set
		for (x, y) in valLoader:
			# send the input to the device
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

			# make the predictions and calculate the validation loss
			pred = unet(x)
			y_squeeze = torch.squeeze(y)
			totalValLoss += lossFunc(pred, y_squeeze)

			_, class_preds = torch.max(pred, 1)
			correct_tensor = class_preds.eq(y_squeeze.data.view_as(class_preds))
			total_correct = np.sum(correct_tensor.cpu().numpy())
			totalValAccuracy += total_correct / (config.INPUT_IMAGE_WIDTH * config.INPUT_IMAGE_HEIGHT * config.BATCH_SIZE)

	# calculate teh average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps
	avgTrainAccuracy = totalTrainAccuracy / trainSteps
	avgValAccuracy = totalValAccuracy / valSteps

	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["val_loss"].append(avgValLoss.cpu().detach().numpy())
	H["train_acc"].append(avgTrainAccuracy)
	H["val_acc"].append(avgValAccuracy)

	# print the model training and calidation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
	print("Train loss: {:.6f}, Val loss: {:.4f}".format(
		avgTrainLoss, avgValLoss))
	print("Train acc: {:.6f}, Val acc: {:.4f}".format(
		avgTrainAccuracy, avgValAccuracy))

# display the total time needed to preform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime-startTime))

torch.save(unet, config.MODEL_PATH)
