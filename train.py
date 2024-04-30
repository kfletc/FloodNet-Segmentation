# import the necessary pachages
from dataset import FloodDataset
import config
# loss function
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os

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

print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")
print(f"[INFO] found {len(valDS)} examples in the val set...")


# create the training, test, and val data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
			 batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
			 num_workers=os.cpu_count())
testLoader = DataLoader(testDS, shuffle=False,
			 batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
			 num_workers=os.cpu_count())
valLoader = DataLoader(valDS, shuffle=False,
			 batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
			 num_workers=os.cpu_count())

# initialize UNet model
##unet = UNet().to(config.DEVICE)

# initialize loss function and optimizer
##lossFunc = 
##opt = 

# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE
valSteps = len(valDS) // config.BATCH_SIZE

# initialize a dictionary to store training history
H = {"train_loss": [], "val_loss":[]}

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
	print(f"[INFO] starting epochs {e}")

	# set the model in training mode
	##unet.train()

	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0
	
	# loop over the training set
	for (x, y) in trainLoader:
		print("test1")
		# send the input to the device
		(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

		# preform a forward pass and calculate the training loss
		##pred = unet(x)
		##loss = lossFunc(pred, y)

		# first, zero out any previously accumulated gradients, then
		# preform backproagation, and then update model parameters
		##opt.zero_grad()
		##loss.backward()
		##opt.step()

		# add teh loss to the total training loss so far
		##totalTrainLoss += loss

	# switch off autograd
	with torch.no_grad():
		# set model in evaluation mode
		##unet.eval()

		# loop over the validation set
		for (x, y) in valLoader:
			print("test2")
			# send the input to the device
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

			# make the predictions and calculate the validation loss
			##pred = unet(x)
			##totalValLoss += lossFunc(pred, y)

	# calculate teh average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / testSteps

	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["val_loss"].append(avgValLoss.cpu().detach().numpy())

	# print the model training and calidation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
	print("Train loss: {:.6f}, Val loss: {:.4f}".format(
		avgTrainLoss, avgValLoss))

# display the total time needed to preform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime-startTime))
