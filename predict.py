# import the nexessary packages
import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader

NUMBER_OF_PLOTS = 5
RANDOM_TESTS = True

def prepare_plot(origImage, origMask, predMask):
	# initialize our figure
	figure, ax = plt.subplots(nrows=len(origImage), ncols=3, figsize=(10,10))

	# plot the orinal image, its mask, and the predicted mask
	for i in range(len(origImage)):
		ax[i,0].imshow(origImage[i])
		ax[i,1].imshow(origMask[i])
		ax[i,2].imshow(predMask[i])

	# set the titles of the subplots
	ax[0,0].set_title("Image")
	ax[0,1].set_title("Orginal Mask")
	ax[0,2].set_title("Predicted Mask")

	# set the layout of the figure and display it
	figure.tight_layout()
	figure.show()

def make_predictions(model, imagePath, maskPath):
	# set model to evaluation mode
	model.eval()

	# turn off gradient tracking
	with torch.no_grad():
		# laod the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0

		# resize the image and make a copy of it for visualization
		image = cv2.resize


if os.path.exists('PickleDumps/test_pickle'):
	pickle_file = open('PickleDumps/test_pickle', 'rb')
	testDS = pickle.load(pickle_file)
	pickle_file.close()
else:
	testDS = FloodDataset(imagePaths=test_imagePaths, maskPaths=test_maskPaths,
			      transforms=transforms)
	
print(f"[INFO] found {len(testDS)} examples in the test set...")

testLoader = DataLoader(testDS, shuffle=RANDOM_TESTS,
			 batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
			 num_workers=0)

print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)


# images to save for plot
ximgs = []
yimgs = []
pimgs = []

for (i, (x, y)) in tqdm(enumerate(testLoader)):
	# send the input to the device
	(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

	pred = unet(x)
	

	# generate plots for first batch
	if i == 0:
		# get number of plots to make
		numViz = 0
		if NUMBER_OF_PLOTS < config.BATCH_SIZE:
			numViz = NUMBER_OF_PLOTS
		else:
			numViz = config.BATCH_SIZE

		for j in tqdm(range(numViz)):
			print(j)
			pimg = pred.squeeze()
			pimg = pimg.cpu().detach().numpy()
			ximg = x.squeeze()
			yimg = y.squeeze()
			ximg = ximg.cpu().numpy()
			yimg = yimg.cpu().numpy()
			ximg = ximg[j,:,:,:]
			yimg = yimg[j,:,:]
			print(pimg.shape)
			pimg = pimg[j,:,:,:]
			ximg = np.transpose(ximg, (1, 2, 0))
			pimg = np.transpose(pimg, (1, 2, 0))
			pimg = pimg[:,:,0]
			
			ximgs.append(ximg)
			yimgs.append(yimg)
			pimgs.append(pimg)
	
prepare_plot(ximgs, yimgs, pimgs)	
	
