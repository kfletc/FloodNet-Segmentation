# import the nexessary packages
import config
from dataset import FloodDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from imutils import paths
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms

NUMBER_OF_PLOTS = 5
RANDOM_TESTS = True


def prepare_plot(origImage, origMask, predMask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=len(origImage), ncols=3, figsize=(10, 10))

    # plot the orinal image, its mask, and the predicted mask
    for i in range(len(origImage)):
        ax[i, 0].imshow(origImage[i])
        ax[i, 1].imshow(origMask[i])
        ax[i, 2].imshow(predMask[i])

    # set the titles of the subplots
    ax[0, 0].set_title("Image")
    ax[0, 1].set_title("Orginal Mask")
    ax[0, 2].set_title("Predicted Mask")

    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()


if os.path.exists('PickleDumps/test_pickle.pickle'):
    pickle_file = open('PickleDumps/test_pickle.pickle', 'rb')
    testDS = pickle.load(pickle_file)
    pickle_file.close()
else:
    test_imagePaths = sorted(list(paths.list_images(config.IMAGE_TEST_DATASET_PATH)))
    test_maskPaths = sorted(list(paths.list_images(config.MASK_TEST_DATASET_PATH)))
    # define transformations
    transforms_image = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
                                           transforms.ToTensor()])

    transforms_mask = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
                                          transforms.PILToTensor()])

    testDS = FloodDataset(imagePaths=test_imagePaths, maskPaths=test_maskPaths,
                          transforms=transforms_image, transforms_mask=transforms_mask)

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

for x, y in tqdm(testLoader):
    # send the input to the device
    (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

    pred = unet(x)

    # get number of plots to make
    numViz = 0
    if NUMBER_OF_PLOTS < config.BATCH_SIZE:
        numViz = NUMBER_OF_PLOTS
    else:
        numViz = config.BATCH_SIZE

    # get numpy arrays batch from tensor
    pimgb = pred.squeeze()
    ximgb = x.squeeze()
    yimgb = y.squeeze()
    ximgb = ximgb.cpu().numpy()
    yimgb = yimgb.cpu().numpy()
    pimgb = pimgb.cpu().detach().numpy()

    for j in tqdm(range(numViz)):
        # get single image from batch
        ximg = ximgb[j, :, :, :]
        yimg = yimgb[j, :, :]
        pimg = pimgb[j, :, :, :]

        # reorder to width, hight, channels
        ximg = np.transpose(ximg, (1, 2, 0))
        pimg = np.transpose(pimg, (1, 2, 0))

        # get the max of the 10 channels to reshape to 2d array
        pimg = np.argmax(pimg, axis=2)

        ximgs.append(ximg)
        yimgs.append(yimg)
        pimgs.append(pimg)
    break

prepare_plot(ximgs, yimgs, pimgs)
plt.savefig(config.IMAGE_EX_PATH)

k = input("Press close to exit")
