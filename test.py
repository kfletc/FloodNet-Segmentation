# import the nexessary packages
from torch.nn import CrossEntropyLoss

import config
from dataset import FloodDataset
import numpy as np
import torch
import os
from imutils import paths
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms

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

testLoader = DataLoader(testDS, shuffle=False,
                        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                        num_workers=0)

print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)

# results from test data
classes = ['Background', 'Building Flooded', 'Building Non-Flooded', 'Road Flooded', 'Road Non-Flooded',
           'Water', 'Tree', 'Vehicle', 'Pool', 'Grass']

testSteps = len(testDS) // config.BATCH_SIZE
lossFunc = CrossEntropyLoss()
totalTestLoss = 0
totalTestAcc = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

for x, y in tqdm(testLoader):
    # send the input to the device
    (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

    pred = unet(x)
    y_squeeze = torch.squeeze(y)
    loss = lossFunc(pred, y_squeeze)
    totalTestLoss += loss
    # calculate pixel accuracy
    _, class_preds = torch.max(pred, 1)
    y_preds = y_squeeze.data.view_as(class_preds)
    y_preds_cpu = y_preds.cpu()
    correct_tensor = class_preds.eq(y_preds)
    correct = correct_tensor.cpu().numpy()
    total_correct = np.sum(correct)
    totalTestAcc += total_correct / (config.INPUT_IMAGE_WIDTH * config.INPUT_IMAGE_HEIGHT * config.BATCH_SIZE) * 100.0
    for b in range(config.BATCH_SIZE):
        for c, x_data in enumerate(y_preds_cpu[b]):
            for d, xy_data in enumerate(x_data):
                label = xy_data.data
                class_correct[label] += correct[b][c][d].item()
                class_total[label] += 1

avgTestLoss = totalTestLoss / testSteps
avgTestAcc = totalTestAcc / testSteps
print("Test loss: {:.6f}".format(avgTestLoss))
print("Test acc: {:.6f}".format(avgTestAcc))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

k=input("Press close to exit")
