# import the nexessary packages
import torch
import os

# base path of the dataset
DATASET_TRAIN_PATH = os.path.join("FloodNet-Supervised_v1.0", "train")
DATASET_TEST_PATH = os.path.join("FloodNet-Supervised_v1.0", "test")
DATASET_VAL_PATH = os.path.join("FloodNet-Supervised_v1.0", "val")

# define the path to the images and masks dataset
IMAGE_TRAIN_DATASET_PATH = os.path.join(DATASET_TRAIN_PATH, "train-org-img")
MASK_TRAIN_DATASET_PATH = os.path.join(DATASET_TRAIN_PATH, "train-label-img")

# define the path to the test images and masks dataset
IMAGE_TEST_DATASET_PATH = os.path.join(DATASET_TEST_PATH, "test-org-img")
MASK_TEST_DATASET_PATH = os.path.join(DATASET_TEST_PATH, "test-label-img")

# define the path to the val images and masks dataset
IMAGE_VAL_DATASET_PATH = os.path.join(DATASET_VAL_PATH, "val-org-img")
MASK_VAL_DATASET_PATH = os.path.join(DATASET_VAL_PATH, "val-label-img")

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_LEVELS = 3

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 5
BATCH_SIZE = 64

# define the image dimensions
INPUT_IMAGE_WIDTH = 600
INPUT_IMAGE_HEIGHT = 400

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

