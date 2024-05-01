# FloodNet-Segmentation
Testing semantic segmentation on the FloodNet dataset with a neural network structure similar to U-Net
Both dataset and pickle dumps of images are too big for github
You can find the dataset at https://github.com/BinaLab/FloodNet-Supervised_v1.0?tab=readme-ov-file
config.py has the folder names for the data in the dataset
currently it expects the whole dataset in a directory called FloodNet-Supervised_v1.0 and 3 subdirectories, train, val, and test with the data in them.
preprocessing.py is for resizing the images and storing them in pickle dumps, however if no pickle dumps exist the other scripts will create them as well.
train.py is for training the network
test.py is for running the test data through the network
genplots.py is for viewing example images with their actual and predicted masks
