# import the necessary packages
from torch.utils.data import Dataset
import cv2

class FloodDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms

	def __len__(self):
		return len(self.imagePaths)

	def __getitem__(self, idx):
		imagePath = self.imagePaths[idx]

		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(self.maskPaths[idx], 0)

		if self.transforms is not None:
			image = self.transforms(image)
			mask = self.transforms(mask)

		return (image, mask)
