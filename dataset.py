# import the necessary packages
from torch.utils.data import Dataset
from torch import Tensor
import torch
import cv2
from tqdm.auto import tqdm

class FloodDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms, transforms_mask):
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms
		self.transforms_mask = transforms_mask
		self.data = []
		for imagePath, maskPath in tqdm(zip(self.imagePaths, self.maskPaths)):
			image = cv2.imread(imagePath)
			mask = cv2.imread(maskPath, 0)
			if self.transforms is not None:
				image = self.transforms(image)
				mask = self.transforms_mask(mask)
			mask = mask.to(torch.long)
			self.data.append((image, mask))

	def __len__(self):
		return len(self.imagePaths)

	def __getitem__(self, idx):
		return self.data[idx]
