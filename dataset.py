# import the necessary packages
from torch.utils.data import Dataset
import cv2
from tqdm.auto import tqdm

class FloodDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms
		self.data = []
		for imagePath, maskPath in tqdm(zip(self.imagePaths, self.maskPaths)):
			image = cv2.imread(imagePath)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			mask = cv2.imread(maskPath, 0)
			if self.transforms is not None:
				image = self.transforms(image)
				mask = self.transforms(mask)
			self.data.append((image, mask))

	def __len__(self):
		return len(self.imagePaths)

	def __getitem__(self, idx):
		return self.data[idx]
