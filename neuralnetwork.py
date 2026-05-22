import torch
from torch import nn
import torch.nn.functional as F
from utils.setup import device

class NeuralNetwork(nn.Module):
	"""Simple 3-layer MLP for image classification."""

	def __init__(self, num_classes = 10):
		super().__init__()
		self.flatten = nn.Flatten()

		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28*28, 512),
			nn.ReLU(),
			nn.Linear(512,512),
			nn.ReLU(),
			nn.Linear(512,num_classes)
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits
	
	def model_name(self):
		return "Neural Net"

class CNN(nn.Module):
	"""Two conv blocks (conv → BN → ReLU → maxpool) followed by two fully connected layers."""

	def __init__(self, in_channels = 1, num_classes = 10, imgsz = (28,28)):
		super().__init__()
		self.flatten = nn.Flatten(1,3)

		conv1_out = 32
		self.conv1 = nn.Conv2d(in_channels,conv1_out,kernel_size=3,stride=1,padding=1)
		self.bn1 = nn.BatchNorm2d(conv1_out)
		
		self.conv2 = nn.Conv2d(conv1_out,conv1_out*2,kernel_size=3,stride=1,padding=1)
		self.bn2 = nn.BatchNorm2d(conv1_out*2)
		
		self.halving_pool = nn.MaxPool2d(kernel_size=2,stride=2)

		self.conv_dropout = nn.Dropout(0.1)
		self.fully_connected_dropout = nn.Dropout(0.2)

		fully_connected_imgsz = (int(imgsz[0]/4), int(imgsz[1]/4))
		fully_connected_in = int(conv1_out*2*(fully_connected_imgsz[0] * fully_connected_imgsz[1]))
		self.fully_connected = nn.Linear(fully_connected_in,256)

		self.fully_connected2 = nn.Linear(256,num_classes)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		
		x = self.halving_pool(x)
		x = self.conv_dropout(x)
		x = F.relu(self.bn2(self.conv2(x)))
		
		x = self.halving_pool(x)
		x = self.flatten(x)
		x = self.fully_connected_dropout(x)

		x = F.relu(self.fully_connected(x))
		x = self.fully_connected2(x)
		
		return x
	
	def model_name(self):
		return "CNN"

class ConvBNReLu(nn.Module):
	"""Reusable Conv2d → BatchNorm2d → ReLU block."""

	def __init__(self,in_channels=None,out_channels=None,kernel_size=3,stride=1,padding=1):
		super().__init__()
		
		if None in [in_channels,out_channels]:
			raise ValueError("in_channels and out_channels must be initialised")
		
		self.block: nn.Module = nn.Sequential(
			nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
			nn.BatchNorm2d(out_channels),
			nn.ReLU()
		)
	
	def forward(self, x):
		return self.block(x)



class CNN2(nn.Module):
	"""Deeper CNN using Conv_BN_ReLu blocks with stride-2 downsampling instead of maxpool."""

	def __init__(self,in_channels=1,num_classes=10,imgsz = (28,28)):
		super().__init__()
		self.relu = nn.ReLU()

		self.flatten = nn.Flatten()
		self.dropout = nn.Dropout2d(0.15)

		self.conv1out = 32
		self.conv1 = nn.Conv2d(in_channels,self.conv1out,kernel_size=3,stride=1,padding=1)
		self.bn1 = nn.BatchNorm2d(self.conv1out)

		self.conv2out = self.conv1out*2
		self.block2 = ConvBNReLu(self.conv1out,self.conv2out,kernel_size=3,stride=2,padding=1)

		self.conv3out = self.conv2out*2
		self.block3 = ConvBNReLu(self.conv2out,self.conv3out,kernel_size=3,stride=2,padding=1)

		fully_connected_in = int(self.conv3out * (imgsz[0]/4 * imgsz[1]/4))
		self.fully_connected = nn.Linear(fully_connected_in,256)
		self.output = nn.Linear(256,num_classes)


	def forward(self, x):
		x = nn.functional.relu(self.bn1(self.conv1(x)))
		# save x as y

		x = self.block2(x)
		x = self.dropout(x)
		# resize y, add to x ?
		x = self.block3(x)
		x = self.dropout(x)

		x = self.flatten(x)
		x = self.fully_connected(x)
		x = self.output(x)
		return x
	
	def model_name(self):
		return "CNN2"

class ResBlock(nn.Module):
	def __init__(self, in_channels,out_channels,kernel_size=3,stride=2,padding=1):
		super().__init__()
		
		self.relu = nn.ReLU()
		self.block1 = ConvBNReLu(in_channels,out_channels,kernel_size=3,stride=1,padding=padding)
		self.block2 = nn.Sequential(
			nn.Conv2d(out_channels,out_channels,kernel_size,stride,padding),
			nn.BatchNorm2d(out_channels),
		)
		
		if stride != 1 or (in_channels != out_channels):
			self.project = nn.Sequential(
				nn.Conv2d(in_channels,out_channels,1,stride,padding=0),
				nn.BatchNorm2d(out_channels)
			)
		else:
			self.project = nn.Identity()

	def forward(self, x: torch.Tensor):
		res = self.project(x)
		x = self.block1(x)
		x = self.relu(x)
		x = self.block2(x)
		x = x + res
		x = self.relu(x)
		return x

class ResNet(nn.Module):
	"""Deeper CNN using Conv_BN_ReLu blocks with stride-2 downsampling instead of maxpool."""

	def __init__(self,in_channels=1,num_classes=10,imgsz = (28,28)):
		super().__init__()
		
		self.flatten = nn.Flatten()
		self.dropout = nn.Dropout2d(0.15)
		self.relu = nn.ReLU()

		block1_out = 32
		self.block1 = ResBlock(in_channels,block1_out,kernel_size=3,stride=1,padding=1)

		block2_out = block1_out*2
		self.block2 = ResBlock(block1_out,block2_out,kernel_size=3,stride=2,padding=1)

		block3_out = block2_out*2
		self.block3 = ResBlock(block2_out,block3_out,kernel_size=3,stride=2,padding=1)

		fully_connected_in = int(block3_out * (imgsz[0]/4 * imgsz[1]/4))
		self.fully_connected = nn.Linear(fully_connected_in,256)
		self.output = nn.Linear(256,num_classes)

	def forward(self, x):
		x = self.block1(x)
		x = self.dropout(x)
		x = self.block2(x)
		x = self.dropout(x)
		x = self.block3(x)
		
		x = self.flatten(x)
		x = self.fully_connected(x)
		x = self.output(x)

		return x
	
	def model_name(self):
		return "ResNet"

if __name__ == "__main__":
	model = NeuralNetwork().to(device)
	print(model)

	model = CNN().to(device)
	print(model)

	model = CNN2().to(device)
	print(model)

	model = ResNet().to(device)
	print(model)