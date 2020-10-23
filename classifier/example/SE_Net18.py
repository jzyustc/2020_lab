"""
SE-ResNet-18 for CIFAR-10 classifier
"""
# Last Modified : 2020/10/23, by jzy_ustc

from classifier.BaseNet import BaseNet
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim


class SEBlock(nn.Module):

	def __init__(self, in_channels, out_channels, r=16):
		super(SEBlock, self).__init__()

		self.out_channels = out_channels
		# judge down_sample
		self.downsample = None
		if (in_channels != out_channels):
			self.downsample = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
										stride=2, padding=0)

		self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
							   stride=1 if self.downsample is None else 2, padding=1)
		self.bn1 = nn.BatchNorm2d(out_channels)

		self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.relu = nn.ReLU(inplace=True)

		# SE part
		self.squeeze = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(out_channels, out_channels // r),
			nn.ReLU(inplace=True),
			nn.Linear(out_channels // r, out_channels),
		)
		self.scale = nn.Sigmoid()

	def forward(self, x):

		identity = x

		x = self.relu(self.bn1(self.conv1(x)))
		x = self.bn2(self.conv2(x))

		if self.downsample is not None:
			identity = self.downsample(identity)

		x += identity
		x = self.relu(x)

		# SE part
		b,c,_,_ = x.shape
		se = self.squeeze(x).view(-1, self.out_channels)
		se = self.fc(se).view(b,c,1,1)
		se = self.scale(se)

		x = torch.mul(x, se)

		return x


class SE_ResNet_18(BaseNet):

	def __init__(self, batch_size, device):
		super(SE_ResNet_18, self).__init__(batch_size, device)

		self.train_data, self.test_data = self.load(batch_size)

		## [3,32,32] : [64,32,32] [64,16,16]
		self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
		self.bn = nn.BatchNorm2d(64)
		self.relu = nn.ReLU()

		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

		## [64,16,16] : [128,8,8] [256,4,4]
		self.layer1 = nn.Sequential(SEBlock(64, 128), SEBlock(128, 128))
		self.layer2 = nn.Sequential(SEBlock(128, 256), SEBlock(256, 256))

		## [512,4,4] : [512,1,1] [10,1,1]
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(256, 10)

		self.net_setting()

	def net_setting(self):
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
		self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[4, 7])

	def forward(self, x):
		x = self.relu(self.bn(self.conv(x)))
		x = self.pool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.avgpool(x)
		x = x.view(-1, 256)
		x = self.fc(x)
		return x

	def load(self, batch_size: int):
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		trainset = torchvision.datasets.CIFAR10('./data', download=False, train=True, transform=transform)
		testset = torchvision.datasets.CIFAR10('./data', download=False, train=False, transform=transform)

		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

		return list(trainloader), list(testloader)
