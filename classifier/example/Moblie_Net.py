"""
MobileNet v2 for CIFAR-10 classifier
"""
# Last Modified : 2020/10/24, by jzy_ustc

from classifier.BaseNet import BaseNet
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim


class MobileBlock_v2(nn.Module):

	def __init__(self, in_channels, out_channels, expansion, stride):
		super(MobileBlock_v2, self).__init__()

		self.stride = stride
		channels = in_channels * expansion

		self.conv1 = nn.Conv2d(in_channels, channels, 1, stride=1)
		self.bn1 = nn.BatchNorm2d(channels)

		self.conv2 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, groups=channels)
		self.bn2 = nn.BatchNorm2d(channels)

		self.conv3 = nn.Conv2d(channels, out_channels, 1, stride=1)
		self.bn3 = nn.BatchNorm2d(out_channels)

		self.shortcut = nn.Sequential()
		if stride == 1 and in_channels != out_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0),
				nn.BatchNorm2d(out_channels)
			)

	def forward(self, x):
		identity = self.shortcut(x)

		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = self.bn3(self.conv3(x))

		x = identity + x if self.stride == 1 else x

		return x


class MobileNet_v2(BaseNet):
	# (expansion, out_planes, num_blocks, stride)
	## in : [32,32,32]
	cfg = [(1, 16, 1, 1),  ## [16,32,32]
		   (6, 24, 2, 2),  ## [24,16,16]
		   (6, 32, 3, 2),  ## [32,8,8]
		   (6, 64, 4, 2),  ## [64,4,4]
		   (6, 96, 3, 1)]  ## [96,4,4]


	def __init__(self, batch_size, device):
		super(MobileNet_v2, self).__init__(batch_size, device)

		self.train_data, self.test_data = self.load(batch_size)

		## [3,32,32] : [32,32,32]
		self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(32)

		## [32,32,32] : [96,4,4]
		in_channels = 32
		layer = []
		for expansion, out_channels, num_blocks, stride in self.cfg:
			strides = [stride] + [1] * (num_blocks - 1)
			for stride in strides:
				layer.append(MobileBlock_v2(in_channels, out_channels, expansion, stride))
				in_channels = out_channels
		self.layer = nn.Sequential(*layer)

		## [96,4,4] : [160,4,4] [160,1,1] [10]
		self.conv2 = nn.Conv2d(96, 160, kernel_size=1)
		self.bn2 = nn.BatchNorm2d(160)
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Linear(160, 10)

		self.net_setting()

	def net_setting(self):
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
		self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[4, 7])

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = self.layer(x)
		x = F.relu(self.bn2(self.conv2(x)))
		x = self.pool(x)
		x = x.view(-1, 160)
		x = self.fc(x)
		return x

	def load(self, batch_size: int):
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		trainset = torchvision.datasets.CIFAR10('./data', download=False, train=True, transform=transform)
		testset = torchvision.datasets.CIFAR10('./data', download=False, train=False, transform=transform)

		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

		return list(trainloader), list(testloader)
