"""
ResNet-18 for CIFAR-10 classifier
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


class BasicBlock(nn.Module):

	def __init__(self, in_channels, out_channels):
		super(BasicBlock, self).__init__()

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

	def forward(self, x):

		identity = x

		x = self.relu(self.bn1(self.conv1(x)))
		x = self.bn2(self.conv2(x))

		if self.downsample is not None:
			identity = self.downsample(identity)

		x += identity
		x = self.relu(x)

		return x


class ResNet_18(BaseNet):

	def __init__(self, batch_size, device):
		super(ResNet_18, self).__init__(batch_size, device)

		self.train_data, self.test_data = self.load(batch_size)

		## [3,32,32] : [64,32,32] [64,16,16]
		self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
		self.bn = nn.BatchNorm2d(64)
		self.relu = nn.ReLU()

		self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

		## [64,16,16] :[64,16,16] [128,8,8] [256,4,4] [512,2,2]
		self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
		self.layer2 = nn.Sequential(BasicBlock(64, 128), BasicBlock(128, 128))
		self.layer3 = nn.Sequential(BasicBlock(128, 256), BasicBlock(256, 256))
		self.layer4 = nn.Sequential(BasicBlock(256, 512), BasicBlock(512, 512))

		## [512,2,2] : [512,1,1] [10,1,1]
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512, 10)

		self.net_setting()

	def net_setting(self):
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
		self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[15, 24])

	def forward(self, x):
		x = self.relu(self.bn(self.conv(x)))
		x = self.pool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avgpool(x)
		x = x.view(-1, 512)
		x = self.fc(x)
		return x

	def load(self, batch_size: int):
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
		])

		trainset = torchvision.datasets.CIFAR10('./data', download=False, train=True, transform=transform)
		testset = torchvision.datasets.CIFAR10('./data', download=False, train=False, transform=transform)

		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

		return list(trainloader), list(testloader)
