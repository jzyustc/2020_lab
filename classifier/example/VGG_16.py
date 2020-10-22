"""
VGG-16 for CIFAR-10 classifier
"""
# Last Modified : 2020/10/22, by jzy_ustc

from classifier.BaseNet import BaseNet
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim


class VGG_16(BaseNet):

	def __init__(self, batch_size, device):
		super(VGG_16, self).__init__(batch_size, device)

		self.train_data, self.test_data = self.load(batch_size)

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)
		self.bn_conv1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
		self.bn_conv2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
		self.bn_conv3 = nn.BatchNorm2d(128)
		self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)
		self.bn_conv4 = nn.BatchNorm2d(128)
		self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
		self.bn_conv5 = nn.BatchNorm2d(256)
		self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
		self.bn_conv6 = nn.BatchNorm2d(256)

		self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

		self.bn_fc1 = nn.BatchNorm1d(256 * 4 * 4)
		self.fc1 = nn.Linear(256 * 4 * 4, 300)
		self.bn_fc2 = nn.BatchNorm1d(300)
		self.fc2 = nn.Linear(300, 10)

		self.softmax = nn.Softmax(dim=1)

		self.net_setting()

	def net_setting(self):
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
		self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[4, 7])

	def forward(self, x):

		# 3-64, [32,32]
		x = F.relu(self.bn_conv1(self.conv1(x)))
		# 64-64, [32,32]
		x = F.relu(self.bn_conv2(self.conv2(x)))
		# 64-64 pool, [16,16]
		x = self.pool(x)

		# 64-128, [16,16]
		x = F.relu(self.bn_conv3(self.conv3(x)))
		# 128-128, [16,16]
		x = F.relu(self.bn_conv4(self.conv4(x)))
		# 128-128 pool, [8,8]
		x = self.pool(x)

		# 128-256, [8,8]
		x = F.relu(self.bn_conv5(self.conv5(x)))
		# 256-256 *2, [8,8]
		x = F.relu(self.bn_conv6(self.conv6(x)))
		x = F.relu(self.bn_conv6(self.conv6(x)))
		# 256-256 pool, [4,4]
		x = self.pool(x)

		x = self.bn_fc1(x.view(-1, 256 * 4 * 4))
		x = self.bn_fc2(F.relu(self.fc1(x)))
		x = self.softmax(self.fc2(x))
		return x

	def load(self, batch_size: int):
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		trainset = torchvision.datasets.CIFAR10('./data', download=False, train=True, transform=transform)
		testset = torchvision.datasets.CIFAR10('./data', download=False, train=False, transform=transform)

		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

		return list(trainloader), list(testloader)
