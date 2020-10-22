"""
LeNet-5 for CIFAR-10 classifier
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


class LeNet_5(BaseNet):

	def __init__(self, batch_size, device):
		super(LeNet_5, self).__init__(batch_size, device)

		self.train_data, self.test_data = self.load(batch_size)

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0, stride=1)
		self.bn_conv1 = nn.BatchNorm2d(6)
		self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0, stride=1)
		self.bn_conv2 = nn.BatchNorm2d(16)

		self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

		self.bn_fc1 = nn.BatchNorm1d(16 * 5 * 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.bn_fc2 = nn.BatchNorm1d(120)
		self.fc2 = nn.Linear(120, 84)
		self.bn_fc3 = nn.BatchNorm1d(84)
		self.fc3 = nn.Linear(84, 10)

		self.softmax = nn.Softmax(dim=1)

		self.net_setting()

	def net_setting(self):
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
		self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[4, 7])

	def forward(self, x):
		x = self.pool(F.relu(self.bn_conv1(self.conv1(x))))
		x = self.pool(F.relu(self.bn_conv2(self.conv2(x))))
		x = self.bn_fc1(x.view(-1, 16 * 5 * 5))
		x = self.bn_fc2(F.relu(self.fc1(x)))
		x = self.bn_fc3(F.relu(self.fc2(x)))
		x = self.softmax(self.fc3(x))
		return x

	def load(self, batch_size: int):
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		trainset = torchvision.datasets.CIFAR10('./data', download=False, train=True, transform=transform)
		testset = torchvision.datasets.CIFAR10('./data', download=False, train=False, transform=transform)

		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

		return list(trainloader), list(testloader)
