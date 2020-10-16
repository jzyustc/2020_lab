"""
an example to use the BaseNet Class

an easy ConvNet to train FashionMNIST, with CrossEntropyLoss, SGD

load data by torchvision package
"""
# Last Modified : 2020/10/16, by jzy_ustc

from classifier.BaseNet import BaseNet
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim


class Net(BaseNet):

	def __init__(self, batch_size, device):
		super(Net, self).__init__(batch_size, device)

		self.train_data, self.test_data = self.load(batch_size)

		self.conv1 = nn.Conv2d(1, 30, 5)
		self.bn_conv1 = nn.BatchNorm2d(30)
		self.conv2 = nn.Conv2d(30, 60, 5)
		self.bn_conv2 = nn.BatchNorm2d(60)

		self.pool = nn.MaxPool2d(2, 2)

		self.bn_fc1 = nn.BatchNorm1d(60 * 4 * 4)
		self.fc1 = nn.Linear(60 * 4 * 4, 300)
		self.bn_fc2 = nn.BatchNorm1d(300)
		self.fc2 = nn.Linear(300, 10)

		self.net_setting()

	def net_setting(self):
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
		self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[4, 7])

	def forward(self, x):
		x = self.pool(F.relu(self.bn_conv1(self.conv1(x))))
		x = self.pool(F.relu(self.bn_conv2(self.conv2(x))))
		x = self.bn_fc1(x.view(-1, 60 * 4 * 4))
		x = self.bn_fc2(F.relu(self.fc1(x)))
		x = self.fc2(x)
		return x

	def load(self, batch_size: int):
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

		trainset = torchvision.datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
		testset = torchvision.datasets.FashionMNIST('./data', download=True, train=False, transform=transform)

		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

		return list(trainloader), list(testloader)
