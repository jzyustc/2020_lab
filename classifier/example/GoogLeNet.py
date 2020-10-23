"""
GoogLeNet for CIFAR-10 classifier
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


class Inception(nn.Module):

	def __init__(self, in_channels, out_channels_1, out_channels_2, out_channels_3, out_channels_4):
		super(Inception, self).__init__()

		# line 1 : 1x1
		self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_1, kernel_size=1, padding=0, stride=1)

		# line 2 : 1x1 then 3x3
		self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_2[0], kernel_size=1, padding=0,
							  stride=1)
		self.p2_2 = nn.Conv2d(in_channels=out_channels_2[0], out_channels=out_channels_2[1], kernel_size=3, padding=1,
							  stride=1)

		# line 3 : 1x1 then 5x5
		self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_3[0], kernel_size=1, padding=0,
							  stride=1)
		self.p3_2 = nn.Conv2d(in_channels=out_channels_3[0], out_channels=out_channels_3[1], kernel_size=5, padding=2,
							  stride=1)

		# line 4 : 3x3 pool then 1x1
		self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
		self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_4, kernel_size=3, padding=1,
							  stride=1)

	def forward(self, x):
		x1 = F.relu(self.p1_1(x))
		x2 = F.relu(self.p2_2(self.p2_1(x)))
		x3 = F.relu(self.p3_2(self.p3_1(x)))
		x4 = F.relu(self.p4_2(self.p4_1(x)))
		x = torch.cat((x1, x2, x3, x4), dim=1)
		return x


class InceptionAux(nn.Module):

	def __init__(self, in_channels, out_channels):
		super(InceptionAux, self).__init__()

		# [in_channels,7,7] : [in_channels,4,4] [128,4,4] [2048,1,1] [1024,1,1][10,1,1]
		self.pool = nn.AdaptiveAvgPool2d((4, 4))
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
		self.bn = nn.BatchNorm2d(128)
		self.fc1 = nn.Linear(128 * 4 * 4, 1024)
		self.fc2 = nn.Linear(1024, 10)

	def forward(self, x):
		# aux1: N x 512 x 7 x 7, aux2: N x 528 x 7 x 7
		x = self.pool(x)
		# aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
		x = self.bn(self.conv(x))
		# N x 128 x 4 x 4
		x = torch.flatten(x, 1)
		x = F.dropout(x, 0.5, training=self.training)
		# N x 2048
		x = F.relu(self.fc1(x), inplace=True)
		x = F.dropout(x, 0.5, training=self.training)
		# N x 1024
		x = self.fc2(x)
		# N x 10
		return x


class GoogLeNet(BaseNet):

	def __init__(self, batch_size, device, aux_logits=True):
		super(GoogLeNet, self).__init__(batch_size, device)

		self.train_data, self.test_data = self.load(batch_size)

		self.aux_logits = aux_logits

		## [3,32,32] : [64,32,32] [64,16,16]
		self.seq1 = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(64)
		)

		## [64,16,16] : [64,16,16] [192,14,14] [192,7,7]
		self.seq2 = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(192)
		)

		## [192,7,7] : [256,7,7] [480,7,7]
		self.seq3 = nn.Sequential(
			Inception(192, 64, (96, 128), (16, 32), 32),
			nn.BatchNorm2d(256),
			Inception(256, 128, (128, 192), (32, 96), 64),
			nn.BatchNorm2d(480),
		)

		## [480,7,7] : [512,7,7]
		self.seq4_1 = nn.Sequential(
			Inception(480, 192, (96, 208), (16, 48), 64),
			nn.BatchNorm2d(512),
		)

		# aux1 : [512,7,7]
		self.aux1 = InceptionAux(512, 10)

		## [512,7,7] : [512,7,7] [528,7,7]
		self.seq4_2 = nn.Sequential(
			Inception(512, 160, (112, 224), (24, 64), 64),
			nn.BatchNorm2d(512),
			Inception(512, 128, (128, 256), (24, 64), 64),
			nn.BatchNorm2d(512),
			Inception(512, 112, (144, 288), (24, 64), 64),
			nn.BatchNorm2d(528),
		)

		# aux2 : [528,7,7]
		self.aux2 = InceptionAux(528, 10)

		## [528,7,7] : [832,7,7]
		self.seq4_3 = nn.Sequential(
			Inception(528, 256, (160, 320), (32, 128), 128),
			nn.BatchNorm2d(832),
		)

		## [832,7,7] : [832,7,7] [1024,7,7]
		self.seq5 = nn.Sequential(
			Inception(832, 256, (160, 320), (32, 128), 128),
			nn.BatchNorm2d(832),
			Inception(832, 384, (192, 384), (48, 128), 128),
			nn.BatchNorm2d(1024),
		)

		## [1024,7,7] : [1024,1,1] [1024,1,1] [10,1,1] [10,1,1]
		self.seq6 = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Dropout(0.4),
		)
		self.fc = nn.Linear(1024, 10)
		self.softmax = nn.Softmax(dim=1)

		self.net_setting()

	def net_setting(self):
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
		self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[4, 7])

	def forward(self, x):

		# TODO : aux???????

		x = self.seq1(x)
		x = self.seq2(x)
		x = self.seq3(x)
		x = self.seq4_1(x)

		if self.training and self.aux_logits:  # eval model lose this layer
			aux1 = self.aux1(x)

		x = self.seq4_2(x)

		if self.training and self.aux_logits:  # eval model lose this layer
			aux2 = self.aux2(x)

		x = self.seq4_3(x)
		x = self.seq5(x)
		x = self.seq6(x)

		x = x.view(-1, 1024)

		x = self.fc(x)
		x = self.softmax(x)

		if self.training and self.aux_logits:  # eval model lose this layer
			return x, aux2, aux1

		return x

	def load(self, batch_size: int):
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		trainset = torchvision.datasets.CIFAR10('./data', download=False, train=True, transform=transform)
		testset = torchvision.datasets.CIFAR10('./data', download=False, train=False, transform=transform)

		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

		return list(trainloader), list(testloader)
