"""
DPN-92 for CIFAR-10 classifier
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


class _DPNLayer(nn.Module):

	def __init__(self, in_channels, mid_channels, out_channels, dense_channels, stride, is_shortcut):
		super(_DPNLayer, self).__init__()

		self.is_shortcut = is_shortcut
		self.out_channels = out_channels

		self.conv_part1 = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, 1),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True)
		)

		self.conv_part2 = nn.Sequential(
			nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, groups=32),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True)
		)

		self.conv_part3 = nn.Sequential(
			nn.Conv2d(mid_channels, out_channels + dense_channels, 1),
			nn.BatchNorm2d(out_channels + dense_channels)
		)

		if self.is_shortcut:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels + dense_channels, 1, stride=stride),
				nn.BatchNorm2d(out_channels + dense_channels)
			)

		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		shortcut = x if not self.is_shortcut else self.shortcut(x)

		x = self.conv_part1(x)
		x = self.conv_part2(x)
		x = self.conv_part3(x)

		d = self.out_channels
		x = torch.cat([shortcut[:, :d, :, :] + x[:, :d, :, :], shortcut[:, d:, :, :], x[:, d:, :, :]], dim=1)
		x = self.relu(x)

		return x


class _DPNBlock(nn.Sequential):

	def __init__(self, in_channels, mid_channels, out_channels, dense_channels, stride, num_layers):
		super(_DPNBlock, self).__init__()

		self.layer = nn.Sequential()
		self.layer.add_module("layer1",
							  _DPNLayer(in_channels, mid_channels, out_channels, dense_channels, stride, True))

		self.in_channels = out_channels + 2 * dense_channels

		for i in range(1, num_layers):
			self.layer.add_module("layer%d" % (i + 1),
								  _DPNLayer(self.in_channels, mid_channels, out_channels, dense_channels, 1, False))
			self.in_channels += dense_channels

	def forward(self, x):
		return self.layer(x)


class DPNNet(BaseNet):

	def __init__(self, batch_size, device, blocks=(3, 12, 3), in_channels=64, mid_channels=96, out_channels=256,
				 dense_channels=(16, 24, 128)):
		super(DPNNet, self).__init__(batch_size, device)

		self.train_data, self.test_data = self.load(batch_size)

		# [3,32,32] : [64,32,32] [64,16,16]
		self.features = nn.Sequential()
		self.features.add_module("conv0", nn.Conv2d(3, in_channels, kernel_size=3, padding=1))
		self.features.add_module("norm0", nn.BatchNorm2d(in_channels))
		self.features.add_module("relu0", nn.ReLU(inplace=True))
		self.features.add_module("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

		# [64,16,16] : [256+16*3,8,8] [512+24*12,4,4] [1024+128*3,2,2]
		for i, layer_num in enumerate(blocks):
			block = _DPNBlock(in_channels, mid_channels, out_channels, dense_channels[i], 1 if i == 0 else 2, blocks[i])
			self.features.add_module("DPNBlock%d" % (i + 1), block)
			in_channels = block.in_channels
			out_channels *= 2
			mid_channels *= 2

		# [c,2,2] : [c,1,1] [c] [10]
		self.features.add_module('norm4', nn.BatchNorm2d(in_channels))
		self.features.add_module('avgpool4', nn.AdaptiveAvgPool2d(1))
		self.classifier = nn.Linear(in_channels, 10)

		self.net_setting()

	def net_setting(self):
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
		self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[4, 7])

	def forward(self, x):
		features = self.features(x)
		out = features.view(features.size(0), -1)
		out = self.classifier(out)
		return out

	def load(self, batch_size: int):
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		trainset = torchvision.datasets.CIFAR10('./data', download=False, train=True, transform=transform)
		testset = torchvision.datasets.CIFAR10('./data', download=False, train=False, transform=transform)

		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

		return list(trainloader), list(testloader)
