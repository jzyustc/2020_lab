"""
DenseNet-121 for CIFAR-10 classifier
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


class _DenseLayer(nn.Sequential):

	def __init__(self, in_features, growth_rate, bn_size, drop_rate):
		super(_DenseLayer, self).__init__()
		self.add_module('bn1', nn.BatchNorm2d(in_features))
		self.add_module('relu1', nn.ReLU(inplace=True))
		self.add_module('conv1', nn.Conv2d(in_features, bn_size * growth_rate, 1))

		self.add_module('bn2', nn.BatchNorm2d(bn_size * growth_rate))
		self.add_module('relu2', nn.ReLU(inplace=True))
		self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1))
		self.drop_rate = drop_rate

	def forward(self, x):
		new_features = super(_DenseLayer, self).forward(x)
		if self.drop_rate:
			new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
		x = torch.cat([x, new_features], 1)
		return x


class _DenseBlock(nn.Sequential):

	def __init__(self, in_features, growth_rate, bn_size, drop_rate, layer_num):
		super(_DenseBlock, self).__init__()
		for i in range(layer_num):
			layer = _DenseLayer(in_features + i * growth_rate, growth_rate, bn_size, drop_rate)
			self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

	def __init__(self, in_features, out_features):
		super(_Transition, self).__init__()
		self.add_module('norm', nn.BatchNorm2d(in_features))
		self.add_module('relu', nn.ReLU(inplace=True))
		self.add_module('conv', nn.Conv2d(in_features, out_features, 1))
		self.add_module('pool', nn.AvgPool2d(2, 2))


class DenseNet(BaseNet):

	def __init__(self, batch_size, device, growth_rate=32, blocks=(6, 12, 8), in_features=64, bn_size=4, drop_rate=0):
		super(DenseNet, self).__init__(batch_size, device)

		self.train_data, self.test_data = self.load(batch_size)

		# [3,32,32] : [64,32,32] [64,16,16]
		self.features = nn.Sequential()
		self.features.add_module("conv0", nn.Conv2d(3, in_features, kernel_size=3, padding=1))
		self.features.add_module("norm0", nn.BatchNorm2d(in_features))
		self.features.add_module("relu0", nn.ReLU(inplace=True))
		self.features.add_module("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

		# [64,16,16] : [256,8,8] [640,4,4] [896,4,4]
		feature_num = in_features
		for i, layer_num in enumerate(blocks):
			block = _DenseBlock(feature_num, growth_rate, bn_size, drop_rate, layer_num)
			self.features.add_module("denseblock%d" % (i + 1), block)
			feature_num += layer_num * growth_rate
			if (i != len(blocks) - 1):
				trans = _Transition(feature_num, feature_num // 2)
				self.features.add_module("transition%d" % (i + 1), trans)
				feature_num //= 2

		# [896,4,4] : [896,1,1] [896] [10]
		self.features.add_module('norm4', nn.BatchNorm2d(feature_num))
		self.features.add_module('relu4', nn.ReLU(inplace=True))
		self.features.add_module('avgpool4', nn.AdaptiveAvgPool2d(1))
		self.classifier = nn.Linear(feature_num, 10)

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
