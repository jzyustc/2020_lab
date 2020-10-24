"""
PNASNet for CIFAR-10 classifier
"""
# Last Modified : 2020/10/24, by jzy_ustc

from classifier.BaseNet import BaseNet
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import torch.utils.data
import torch.optim as optim


class MaxPool(nn.Module):

	def __init__(self, kernel_size, stride=1, padding=1, zero_pad=False):
		super(MaxPool, self).__init__()
		self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0)) if zero_pad else None
		self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

	def forward(self, x):
		if self.zero_pad:
			x = self.zero_pad(x)
		x = self.pool(x)
		if self.zero_pad:
			x = x[:, :, 1:, 1:]
		return x


class SeparableConv2d(nn.Module):

	def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
		super(SeparableConv2d, self).__init__()
		self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, dw_kernel, dw_stride, dw_padding, bias=bias,
										  groups=in_channels)
		self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)

	def forward(self, x):
		x = self.depthwise_conv2d(x)
		x = self.pointwise_conv2d(x)
		return x


class BranchSeparable(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, stem_cell=False, zero_pad=False):
		super(BranchSeparable, self).__init__()
		padding = kernel_size // 2
		middle_channels = out_channels if stem_cell else in_channels
		self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0)) if zero_pad else None
		self.relu_1 = nn.ReLU()
		self.separable_1 = SeparableConv2d(in_channels, middle_channels, kernel_size, stride, padding)
		self.bn_sep_1 = nn.BatchNorm2d(middle_channels, eps=0.001)
		self.relu_2 = nn.ReLU()
		self.separable_2 = SeparableConv2d(middle_channels, out_channels, kernel_size, 1, padding)
		self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001)

	def forward(self, x):
		x = self.relu_1(x)
		if self.zero_pad:
			x = self.zero_pad(x)
		x = self.separable_1(x)
		if self.zero_pad:
			x = x[:, :, 1:, 1:].contiguous()
		x = self.bn_sep_1(x)
		x = self.bn_sep_2(self.separable_2(self.relu_2(x)))
		return x


class ReluConvBn(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1):
		super(ReluConvBn, self).__init__()
		self.relu = nn.ReLU()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False)
		self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

	def forward(self, x):
		x = self.bn(self.conv(self.relu(x)))
		return x


class FactorizedReduction(nn.Module):

	def __init__(self, in_channels, out_channels):
		super(FactorizedReduction, self).__init__()
		self.relu = nn.ReLU()
		self.path_1 = nn.Sequential(
			nn.AvgPool2d(1, 2, count_include_pad=False),
			nn.Conv2d(in_channels, out_channels // 2, 1, bias=False)
		)
		self.path_2_pad = nn.ZeroPad2d((0, 1, 0, 1))
		self.path_2 = nn.Sequential(
			nn.AvgPool2d(1, 2, count_include_pad=False),
			nn.Conv2d(in_channels, out_channels // 2, 1, bias=False)
		)
		self.final_path_bn = nn.BatchNorm2d(out_channels, eps=0.001)

	def forward(self, x):
		x = self.relu(x)

		x_path1 = self.path_1(x)

		x_path2 = self.path_2_pad(x)
		x_path2 = x_path2[:, :, 1:, 1:]
		x_path2 = self.path_2(x_path2)

		out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))

		return out


class CellBase(nn.Module):

	def cell_forward(self, x_left, x_right):

		# 4
		x_comb_iter_0_left = self.comb_iter_0_left(x_left)
		x_comb_iter_0_right = self.comb_iter_0_right(x_left)
		x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

		# 0
		x_comb_iter_1_left = self.comb_iter_1_left(x_right)
		x_comb_iter_1_right = self.comb_iter_1_right(x_right)
		x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

		# 1
		x_comb_iter_2_left = self.comb_iter_2_left(x_right)
		x_comb_iter_2_right = self.comb_iter_2_right(x_right)
		x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

		# 2
		x_comb_iter_3_left = self.comb_iter_3_left(x_comb_iter_2)
		x_comb_iter_3_right = self.comb_iter_3_right(x_right)
		x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

		# 3
		x_comb_iter_4_left = self.comb_iter_4_left(x_left)
		if self.comb_iter_4_right:
			x_comb_iter_4_right = self.comb_iter_4_right(x_right)
		else:
			x_comb_iter_4_right = x_right
		x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

		x_out = torch.cat(
			[x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
			 x_comb_iter_4], 1)
		return x_out


class CellStem0(CellBase):

	def __init__(self, in_channels_left, out_channels_left, in_channels_right,
				 out_channels_right):
		super(CellStem0, self).__init__()
		self.conv_1x1 = ReluConvBn(in_channels_right, out_channels_right, 1)

		self.comb_iter_0_left = BranchSeparable(in_channels_left, out_channels_left, 5, 2, stem_cell=True)
		self.comb_iter_0_right = nn.Sequential(
			MaxPool(3, stride=2),
			nn.Conv2d(in_channels_left, out_channels_left, 1, bias=False),
			nn.BatchNorm2d(out_channels_left, eps=0.001)
		)
		self.comb_iter_1_left = BranchSeparable(out_channels_right, out_channels_right, 7, stride=2)
		self.comb_iter_1_right = MaxPool(3, stride=2)
		self.comb_iter_2_left = BranchSeparable(out_channels_right, out_channels_right, 5, stride=2)
		self.comb_iter_2_right = BranchSeparable(out_channels_right, out_channels_right, 3, stride=2)
		self.comb_iter_3_left = BranchSeparable(out_channels_right, out_channels_right, 3)
		self.comb_iter_3_right = MaxPool(3, stride=2)
		self.comb_iter_4_left = BranchSeparable(in_channels_right, out_channels_right, 3, stride=2, stem_cell=True)
		self.comb_iter_4_right = ReluConvBn(out_channels_right, out_channels_right, 1, stride=2)

	def forward(self, x_left):
		x_right = self.conv_1x1(x_left)
		x_out = self.cell_forward(x_left, x_right)
		return x_out


class Cell(CellBase):

	def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right, is_reduction=False,
				 zero_pad=False, match_prev_layer_dimensions=False):
		super(Cell, self).__init__()

		stride = 2 if is_reduction else 1

		self.match_prev_layer_dimensions = match_prev_layer_dimensions
		if match_prev_layer_dimensions:
			self.conv_prev_1x1 = FactorizedReduction(in_channels_left, out_channels_left)
		else:
			self.conv_prev_1x1 = ReluConvBn(in_channels_left, out_channels_left, kernel_size=1)

		self.conv_1x1 = ReluConvBn(in_channels_right, out_channels_right, kernel_size=1)
		self.comb_iter_0_left = BranchSeparable(out_channels_left, out_channels_left, 5, stride=stride,
												zero_pad=zero_pad)
		self.comb_iter_0_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
		self.comb_iter_1_left = BranchSeparable(out_channels_right, out_channels_right, 7, stride=stride,
												zero_pad=zero_pad)
		self.comb_iter_1_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
		self.comb_iter_2_left = BranchSeparable(out_channels_right, out_channels_right, 5, stride=stride,
												zero_pad=zero_pad)
		self.comb_iter_2_right = BranchSeparable(out_channels_right, out_channels_right, 3, stride=stride,
												 zero_pad=zero_pad)
		self.comb_iter_3_left = BranchSeparable(out_channels_right, out_channels_right, 3)
		self.comb_iter_3_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
		self.comb_iter_4_left = BranchSeparable(out_channels_left, out_channels_left, 3, stride=stride,
												zero_pad=zero_pad)

		if is_reduction:
			self.comb_iter_4_right = ReluConvBn(out_channels_right, out_channels_right, 1, stride=stride)
		else:
			self.comb_iter_4_right = None

	def forward(self, x_left, x_right):
		x_left = self.conv_prev_1x1(x_left)
		x_right = self.conv_1x1(x_right)
		x_out = self.cell_forward(x_left, x_right)
		return x_out


class PNASNet5(BaseNet):

	def __init__(self, batch_size, device):
		super(PNASNet5, self).__init__(batch_size, device)

		self.train_data, self.test_data = self.load(batch_size)

		self.conv_0 = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(32, eps=0.001)
		)

		self.cell_stem_0 = CellStem0(in_channels_left=32, out_channels_left=18,
									 in_channels_right=32, out_channels_right=18)
		self.cell_stem_1 = Cell(in_channels_left=32, out_channels_left=36,
								in_channels_right=90, out_channels_right=36,
								match_prev_layer_dimensions=True, is_reduction=True)

		self.cell_0 = Cell(in_channels_left=90, out_channels_left=72,
						   in_channels_right=180, out_channels_right=72,
						   match_prev_layer_dimensions=True)
		self.cell_1 = Cell(in_channels_left=180, out_channels_left=72,
						   in_channels_right=360, out_channels_right=72)
		self.cell_2 = Cell(in_channels_left=360, out_channels_left=72,
						   in_channels_right=360, out_channels_right=72)

		self.cell_3 = Cell(in_channels_left=360, out_channels_left=144,
						   in_channels_right=360, out_channels_right=144,
						   is_reduction=True, zero_pad=True)

		self.cell_4 = Cell(in_channels_left=360, out_channels_left=144,
						   in_channels_right=720, out_channels_right=144,
						   match_prev_layer_dimensions=True)
		self.cell_5 = Cell(in_channels_left=720, out_channels_left=144,
						   in_channels_right=720, out_channels_right=144)

		self.cell_6 = Cell(in_channels_left=720, out_channels_left=288,
						   in_channels_right=720, out_channels_right=288,
						   is_reduction=True)

		self.cell_7 = Cell(in_channels_left=720, out_channels_left=288,
						   in_channels_right=1440, out_channels_right=288,
						   match_prev_layer_dimensions=True)
		self.cell_8 = Cell(in_channels_left=1440, out_channels_left=288,
							in_channels_right=1440, out_channels_right=288)

		self.relu = nn.ReLU()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.dropout = nn.Dropout(0.5)
		self.last_linear = nn.Linear(1440, 10)

		self.net_setting()

	def net_setting(self):
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
		self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[4, 7])

	def features(self, x):
		# [3,32,32]
		x_conv_0 = self.conv_0(x)
		# [32,32,32]
		x_stem_0 = self.cell_stem_0(x_conv_0)
		# [90,16,16]
		x_stem_1 = self.cell_stem_1(x_conv_0, x_stem_0)
		# [180,8,8]
		x_cell_0 = self.cell_0(x_stem_0, x_stem_1)
		# [360,8,8]
		x_cell_1 = self.cell_1(x_stem_1, x_cell_0)
		# [360,8,8]
		x_cell_2 = self.cell_2(x_cell_0, x_cell_1)
		# [360,8,8]
		x_cell_3 = self.cell_3(x_cell_1, x_cell_2)
		# [720,8,8]
		x_cell_4 = self.cell_4(x_cell_2, x_cell_3)
		# [720,4,4]
		x_cell_5 = self.cell_5(x_cell_3, x_cell_4)
		# [720,4,4]
		x_cell_6 = self.cell_6(x_cell_4, x_cell_5)
		# [1440,4,4]
		x_cell_7 = self.cell_7(x_cell_5, x_cell_6)
		# [1440,4,4]
		x_cell_8 = self.cell_8(x_cell_6, x_cell_7)
		# [1440,4,4]
		return x_cell_8

	def logits(self, features):
		x = self.relu(features)
		# [4320,2,2]
		x = self.avg_pool(x)
		# [4320,1,1]
		x = x.view(x.size(0), -1)
		x = self.dropout(x)
		x = self.last_linear(x)
		return x

	def forward(self, input):
		x = self.features(input)
		x = self.logits(x)
		return x

	def load(self, batch_size: int):
		transform = transforms.Compose(
			[transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		trainset = torchvision.datasets.CIFAR10('./data', download=False, train=True, transform=transform)
		testset = torchvision.datasets.CIFAR10('./data', download=False, train=False, transform=transform)

		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

		return list(trainloader), list(testloader)
