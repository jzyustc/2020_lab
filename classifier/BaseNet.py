"""
Base Net Class for Classifier Net

including initial, train and test process of Classifier

to use:
# self.train_data, self.test_data = (Tensor)...
# self.net = (Net Model)...
# self.criterion = (Loss Function)...
# self.optimizer = (Optimization)...

optional :
# self.lr_scheduler = (lr_scheduler)...

to show how to use, please look at './ConvNet.py'
"""
# Last Modified : 2020/10/16, by jzy_ustc

import torch.nn as nn
from .Plot import Plot
from .Monitor import Monitor
import time


class BaseNet(nn.Module):

	# initial
	# param1 : Net Class
	# param2 : batch_size
	# param3 : device
	def __init__(self, batch_size: int, device):
		super(BaseNet, self).__init__()

		self.device = device
		self.batch_size = batch_size
		self.monitor = Monitor()

		# the following part must be rewritten in the extended class
		self.net = None
		self.criterion = None
		self.optimizer = None
		self.lr_scheduler = None

		self.train_data, self.test_data = None, None

	# training model
	# param1: epoch_num
	# param2: record_num, iterate interval to record the loss and accuracy
	def training_model(self, epoch_num: int, record_num: int, plot=False):

		if len(self.train_data) % record_num != 0:
			raise Exception('record_num should be divisivble by number of train data : %d' % len(self.train_data))

		if self.criterion is None:
			raise Exception("Criterion Not Setting!")
		if self.optimizer is None:
			raise Exception("Criterion Not Setting!")
		if self.train_data is None:
			raise Exception("Criterion Not Setting!")

		print("start train:\n")

		self.monitor.iter = []
		self.monitor.loss = []
		self.monitor.acc = []

		running_loss = 0.0
		running_acc = 0.0

		for epoch in range(epoch_num):

			iter = 0
			for data in self.train_data:

				inputs, labels = data
				inputs = inputs.to(self.device)
				labels = labels.to(self.device)

				self.optimizer.zero_grad()

				outputs = self.to(self.device)(inputs)
				loss = self.criterion(outputs, labels)
				loss.backward()
				self.optimizer.step()

				running_loss += loss.item()

				_, pred = outputs.max(1)
				acc = int(sum(pred == labels)) / len(labels)
				running_acc += acc

				if iter % record_num == record_num - 1:
					iter_num = epoch * len(self.train_data) + iter
					print("iter num : " + str(iter_num))
					print("training loss : " + str(running_loss / record_num))
					print("training accuracy : " + str(running_acc / record_num), '\n')

					self.monitor.iter.append(iter_num * self.batch_size)
					self.monitor.loss.append(running_loss / record_num)
					self.monitor.acc.append(running_acc / record_num)

					running_loss = 0.0
					running_acc = 0.0

					# plot
					if plot:
						iter_data, loss_data, acc_data = self.monitor.select()
						plot_fig = Plot(['loss', 'acc'], iter=iter_data, loss=loss_data, acc=acc_data)
						plot_fig.show()

				iter += 1

			if self.lr_scheduler is not None:
				self.lr_scheduler.step()

		print("finished training\n\n")

	# testing model
	def testing_model(self):

		if self.test_data is None:
			raise Exception("Criterion Not Setting!")

		running_loss = 0.0
		running_acc = 0.0

		self.eval()

		iter = 0

		for data in self.test_data:
			inputs, labels = data
			inputs = inputs.to(self.device)
			labels = labels.to(self.device)

			outputs = self.to(self.device)(inputs)
			loss = self.criterion(outputs, labels)

			running_loss += loss.item()

			_, pred = outputs.max(1)
			acc = int(sum(pred == labels)) / len(labels)
			running_acc += acc
			iter += 1

		return running_loss / iter, running_acc / iter
