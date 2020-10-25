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

import random
import time
import torch
import torch.nn as nn
from .Plot import Plot
from .Monitor import Monitor


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
	def training_model(self, epoch_num: int, record_num: int, plot=False, timer=False, test=False, test_num=1000):

		self.epoch_num = epoch_num

		if self.criterion is None:
			raise Exception("Criterion Not Setting!")
		if self.optimizer is None:
			raise Exception("Criterion Not Setting!")
		if self.train_data is None:
			raise Exception("Criterion Not Setting!")

		print("start train:\n")

		self.train()

		self.monitor.iter = []
		self.monitor.loss = []
		self.monitor.acc = []

		self.running_loss = 0.0
		self.running_acc = 0.0

		start_time = time.time()

		for epoch in range(epoch_num):

			iter = 0

			# shuffle all of the dataset
			random.shuffle(self.train_data)

			for data in self.train_data:

				iter += 1
				inputs, labels = data
				inputs = inputs.to(self.device)
				labels = labels.to(self.device)

				self.optimizer.zero_grad()

				outputs = self.to(self.device)(inputs)
				loss = self.criterion(outputs, labels)
				loss.backward()
				self.optimizer.step()

				self.running_loss += loss.item()

				_, pred = outputs.max(1)
				acc = int(sum(pred == labels)) / len(labels)
				self.running_acc += acc

				if iter % record_num == 0:
					self.record_one_train_result(epoch, iter, record_num, plot, test, test_num)

			# record all not shown data in this epoch
			self.record_one_train_result(epoch, iter, iter % record_num, plot, test, test_num)

			if self.lr_scheduler is not None:
				self.lr_scheduler.step()

		if timer:
			print("finished training, using %d seconds\n\n" % (time.time() - start_time))
		else:
			print("finished training\n\n")

	# record data (including adding to Monitor and Plot) once in training
	def record_one_train_result(self, epoch: int, iter: int, record_num: int, plot=False, test=False, test_num=None):
		iter_num = epoch * len(self.train_data) + iter
		print("iter num : " + str(iter_num))
		print("training loss : " + str(self.running_loss / record_num))
		print("training accuracy : " + str(self.running_acc / record_num))

		self.monitor.iter.append(iter_num * self.batch_size)
		self.monitor.loss.append(self.running_loss / record_num)
		self.monitor.acc.append(self.running_acc / record_num)

		self.running_loss = 0.0
		self.running_acc = 0.0

		# test
		if test:
			running_loss_test, running_acc_test = self.testing_model(timer=False, num=test_num)
			self.monitor.acc_test.append(running_acc_test)
			self.monitor.loss_test.append(running_loss_test)
			print("testing loss : " + str(running_loss_test))
			print("testing accuracy : " + str(running_acc_test))

		# plot
		if plot:
			iter_data, loss_data, acc_data = self.monitor.select()
			if test:
				iter_data_test, loss_data_test, acc_data_test = self.monitor.select_test()
			else:
				iter_data_test, loss_data_test, acc_data_test = None, None, None
			plot_fig = Plot(['loss', 'acc'], iter=iter_data, loss=loss_data, acc=acc_data, loss_test=loss_data_test,
							acc_test=acc_data_test)
			plot_fig.show()

		print("\n")

	# testing model
	def testing_model(self, timer=False, num=None):

		if self.test_data is None:
			raise Exception("Criterion Not Setting!")

		running_loss = 0.0
		running_acc = 0.0

		self.eval()

		iter = 0

		start_time = time.time()

		random.shuffle(self.test_data)
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

			if num is not None and iter > num: break

		if timer: print("finished testing, using %d seconds\n\n" % (time.time() - start_time))

		return running_loss / iter, running_acc / iter

	# save trained model, with ".pth"
	# how to load :
	#
	# 	checkpoint = torch.load(path)
	# 	model.load_state_dict(checkpoint['model'])
	# 	optimizer.load_state_dict(checkpoint['optimizer'])
	# 	epoch = checkpoint(['epoch'])
	#
	def save_model(self, path: str):
		state = {'model': self.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': self.epoch_num}
		torch.save(state, path)  # 保存整个模型，体积比较大
