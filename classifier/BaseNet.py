# Base Net Class for Classifier Net
import torch.nn as nn
from .plot import Plot


class BaseNet(nn.Module):

	# initial
	# param1 : Net Class
	# param2 : batch_size
	# param3 : device
	def __init__(self, batch_size: int, device, plot=False):
		super(BaseNet, self).__init__()

		self.device = device
		self.batch_size = batch_size
		self.plot = plot
		self.plot_fig = Plot()


		self.net = None
		self.criterion = None
		self.optimizer = None
		self.lr_scheduler = None

		self.train_data, self.test_data = None, None

	# training model
	# param1: epoch_num
	# param2: record_num, iterate interval to record the loss and accuracy
	def training_model(self, epoch_num: int, record_num: int):

		if self.criterion == None:
			print("Criterion Not Setting!")
			return
		if self.optimizer == None:
			print("Criterion Not Setting!")
			return
		if self.train_data == None:
			print("Criterion Not Setting!")
			return

		print("start train:\n")

		if self.plot:
			self.plot_fig.training_show()

		self.plot_fig.record_iter = []
		self.plot_fig.record_loss = []
		self.plot_fig.record_acc = []

		running_loss = 0.0
		running_acc = 0.0
		iter = 0

		for epoch in range(epoch_num):

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
				num_correct = (pred == labels).sum()
				acc = int(num_correct) / inputs.shape[0]
				running_acc += acc

				if iter % record_num == record_num - 1:
					iter_num = epoch * len(self.train_data) + iter
					print("iter num : " + str(iter_num * self.batch_size))
					print("training loss : " + str(running_loss / record_num))
					print("training accuracy : " + str(running_acc / record_num), '\n')

					self.plot_fig.record_iter.append(iter_num * self.batch_size)
					self.plot_fig.record_loss.append(running_loss / record_num)
					self.plot_fig.record_acc.append(running_acc / record_num)

					running_loss = 0.0
					running_acc = 0.0

					if self.plot:
						print("plot!")
						self.plot_fig.training_update()


				iter += 1

			if self.lr_scheduler is not None:
				self.lr_scheduler.step()

		print("finished training\n\n")

	# testing model
	def testing_model(self):

		if self.test_data == None:
			print("Criterion Not Setting!")
			return

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
			iter_correct = (pred == labels).sum()
			acc = int(iter_correct) / inputs.shape[0]
			running_acc += acc
			iter += 1

		return running_loss / iter, running_acc / iter
