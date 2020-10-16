"""
A class to plot the monitor result in training

support monitor data type :

- Loss - iteration
- Accuracy - iteration
- to be continued ...

to use:

# plot_fig = Plot(['loss', 'acc'], iter=iter_data, loss=loss_data, acc=acc_data)
# plot_fig.show()

"""
# Last Modifined : 2020/10/16 jzy_ustc

import matplotlib.pyplot as plt


class Plot:

	def __init__(self, list_subplot: [str], iter=None, acc=None, loss=None):

		# figure
		self.fig = plt.figure()

		# data load
		self.iter = iter
		self.acc = acc
		self.loss = loss

		# subplots list load
		self.list_subplots = list_subplot

		# subplots list define
		self.subplots_title = {'acc': 'Accuracy-iteration', 'loss': 'Loss-iteration'}
		self.subplots_data = {'acc': [self.iter, self.acc], 'loss': [self.iter, self.loss]}

		# list of subplots
		self.input_check()
		self.length = len(self.list_subplots)

		# group the subplots to the figure
		self.x_size = max(4, int(self.length ** 0.5) + 1)
		self.x_num = int(self.length / self.x_size) + 1
		self.y_num = min(self.length, self.x_size)

		# add subplots
		for i in range(self.length):
			subplot = self.list_subplots[i]
			title = self.subplots_title[subplot]
			x = self.subplots_data[subplot][0]
			y = self.subplots_data[subplot][1]
			self.add_subplot(title, x, y, i + 1)

	# from the input, choose the sublplot list to plot
	def input_check(self):
		for subplot in self.list_subplots:

			# check whether input right names
			if subplot not in self.subplots_title:
				raise Exception('Undefined subplot name : ' + subplot + " !")

			# check whether input the data
			if self.subplots_title[subplot][0] is None or self.subplots_title[subplot][1] is None:
				raise Exception('Input data for ' + subplot + " is wrong!")

	# add a subplot for figure
	def add_subplot(self, title: str, x, y, i: int):
		self.fig.add_subplot(self.x_num, self.y_num, i)
		plt.plot(x, y, 'ko--')
		plt.title(title)
		plt.xlim([0, x[-1] + 1])
		plt.ylim([0, max(y)])

	# show figure
	def show(self):
		plt.show()
