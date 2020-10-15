# Plot the loss and accuracy
import matplotlib.pyplot as plt


class Plot:

	def __init__(self):
		self.fig = plt.figure()
		self.record_iter = []
		self.record_loss = []
		self.record_acc = []

		self.ax_loss = None
		self.ax_acc = None

		self.add_loss()
		self.add_acc()

	def training_show(self):
		plt.show()

	def training_update(self):
		plt.pause(0.001)

	# TODO :error!!!
	def add_loss(self):
		self.ax_loss = self.fig.add_subplot(1, 2, 1)  # Loss
		plt.plot(self.record_iter, self.record_loss, 'ko--')
		plt.title('Loss-iteration')
		# plt.xlim([0, self.record_iter[-1] + 1])
		# plt.ylim([0, max(self.record_loss)])

	def add_acc(self):
		self.ax_acc = self.fig.add_subplot(1, 2, 2)  # Accuracy
		plt.plot(self.record_iter, self.record_acc, 'ko--')
		plt.title('Accuracy-iteration')
		# plt.xlim([0, self.record_iter[-1] + 1])
		# plt.ylim([min(self.record_acc), max(self.record_acc)])
