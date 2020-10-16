"""
A class to record the data in training, such as loss and accuracy

Provide support for Plot, and output the data

"""


# Last Modified : 2020/10/16, jzy_ustc


class Monitor:

	def __init__(self):
		self.iter = []
		self.loss = []
		self.acc = []

	# TODO : select
	def select(self):
		return self.iter, self.loss, self.acc

	# TODO : output
	def output(self, output_file_path):
		with open(output_file_path, 'wb') as file:
			file.write('123')
			file.close()
