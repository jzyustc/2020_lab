from classifier.operation.load_data import *

path = "classifier/example/test_data"

train_data, test_data = load_data(path, 2, grayscale=True)

print(len(train_data))
for data in train_data:
	inputs, labels = data
	print(inputs.shape, labels.shape)
