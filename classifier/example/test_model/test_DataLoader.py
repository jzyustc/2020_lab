from classifier.operation.DataLoader import DataLoader

path = "../test_data"

data_loader = DataLoader(path, 2, grayscale=True, droplast=True)

train_data = data_loader.load_train_data()
test_data = data_loader.load_test_data()

print(len(train_data))
for data in train_data:
	inputs, labels = data
	print(inputs.shape, labels.shape)
