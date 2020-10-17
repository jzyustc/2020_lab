import torch
from classifier.example.ConvNet import ConvNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device : ", device, '\n')

net = ConvNet(8, device)

net.training_model(10, 1500, plot=True)

test_result = net.testing_model()
print("training loss : " + str(test_result[0]))
print("training accuracy : " + str(test_result[1]))
