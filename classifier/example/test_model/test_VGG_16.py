import torch
from classifier.example.VGG_16 import VGG_16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device : ", device, '\n')

net = VGG_16(8, device)

net.training_model(10, 1500, plot=True, timer=True)

test_result = net.testing_model(timer=True)
print("training loss : " + str(test_result[0]))
print("training accuracy : " + str(test_result[1]))
