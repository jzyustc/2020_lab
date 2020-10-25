import torch
from classifier.example.Res_Net18 import ResNet_18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device : ", device, '\n')

net = ResNet_18(128, device)

net.training_model(30, 100, plot=True, timer=True)

test_result = net.testing_model(timer=True)
print("training loss : " + str(test_result[0]))
print("training accuracy : " + str(test_result[1]))
