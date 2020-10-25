import torch
from classifier.example.Res_Net18 import ResNet_18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device : ", device, '\n')

net = ResNet_18(64, device)

net.training_model(30, 1500, plot=True, timer=True, test=True, test_num=1000)

test_result = net.testing_model(timer=True)
print("testing loss : " + str(test_result[0]))
print("testing accuracy : " + str(test_result[1]))
