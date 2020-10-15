from classifier.example import *
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device : ", device, '\n')

net = Net(8, device, plot=True)

net.training_model(10, 1000)

print(len(net.plot_fig.record_acc))

test_result = net.testing_model()
print("training loss : " + str(test_result[0]))
print("training accuracy : " + str(test_result[1]))
