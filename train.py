import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset import MyDateSet
from harnet import HARNet
import eval
import os
from utils import weights_init_normal

def main():

    data_set = MyDateSet("./data/6x6_256", "./data/3x3_256")
    train_data = DataLoader(dataset=data_set, num_workers=2, batch_size=1, shuffle=True)

    cuda = torch.cuda.is_available()

    model = HARNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.apply(weights_init_normal)
    mse_loss = nn.MSELoss()
    adam = optim.Adam(model.parameters())
    
    for epoch in range(100):
        for i, data in enumerate(train_data):
            x, y = data
            X, Y = Variable(x), Variable(y)
            # print(X.data.size())
            if cuda:
                X = X.cuda()
                Y = Y.cuda()
            out = model(X)

            adam.zero_grad()
            loss = mse_loss(out, Y)
            loss.backward()
            adam.step()

            if i % 10 == 0:
                print("epoch %d/100" %(epoch+1))

        eval(model)
        save_checkpoint(model, epoch)
    print('Finished Training')



def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("./checkpoint/"):
        os.makedirs("./checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))