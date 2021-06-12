import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset import MyDateSet
from harnet import HARNet

def main():

    data_set = MyDateSet("./training", "./ground_truth")
    train_data = DataLoader(dataset=data_set, num_workers=2, batch_size=1, shuffle=True)

    cuda = torch.cuda.is_available()

    model = HARNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mse_loss = nn.MSELoss()
    adam = optim.Adam(model.parameters())
    
    for epoch in range(5):
        running_loss = 0.0
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
            running_loss += loss
            if i % 10 == 0:
                print("epoch %d/5, batch %d/300" %((epoch+1), (i+1)))
            if i % 100 == 99:  # print every 100
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
        print("epoch %d /5, Loss: %f" %((epoch+1), running_loss/len(data_set)))
        save_checkpoint(model, epoch)
    print('Finished Training')





def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("./checkpoint/"):
        os.makedirs("./checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))