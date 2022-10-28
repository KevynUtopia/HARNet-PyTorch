import torch
import torch.nn as nn
import torch.nn.functional as F


class HARNet(nn.Module):
    def __init__(self):
        super(HARNet, self).__init__()
        self.conv_block = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        model1 = []
        for _ in range(19):
            model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)]
        self.conv_block1 = nn.Sequential(*model1)
        model2 = []
        for _ in range(19):
            model2 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)]
        self.conv_block2 = nn.Sequential(*model2)
        model3 = []
        for _ in range(19):
            model3 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)]
        self.conv_block3 = nn.Sequential(*model3)
        model4 = []
        for _ in range(19):
            model4 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)]
        self.conv_block4 = nn.Sequential(*model4)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(320, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(384, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()


    def forward(self, x):
        
        residual = x
        output = self.relu(self.conv1(x)) # (1, 128, 400, 400)

        block_residual = output # skip connection
        output = self.relu(self.conv2(output)) # (1, 64, 400, 400)
        # for _ in range(19):
        #   output = self.relu(self.conv_block(output)) # (1, 64, 400, 400)
        output = self.relu(self.conv_block1(output)) 
        output = torch.cat((output, block_residual), 1) # (1, 192, 400, 400)

        block_residual = output
        output = self.relu(self.conv3(output)) # (1, 64, 400, 400)
        # for _ in range(19):
        #   output = self.relu(self.conv_block(output)) # (1, 64, 400, 400)
        output = self.relu(self.conv_block1(output)) 
        output = torch.cat((output, block_residual), 1) # (1, 256, 400, 400)

        block_residual = output
        output = self.relu(self.conv4(output)) # (1, 64, 400, 400)
        # for _ in range(19):
        #   output = self.relu(self.conv_block(output)) # (1, 64, 400, 400)
        output = self.relu(self.conv_block1(output)) 
        output = torch.cat((output, block_residual), 1) # (1, 320, 400, 400)

        block_residual = output
        output = self.relu(self.conv5(output)) # (1, 64, 400, 400)
        # for _ in range(19):
        #   output = self.relu(self.conv_block(output)) # (1, 64, 400, 400)
        output = self.relu(self.conv_block1(output)) 
        output = torch.cat((output, block_residual), 1) # (1, 384, 400, 400)

        output = self.conv6(output)

        output += residual
        
        return output
