import torch.nn as nn
class stu_net(nn.Module):
    def __init__(self, input_nc=2, output_nc=1):
        super(stu_net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=0),
            );
        self.act1 = nn.Sequential(nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=0),
            );
        self.act2= nn.Sequential(nn.Tanh())
    def encoder(self, input):
        G11 = self.conv1(input)
        G11_1 = self.act1(G11)
        return [G11_1,G11]
    def decoder(self, f_en):
        G21 = self.conv2(f_en[0]);
        G21_1 = self.act2(G21)
        return [G21_1,G21]

