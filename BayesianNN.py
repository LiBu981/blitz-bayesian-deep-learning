import torch
import torch.nn as nn
import torch.nn.functional as F

from blitz.modules import BayesianLinear


class BayesianNet(nn.Module):

    def __init__(self):
        super (BayesianNet, self).__init__()

        #layers
        self.blinear1 = BayesianLinear(784,100)
        self.blinear2 = BayesianLinear(100,50)
        self.blinear3 = BayesianLinear(50,10)

    
    def forward(self, x):
        x = self.blinear1(x)
        x = F.sigmoid(x)
        x = self.blinear2(x)
        x = F.sigmoid(x)
        x = self.blinear3(x)
        x = F.sigmoid(x)
        return(x)

