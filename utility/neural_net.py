import  torch
import torch.nn as nn
from collections import OrderedDict

class Encoder(nn.Module): 
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.xavier_normal_(self.linear.weight.data, gain=1.0)
        nn.init.zeros_(self.linear.bias.data)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x


class Modified_DNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.encoder_1 = Encoder(layers[0], layers[1])
        self.encoder_2 = Encoder(layers[0], layers[1])

        self.depth = len(layers) 
        self.activation = nn.Tanh()
        self.layer_list = nn.ModuleList([])
        # self.bn = nn.BatchNorm1d(layers[1])
        self.sig = nn.Sigmoid()

        for i in range(self.depth-2):

            linear = nn.Linear(layers[i], layers[i+1])
            # nn.init.xavier_normal_(linear.weight.data, gain=1.0)
            # nn.init.zeros_(linear.bias.data)
            self.layer_list.append(linear)
            
        linear = nn.Linear(layers[self.depth-2], layers[self.depth-1])
        # nn.init.xavier_normal_(linear.weight.data, gain=1.0)
        # nn.init.zeros_(linear.bias.data)
        self.layer_list.append(linear)
    
    def forward(self, x):
        E1 = self.encoder_1(x)
        E2 = self.encoder_2(x)

        for i in range(self.depth-2):
            x = self.layer_list[i](x)
            x = self.activation(x)
            x = torch.multiply(x, E1) + torch.multiply(1-x, E2)
            
            # x = self.bn(x)
            
        out = self.layer_list[self.depth-2](x)
        # out = self.sig(out)
        return out

class DNN(nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.depth = len(layers) 
        self.activation = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(layers[1])
        self.layer_list = nn.ModuleList()
        # self.A = nn.Parameter(torch.ones(self.depth-1))
       
        
        for i in range(self.depth-2):

            linear = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_normal_(linear.weight.data, gain=1.0)
            nn.init.zeros_(linear.bias.data)

            self.layer_list.append(linear)
            
        
        linear = nn.Linear(layers[self.depth-2], layers[self.depth-1])
        nn.init.xavier_normal_(linear.weight.data, gain=1.0)
        nn.init.zeros_(linear.bias.data)

        self.layer_list.append(linear)
        
        # layerDict = OrderedDict(layer_list)
        # self.layers = nn.Sequential(layerDict)

    def forward(self, x):
        for i in range(self.depth-2):
            x = self.layer_list[i](x)
            # x = torch.mul(x, self.A[i])
            x = self.activation(x)
            # x = self.bn(x)

        out = self.layer_list[self.depth-2](x)
        # out = torch.mul(out, self.A[self.depth-2])
        out = self.sig(out)
        return out
