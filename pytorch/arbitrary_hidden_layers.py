import torch
import torch.nn as nn

# 实现任意层神经网络的定义

class MLP(nn.Module):
    def __init__(self, h_sizes, out_size) -> None:
        super(MLP, self).__init__()

        # hidden layers
        self.hidden = []
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.add_module('hidden_layer_{}'.format(k), self.hidden[k])
        
        self.out = nn.Linear(h_sizes[-1], out_size)



# 用modulelist实现任意层神经网络的定义

class MLP(nn.Module):
    def __init__(self, h_sizes, out_size) -> None:
        super(MLP, self).__init__()

        # hidden layers
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
        
        self.out = nn.Linear(h_sizes[-1], out_size)

    def forward(self, x):
        for layer in self.hidden:
            x = layer(x)
        x = self.out(x)
        return x