import torch
import torch.nn as nn

class NN_pricing(nn.Module):
    def __init__(self,hyperparas):
        '''
        hyperparas = {'input_dim':5,'hidden_dim':30,'hidden_nums':3,'output_dim':88,'block_layer_nums':3}
        
        
        '''

        super().__init__()
        self.input_dim = hyperparas['input_dim']
        self.hidden_dim = hyperparas['hidden_dim']
        self.hidden_nums = hyperparas['hidden_nums']
        self.output_dim = hyperparas['output_dim']

        self.layer_list = []
        self.layer_list.append(nn.Sequential(nn.Linear(self.input_dim,self.hidden_dim),nn.ELU() ) )

        for _ in range(self.hidden_dim-1):
            self.layer_list.append(nn.Sequential(nn.Linear(self.hidden_dim,self.hidden_dim),nn.ELU()))

        self.layer_list.append(nn.Linear(self.hidden_dim,self.output_dim))

        self.linear_stock = nn.Sequential(*self.layer_list)

    def forward(self,inputs):
        
        return self.linear_stock(inputs)
    

class ResNetBlock(nn.Module):
    def __init__(self, hyperparas):
        '''
        hyperparas = {'hidden_dim':64,'block_layer_nums':3}
        
        '''


        super(ResNetBlock, self).__init__()
        
        self.hidden_dim = hyperparas['hidden_dim']
        self.block_layer_nums =hyperparas['block_layer_nums']
        

        
        
        # Define layers for the function f (MLP)
        self.layers = nn.ModuleList()
        
        for _ in range(self.block_layer_nums - 1):  # -2 because we already added one layer and last layer is already defined
            self.layers.append(nn.Linear(self.hidden_dim,self.hidden_dim ))
        
        
        
        # Layer normalization
        self.layernorms = nn.ModuleList()
        for _ in range(self.block_layer_nums - 1):  # -1 because layer normalization is not applied to the last layer
            self.layernorms.append(nn.LayerNorm(self.hidden_dim))
        
    def forward(self, x):
        # Forward pass through the function f (MLP)
        out = x
        for i in range(self.block_layer_nums - 1):  # -1 because last layer is already applied outside the loop
            out = self.layers[i](out)
            out = self.layernorms[i](out)
            out = torch.relu(out)
        
        
        
        # Element-wise addition of input x and output of function f(x)
        out = x + out
        
        return out
    

class ResNN_pricing(nn.Module):
    def __init__(self,hyperparas):
        '''
        hyperparas = {'input_dim':5,'hidden_dim':64,'hidden_nums':10,'output_dim':88,'block_layer_nums':3}

        '''
        super().__init__()
        self.input_dim = hyperparas['input_dim']
        self.hidden_dim = hyperparas['hidden_dim']
        self.hidden_nums = hyperparas['hidden_nums']
        self.output_dim = hyperparas['output_dim']
        self.block_layer_nums = hyperparas['block_layer_nums']

        self.layer_list = []
        self.layer_list.append(nn.Sequential(nn.Linear(self.input_dim,self.hidden_dim),nn.ReLU() ) )

        for _ in range(self.hidden_nums-1):
            self.layer_list.append(ResNetBlock(hyperparas)
                                   )

        self.layer_list.append(nn.Linear(self.hidden_dim,self.output_dim))

        self.linear_stock = nn.Sequential(*self.layer_list)

    def forward(self,inputs):
        
        return self.linear_stock(inputs)



