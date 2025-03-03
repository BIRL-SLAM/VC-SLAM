import sys
from ChebyKANLayer import ChebyKANLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CAPUDFNetwork1(nn.Module):
    def __init__(self,
                 scale=1
                ):
        super(CAPUDFNetwork1, self).__init__()
        self.chebykan1 = ChebyKANLayer(3, 256, 40)
        self.ln1 = nn.LayerNorm(256) # To avoid gradient vanishing caused by tanh
        self.chebykan2 = ChebyKANLayer(256,64, 5)
        self.ln2 = nn.LayerNorm(64)
        self.chebykan3 = ChebyKANLayer(64,16, 4)
        self.ln3 = nn.LayerNorm(16) # To avoid gradient vanishing caused by tanh
        self.chebykan4 = ChebyKANLayer(16, 4, 3)
        self.ln4 = nn.LayerNorm(4)
        self.model =  ChebyKANLayer(4, 1, 2)
        self.scale = scale
        self.device = torch.device("cpu")
        self.model.to(self.device)
        
        # Define optimizer
        #optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        #def get_model_parameters(self):
            #self.parameters = self.model.parameters()
            #return self.parameters
    def model_train(self):
        return self.model.train()
        
    def forward(self, inputs):
        input_x = inputs * self.scale
        input_x = input_x.to(self.device)
        input_x = self.chebykan1(input_x)
        input_x = self.ln1(input_x)
        input_x = self.chebykan2(input_x)
        input_x = self.ln2(input_x)
        input_x = self.chebykan3(input_x)
        input_x = self.ln3(input_x)
        input_x = self.chebykan4(input_x)
        input_x = self.ln4(input_x)
        output = self.model(input_x)
        ##输出除以尺度因子 self.scale 并返回结果
        res = torch.abs(output)
        # res = 1 - torch.exp(-x)
        return res / self.scale

    def udf(self, x):
        return self.forward(x)

    def udf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.udf(x)
        # y.requires_grad_(True)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)
       



