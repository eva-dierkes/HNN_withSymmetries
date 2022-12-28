import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import Utils.torchUtils as torchUtils

class DenseNet(pl.LightningModule):
    def __init__(self,
                 input_dim, output_dim, 
                 hidden_dim_list = [200],
                 lossf=F.mse_loss,
                 activation = torch.tanh,
                 optimizer=torch.optim.Adam, optim_args=None,
                 x_scaler= None,
                 dropout=0,
                 scheduler_dict = {},
                 init_weights=False):
        
        super(DenseNet, self).__init__()   
        self.save_hyperparameters()
        self.lossf = lossf
        self.nonlinearity = activation
        self.optimizer = optimizer
        self.optim_args = optim_args   
        self.x_scaler = x_scaler 
        self.scheduler_dict = scheduler_dict

        self.dimq = int(input_dim/2)

        self.dropout = nn.Dropout(dropout)
        self.linears = nn.ModuleList([nn.Linear(input_dim,hidden_dim_list[0])])
        self.linears.extend([nn.Linear(hidden_dim_list[index],hidden_dim_list[index+1]) for index,_ in enumerate(hidden_dim_list[:-1]) ])
        self.linears.append(nn.Linear(hidden_dim_list[-1],output_dim,bias=None))

        if init_weights=='orthogonal':
            for l in self.linears:
                torch.nn.init.orthogonal_(l.weight)
                
        if self.nonlinearity == torch.sin:
            nn.init.uniform_(self.linears[0].weight, - 1 / input_dim, 
                                                       1 / input_dim)
            for l in self.linears[1:]:
                nn.init.uniform_(l.weight, - np.sqrt(6 / input_dim) / 30,
                                             np.sqrt(6 / input_dim) / 30)
        
    def forward(self,x,t=0):
        for l in self.linears[:-1]:
            x = self.nonlinearity(l(x))
            x = self.dropout(x)
        x = self.linears[-1](x)
        return x
    
    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward(retain_graph=True)
    
    def output(self,x):
        return self.forward(x)
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        loss = self.lossf(self, x, y,'train')
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x,y = val_batch
        loss = self.lossf(self, x, y,'val')
        return loss
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(),**self.optim_args)
        scheduler = torchUtils.create_scheduler(self.scheduler_dict, optimizer)

        return ([optimizer], scheduler)