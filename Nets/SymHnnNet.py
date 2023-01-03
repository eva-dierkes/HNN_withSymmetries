import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

import Utils.torchUtils as torchUtils

class SymHnnNet(pl.LightningModule):
    def __init__(self,
                 input_dim, 
                 output_dim=1, 
                 hidden_dim_list=[200],
                 lossf=None,
                 activation = torch.tanh,
                 optimizer=torch.optim.Adam, 
                 optim_args=None,
                 x_scaler= None,
                 dropout=0,
                 scheduler_dict = {},
                 init_weights=False,
                 symmetry_grid= None,
                 state_sampler=None,
                 symmetry_type=['Trans']):
        
        super(SymHnnNet, self).__init__()   
        self.save_hyperparameters()
        self.input_dim = input_dim
        if lossf != None:
            self.lossf = lossf
        
        self.nonlinearity = activation  
        self.optimizer = optimizer
        self.optim_args = optim_args 
        self.x_scaler = x_scaler 
        self.scheduler_dict = scheduler_dict
        self.symmetry_type = symmetry_type
        
        tmp = torch.eye(self.input_dim)
        self.register_buffer("M",torch.eye(self.input_dim))
        self.M[:int(self.input_dim/2),:] = -tmp[int(self.input_dim/2):,:]
        self.M[int(self.input_dim/2):,:] =  tmp[:int(self.input_dim/2),:]
        
        self.state_sampler = state_sampler
        self.dimq = int(self.input_dim/2)
        self.nbr_sym = 1#self.dimq
        if 'Trans' in symmetry_type:
            self.translation_factor = nn.Parameter( torch.Tensor( np.random.uniform(-1,1,(self.nbr_sym, self.dimq ))),requires_grad=True)  # nbr_sym x qdim
        else:
            self.translation_factor = nn.Parameter(torch.zeros( self.nbr_sym, self.dimq), requires_grad=False)  #Just zeors if no Translation is whished, not trainable
        if 'Rot' in symmetry_type:
            self.rotation_factor = nn.Parameter( torch.Tensor( np.random.uniform(-1,1,(self.nbr_sym, self.dimq, self.dimq ))),requires_grad=True)  # nbr_sym x qdim
        else:
            self.rotation_factor = nn.Parameter( torch.zeros( self.nbr_sym, self.dimq,self.dimq), requires_grad=False)         #Just zeors if no Rottion is whished, not trainable


        if symmetry_grid!=None:
            self.symmetry_grid = symmetry_grid.to('cuda')

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
    
    def lossf(self, _, x, y, status): 
        nbr_epoch_without_symloss = 100
        nbr_smoothing_epochs = 100

        y_hat = self.output(x)                            # (dH/dp, -dH/dq)
        loss_hamilton = torch.nn.MSELoss()(y_hat,y)

        if self.current_epoch > nbr_epoch_without_symloss and status!='after' and self.symmetry_type:
            if status=='train':
                samples = self.state_sampler(quantity=81)      
                samples = torch.Tensor(samples).type_as(x)
                samples.requires_grad = True
            else:
                samples = self.symmetry_grid
            # w_hat
            dH_symmetryGrid = self.output(samples)        #(dH/dp, -dH/dq)  
            l_sym = torch.zeros(self.nbr_sym).type_as(x)  
            for index_sym in range(self.nbr_sym):              
                v_hat = torch.stack([-(self.rotation_factor[index_sym,:,:]   @ samples[i,:self.dimq ] + self.translation_factor[index_sym,:]) @ -dH_symmetryGrid[i, self.dimq:] + \
                                      (self.rotation_factor[index_sym,:,:].T @ samples[i, self.dimq:]).T                                      @  dH_symmetryGrid[i,:self.dimq ] for i in range(samples.shape[0])]) 
                l_sym[index_sym] = torch.linalg.norm(v_hat,dim=0)
    
            sym_norm = torch.linalg.norm(self.translation_factor,dim=1) + torch.linalg.norm(self.rotation_factor,dim=(1,2),ord='fro')

            # Correct sym error by norm of symmetry
            l_sym /= sym_norm
            l_sym_norm = ((sym_norm)**2 - 1) **2
            
            loss_dotproduct_transl = 0
            loss_dotproduct_rot = 0
            for i in range(self.nbr_sym):
                for j in range(i):
                    loss_dotproduct_transl += (torch.dot(self.translation_factor[i,:],self.translation_factor[j,:]))**2
                    loss_dotproduct_rot += torch.trace(self.rotation_factor[i,:,:] @ self.rotation_factor[j,:,:].T)  # "orthogonal" if ⟨𝐴,𝐵⟩=tr(𝐴𝐵𝑇)=0
            
            loss_dotproduct = loss_dotproduct_rot + loss_dotproduct_transl
            loss_symmetry = torch.sum(l_sym)/self.nbr_sym 
            loss_sym_norm = torch.sum(l_sym_norm)/self.nbr_sym 

            smooth_sym_factor = self.get_sym_loss_factor(current_epoch=self.current_epoch,
                                                        start_increasing= nbr_epoch_without_symloss,
                                                        end_increasing  = nbr_epoch_without_symloss+nbr_smoothing_epochs)
        else:
            loss_symmetry = 0
            loss_sym_norm = 0
            loss_dotproduct = 0
            smooth_sym_factor = 0
            
        loss = loss_hamilton + smooth_sym_factor*0.5*(loss_symmetry + loss_sym_norm + loss_dotproduct)

        if status != 'after':
            self.log(f'{status}/smooth_sym_factor',smooth_sym_factor, prog_bar=False, on_epoch=True, on_step=False)
            self.log(f'{status}/loss',loss, prog_bar=False, on_epoch=True, on_step=False)
            self.log(f'{status}/hamilton_loss',loss_hamilton, prog_bar=False, on_epoch=True, on_step=False)
            if self.current_epoch > nbr_epoch_without_symloss and self.symmetry_type:
                self.log(f'{status}/l_sym',loss_symmetry, prog_bar=False, on_epoch=True, on_step=False)
                self.log(f'{status}/l_sym_norm',loss_sym_norm, prog_bar=False, on_epoch=True, on_step=False)
                self.log(f'{status}/loss_dotproduct',loss_dotproduct, prog_bar=False, on_epoch=True, on_step=False)
                self.log(f'{status}/loss_symmetry',loss_symmetry, prog_bar=False, on_epoch=True, on_step=False)
                for i in range(self.nbr_sym):
                    self.log(f'{status}/l_sym_norm{i}',l_sym_norm[i], prog_bar=False, on_epoch=True, on_step=False)
                    self.log(f'{status}/l_sym{i}',l_sym[i], prog_bar=False, on_epoch=True, on_step=False)
            if status == 'val':
                for i in range(self.nbr_sym):
                    for j in range(self.dimq):
                        self.log(f'w_{i}/translation factor[{i},{j}]',self.translation_factor[i,j],prog_bar=False, on_epoch=True, on_step=False)
                    self.log(f'w_{i}/||translation factor||^2',torch.norm(self.translation_factor,dim=1)[i]**2,prog_bar=False, on_epoch=True, on_step=False)
                    for rot_col in range(self.dimq):
                        for rot_row in range(self.dimq):
                            self.log(f'rotMatrix{i}/rot_factor[{rot_row},{rot_col}]',self.rotation_factor[i,rot_row,rot_col],prog_bar=False, on_epoch=True, on_step=False)
                    self.log(f'rotMatrix{i}/||rotMatrix||^2',torch.norm(self.rotation_factor[i,:,:])**2,prog_bar=False, on_epoch=True, on_step=False)
        return loss
        
    def forward(self,x):
        for l in self.linears[:-1]:
            x = self.nonlinearity(l(x))
            x = self.dropout(x)
        x = self.linears[-1](x)
        return x

    def output(self,x):
        x_copy = x.clone().detach().requires_grad_(True)
        F1 = self.forward(x_copy)
        dF1 = torch.autograd.grad(F1.sum(), x_copy, create_graph=True)[0]
        vector_field = dF1 @ self.M  
        return vector_field   #(dH/dp, -dH/dq)
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        loss = self.lossf(0, x, y,'train')
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        torch.set_grad_enabled(True)
        x,y = val_batch
        loss = self.lossf(0, x, y,'val')
        return loss
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(),**self.optim_args)
        scheduler = torchUtils.create_scheduler(self.scheduler_dict, optimizer)
        return ([optimizer], scheduler)
    
    def get_sym_loss_factor(self,current_epoch, start_increasing, end_increasing):
        if current_epoch <= start_increasing:
            factor = 0
        elif current_epoch > start_increasing and current_epoch < end_increasing:
            factor = (current_epoch-start_increasing)/(end_increasing-start_increasing)
        else:
            factor = 1
        return factor 