import torch
from operator import itemgetter
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
from torchsummary import summary
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Nets import DenseNet, HnnNet, SymHnnNet
import Utils.dataUtils as dataUtils
import Utils.torchUtils as torchUtils
import Utils.lossUtils as lossUtils

class EvalModel():
    def __init__(self,model, requires_grad=False):
        self.model = model.eval()
        self.x_scaler = model.x_scaler
        self.scaling_factor = 1.0
        if self.x_scaler:
            if hasattr(self.x_scaler,'data_range_'):
                self.scaling_factor = self.x_scaler.data_range_
            elif hasattr(self.x_scaler, 'scale_'):
                self.scaling_factor = self.x_scaler.scale_
        self.requires_grad = requires_grad
        
    def identified_ode(self,t,x):
        x = x.reshape(1,-1)
        if self.x_scaler!=None:
            x = self.x_scaler.transform(x)
        x = torch.tensor(x,dtype=torch.float)
        dx = self.model.output(x).detach().numpy()
        if self.x_scaler!=None:
            dx *= self.scaling_factor
        if dx.shape[0] ==1:
            dx = dx.flatten()
        return dx

            
def create_callbacks(model_file_name,model_file_path,train_args,save_dir,verbose=0):
    #callbacks 
    cb_checkpoint = cb.ModelCheckpoint(monitor='val/hamilton_loss',
                                       dirpath=save_dir+'Models',
                                       filename=model_file_name)
    cb_checkpoint_last = cb.ModelCheckpoint(monitor=None,
                                       dirpath=save_dir+'Models',
                                       filename=model_file_name+'_last')
    cb_earlystopping = cb.EarlyStopping(monitor='val/hamilton_loss',
                                        patience=train_args['early_stop'],
                                        verbose=verbose)
    tb_logger = pl.loggers.TensorBoardLogger(save_dir+'TensorBoard',
                                             name=model_file_name)
    lr_monitor = cb.LearningRateMonitor()
    
    return [cb_checkpoint, cb_checkpoint_last, cb_earlystopping, lr_monitor], tb_logger


def create_scheduler(scheduler_dict, optimizer):
    scheduler_list= []
    for key in scheduler_dict:
        scheduler_tag = key.split('_')[0]
        if scheduler_tag=='stepLR' and scheduler_dict[key] != None: 
            scheduler_list.append(torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_dict[key]))
        elif scheduler_tag=='multistepLR' and scheduler_dict[key] != None:
            scheduler_list.append(torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_dict[key]))
        elif scheduler_tag=='reduceLRonPlateau' and scheduler_dict[key] != None:
            scheduler_list.append(
                {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_dict[key]), 
                'monitor': 'val/loss'}
                )
    return scheduler_list

def train(data,exampleClass,data_config,net_config):
    # Extract data
    X_train,X_val, X_test, Y_train, Y_val, Y_test = itemgetter('X_train','X_val','X_test','Y_train','Y_val','Y_test')(data)
  
    pl.seed_everything(net_config['seed'])

    #scale if wanted in config
    if net_config['train_args']['normalized_inputdata']==1:
        x_scaler = MinMaxScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_val = x_scaler.transform(X_val)
        X_test = x_scaler.transform(X_test)

        Y_train /= x_scaler.data_range_
        Y_val /= x_scaler.data_range_
        Y_test /= x_scaler.data_range_

    elif net_config['train_args']['normalized_inputdata']==2:
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_val = x_scaler.transform(X_val)
        X_test = x_scaler.transform(X_test)
        
        Y_train /= x_scaler.scale_
        Y_val /= x_scaler.scale_
        X_test /= x_scaler.scale_
    else:
        x_scaler = None
        

    # Prepare Data
    X_train = torch.tensor(X_train, dtype=torch.float, requires_grad=False)
    X_val = torch.tensor(X_val, dtype=torch.float, requires_grad=False)
    X_test = torch.tensor(X_test, dtype=torch.float, requires_grad=False)
    Y_train = torch.tensor(Y_train, dtype=torch.float, requires_grad=False)  
    Y_val = torch.tensor(Y_val, dtype=torch.float, requires_grad=False)   
    Y_test = torch.tensor(Y_test, dtype=torch.float, requires_grad=False)  

    #%% Train and Save 
    pl.seed_everything(net_config['seed'])
    train_data, val_data = dataUtils.wrap_data_into_dataloader(X_train=X_train, 
                                                                Y_train=Y_train, 
                                                                X_val=X_val, 
                                                                Y_val=Y_val, 
                                                                batch_size=net_config['train_args']['batch_size'], 
                                                                num_workers=30, 
                                                                size_multiplier=net_config['train_args']['size_multiplier'])

    model, trainer, \
        [cb_earlystopping, \
        cb_checkpoint], \
            tb_logger = torchUtils.setup_and_train_model(model_type=net_config['net'], 
                                                        data_type=data_config['example'],
                                                        exampleClass=exampleClass, 
                                                        net_config=net_config,
                                                        train_data=train_data, 
                                                        val_data=val_data,
                                                        train_args=net_config['train_args'],
                                                        x_scaler = x_scaler,
                                                        save_dir=data_config['save_dir'])

    
    train_loss = model.lossf(model,X_train,Y_train,'after').detach().numpy()
    val_loss = model.lossf(model,X_val,Y_val,'after').detach().numpy()
    test_loss = model.lossf(model,X_test,Y_test,'after').detach().numpy()
    print(' - Loss: %7.6f'%(train_loss), end='')
    print(' - Val_loss: %7.6f'%(val_loss), end='')
    print(' - Test_loss: %7.6f'%(test_loss))

    dataUtils.write_run_to_csv(example_class=data_config["example"],
                            net_architecture=f'{net_config["net"]}_{net_config["loss_tag"]}_{"".join(net_config["symmetry_type"]) if "symmetry_type" in net_config else ""}',
                            save_dir=data_config['save_dir'],
                            train_args=net_config['train_args'],
                            ode_args=data_config['ode_args'],
                            data_args=data_config['data_args'],
                            train_loss=train_loss,val_loss=val_loss,test_loss=test_loss,
                            stopped_epoch=trainer.current_epoch+1-cb_earlystopping.wait_count)


def setup_and_train_model(model_type,data_type,exampleClass,net_config,train_data,val_data,train_args,x_scaler,save_dir):

    # Define Optimizer with params from net_config train_args
    optimizer = torch.optim.Adam
    optim_args = {'lr':train_args['lr'],'weight_decay':train_args['weight_decay']}

    # Initilize Model 
    model_type = net_config['net']
    if model_type=='NN':
        output_dim = train_data.dataset.X.shape[1]
        model = DenseNet.DenseNet
    elif model_type=='HNN':
        output_dim = 1
        model = HnnNet.HnnNet
    elif model_type=='SymHnn':
        output_dim = 1
        model = lambda input_dim, output_dim, hidden_dim_list=[200], lossf=torch.nn.functional.mse_loss, activation = torch.tanh, optimizer=torch.optim.Adam, optim_args=None, \
                 x_scaler= None, dropout=0, scheduler_dict = {}, init_weights=False, symmetry_grid=exampleClass.symmGrid, states_sampler=exampleClass.generate_random_inital_value: \
                    SymHnnNet.SymHnnNet(input_dim, output_dim, hidden_dim_list, lossf, activation, optimizer, optim_args,\
                         x_scaler, dropout, scheduler_dict, init_weights, symmetry_grid, states_sampler, symmetry_type=net_config['symmetry_type'])
        model_type += '_' +''.join(net_config['symmetry_type'])

    # Select lossfunction if defined in net config
    if net_config['loss_tag']=='mse':
        lossf = lambda model,x,y,status: lossUtils.my_mse_loss(model,x,y,status)[0]
        model_type += '_mse'
    else:
        lossf = None

    print(f'Train {model_type} model')
    # Update model with hyperparams
    model = model(input_dim=train_data.dataset.X.shape[-1], 
                    output_dim=output_dim,
                    hidden_dim_list=train_args['hidden'],
                    lossf=lossf, 
                    optimizer=optimizer,
                    activation=train_args['activation'],
                    optim_args=optim_args,
                    x_scaler=x_scaler,
                    dropout=train_args['dropout'],
                    scheduler_dict=train_args['scheduler_dict'],
                    init_weights=train_args['init_weights'])
    
    summary(model, (train_data.dataset.X.shape[-1],), device=model.device.type )
    model_file_suffix = dataUtils.create_model_file_name(train_args)
    model_file_name = 'model_'+data_type+'_'+model_type+model_file_suffix
    model_file_path = save_dir+model_file_name
    
    # Get callbacks 
    callbacks, tb_logger = torchUtils.create_callbacks(model_file_name,
                                                         model_file_path,
                                                         train_args,
                                                         save_dir)

    trainer = pl.Trainer(accelerator='gpu' if train_args['gpus']!=0 else 'cpu',
                        devices=train_args['gpus'] if net_config['loss_tag']=='' else 1,        # if lossfunction is not part of the model it can not be paralized on multiple gpu
                        callbacks=callbacks,
                        logger=tb_logger,
                        max_epochs=train_args['epochs'],
                        enable_progress_bar=False,
                        deterministic=True,
                        auto_select_gpus=True,
                        enable_model_summary=False,
                        log_every_n_steps=1
                        )
    
    trainer.fit(model, train_data, val_data)
    model = model.load_from_checkpoint(callbacks[0].best_model_path)

    # Set lossf for the loaded model (as customized lossf can not be saved in the model)
    if lossf != None:
        model.lossf = lossf
    
    return model, trainer, [callbacks[2],callbacks[0]], tb_logger

