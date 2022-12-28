import numpy as np
import pandas as pd
import os
from datetime import datetime
import torch
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import Utils.odeUtils as odeUtils


class DataGenerator:
    def __init__(self,example_class,trajectory_args):
        self.example_class = example_class
        self.trajectory_args = trajectory_args
        self.t_span = np.linspace(self.trajectory_args['t_start'],self.trajectory_args['t_end'], self.trajectory_args['nbr_points_per_traj'])
        self.noise_strength = trajectory_args['noise']
        if 'region_tag' in trajectory_args.keys():
            self.region_tag = trajectory_args['region_tag']
        else:
            self.region_tag = None
        
    def generate_single_trajectory(self,x0):
        if callable(getattr(self.example_class,'generate_single_trajectory',None)):
            solution = self.example_class.generate_single_trajectory(x0,self.trajectory_args)
        else:
            solver_output = solve_ivp(self.example_class.ode,
                            [self.trajectory_args['t_start'],self.trajectory_args['t_end']],
                            x0.flatten(),
                            t_eval=self.t_span,
                            rtol=1e-10,atol=1e-10) # integrate
            solution = solver_output.y.T
        if solver_output.success == False:
            solution = np.full([self.t_span.shape[0],self.example_class.nbr_states], np.nan)

        return solution
        
    def caluclate_derivative_for_data(self,x):
        dx = np.zeros(x.shape)
        for i in range(0,len(x)):
            dx[i,:] = self.example_class.ode(0,x[i,:])
        return dx
    
    def add_noise_to_data(self,x,dx):
        x = x + self.noise_strength*np.random.randn(x.shape[0], x.shape[1])
        dx = dx + self.noise_strength*np.random.randn(dx.shape[0], dx.shape[1])
        return x,dx
        
    def generate_data_set(self,nbr_of_traj, noNoise=False):
        x=[]
        all_random_initial_values = self.example_class.generate_random_inital_value(quantity=nbr_of_traj, region_tag=self.region_tag)
        for x0 in all_random_initial_values:
            if len(self.t_span)>1:
                single_traj = self.generate_single_trajectory(x0)
            else:
                single_traj = np.expand_dims(x0,axis=0)
            x.append(single_traj)
        test_trajectories = [np.concatenate([traj,self.caluclate_derivative_for_data(traj)],axis=-1) for traj in x]   # save Test trajectories for later plots
        x = np.concatenate(x)
        dx = self.caluclate_derivative_for_data(x)
        if not noNoise:
            x,dx = self.add_noise_to_data(x, dx)
        
        return x,dx,test_trajectories
       

class WrapDataset(Dataset):
    def __init__(self,X,Y, size_multiplier = 1):
        self.X = X
        self.Y = Y
        self.size_multiplier = size_multiplier
        
    def __len__(self):
        return self.size_multiplier*len(self.X)
    
    def __getitem__(self,idx):
        idx = idx % len(self.X)
        return self.X[idx], self.Y[idx]
    

def prepare_training_data(data_config):

    if data_config['example']=='PendCart':
        exampleClass = odeUtils.Pendulum_on_a_cart(data_config['ode_args'])
    elif data_config['example']=='Kepler_cartesian':
        exampleClass = odeUtils.Kepler_cartesian(data_config['ode_args'])

    data_config['train_data_file_path'] = create_train_data_file_path(data_config['example'],
                                                            data_config['ode_args'],
                                                            data_config['data_args'])

    zero_noise_data_file_path = ''.join([data_config['save_dir'],
                                        data_config['train_data_file_path'].split('noise')[0],
                                        'noise0seed',
                                        data_config['train_data_file_path'].split('noise')[1].split('seed')[-1]])
    zero_noise_data_file_name =  zero_noise_data_file_path + '/data.npy'
    
    data_config['save_dir'] = data_config['save_dir']+data_config['train_data_file_path']+'/'
    data_file_name = data_config['save_dir']+'data.npy'
    

    if os.path.exists(data_file_name):
        data_dict = np.load(data_file_name,allow_pickle=True).item()
        x = data_dict['x']
        dx = data_dict['dx']
        all_train_traj = data_dict['traj']
        print('Load data succesfull')
    elif os.path.exists(zero_noise_data_file_name) and data_config['data_args']['noise'] != 0:
        data_dict = np.load(zero_noise_data_file_name,allow_pickle=True).item()
        x = data_dict['x']
        dx = data_dict['dx']
        all_train_traj = data_dict['traj']
        data_generator = DataGenerator(exampleClass, trajectory_args=data_config['data_args'])
        x,dx = data_generator.add_noise_to_data(x,dx)
        print('Zero Noise dataset found. Noise added.')
        if not os.path.exists(data_config['save_dir']):
                os.makedirs(data_config['save_dir'])
        np.save(data_file_name,{'x':x,'dx':dx,'traj':all_train_traj})
    else:
        print('Load data failed. Generate and save new zero noise dataset')
        data_generator = DataGenerator(exampleClass,
                                        trajectory_args=data_config['data_args'])
        
        x, dx, all_train_traj = data_generator.generate_data_set(nbr_of_traj=data_config['data_args']['nbr_of_traj'],noNoise=True)
        if not os.path.exists(zero_noise_data_file_path):
            os.makedirs(zero_noise_data_file_path)
        np.save(zero_noise_data_file_name,{'x':x,'dx':dx,'traj':all_train_traj})
        #Save noisy measurements
        if data_config['data_args']['noise'] != 0:
            if not os.path.exists(data_config['save_dir']):
                os.makedirs(data_config['save_dir'])
            x,dx = data_generator.add_noise_to_data(x,dx)
            np.save(data_file_name,{'x':x,'dx':dx,'traj':all_train_traj})
            print('Add Noise to zero noise dataset and save noisy dataset')        

    if data_config['example'] == 'PendCart_rotated':
        data_config['example'] = ''.join([data_config['example'],str(data_config['ode_args']['rotation_angle'] if 'rotation_angle' in data_config['ode_args'] else '')])
    return x,dx, exampleClass, all_train_traj
    

def create_train_data_file_path(Example_type, example_args, train_traj_args):
    data_name = ''.join(Example_type)
    if Example_type == 'PendCart_rotated':
        data_name = ''.join([data_name,str(example_args['rotation_angle'] if 'rotation_angle' in example_args else '')])
    
    ode_tag = '_'
    for key in example_args:
        value = example_args[key]
        ode_tag = ''.join([ode_tag,key,str(round(value,3))])
        ode_tag = ode_tag.replace('.','-')
    
    train_tag = '_'

    tags = {}
    for key in train_traj_args:
        tags[key] = key
    #renaming key to be shorter!
    tags.update({'t_start':'ts','t_end':'te','nbr_of_traj':'ntraj',
            'nbr_points_per_traj':'nbrt','noise':'_noise',
            'train_data_type':'','region_spread':'',
            'normalized_data':'normalized'})
    
    for key in train_traj_args:
        value = train_traj_args[key]
        train_tag = ''.join([train_tag,tags[key],str(round(value,5))])
    train_tag = train_tag.replace('.','-')
        
    data_name = ''.join([data_name, 
                         ode_tag,train_tag,])
    return data_name


def create_model_file_name(train_args):
    name = ''
    for key in train_args:
        value = train_args[key]
        if key=='activation':
            if value==torch.tanh:
                name = ''.join([name,'-tanh'])
            elif value==torch.sin:
                name = ''.join([name,'-sin'])
            elif type(value)==type(torch.nn.Softplus()):
                name = ''.join([name,'-softplus'])
            continue
        elif key=='hidden':
            name = ''.join([name,'-'.join([str(dim) for dim in train_args['hidden']]),key])
            continue
        elif key=='scheduler_dict':
            for scheduler_key in value.keys():
                scheduler_tag = scheduler_key.split('_')[0]
                if scheduler_tag=='stepLR':
                    if value[scheduler_key]['gamma']!= 1.0:
                        name = ''.join([name,f'-stepLR{value[scheduler_key]["gamma"]}gamma{value[scheduler_key]["step_size"]}step'.replace('.','_')])
                    continue
                elif scheduler_tag=='multistepLR':
                    if value[scheduler_key]['gamma']!= 1.0:
                        steps = "-".join([str(ms) for ms in value[scheduler_key]["milestones"]])
                        name = ''.join([name,f'-multistepLR{value[scheduler_key]["gamma"]}gamma{steps}step'.replace('.','_')])
                    continue
                elif scheduler_tag=='reduceLRonPlateau':
                    if value[scheduler_key]['factor']!= 1.0:
                        name = ''.join([name,f'-LRonPlatau'])#{value["factor"]}factor{value["patience"]}patience{value["min_lr"]}minLR'.replace('.','_')])
                        for k in value[scheduler_key].keys():
                            name = ''.join([name,f'{value[scheduler_key][k]}{k}'.replace('.','_')])
                    continue
            continue   
        elif key=='init_weights':
            if value is not None:
                name = ''.join([name,f'-init_{value}'])
            continue
            
        if value <1: value = str(value).replace('.','_')
        else: value = str(value)
        if value != '0':
            name = ''.join([name,'-',value,key])
        
    return name
    

def write_run_to_csv(example_class,net_architecture,save_dir,train_args,ode_args,data_args,
                           train_loss,val_loss,test_loss=0,stopped_epoch=0):
    if example_class=='Pend':
        csv_path = save_dir+'../Pend_trainings.csv'
    elif example_class=='PendCart':
        csv_path = save_dir+'../PendCart_trainings.csv'
    elif example_class=='PendCart_rotated':
        csv_path = save_dir+'../PendCart_rotated_trainings.csv'
    elif example_class=='Kepler':
        csv_path = save_dir+'../Kepler_trainings.csv'
    elif example_class=='Kepler_cartesian':
        csv_path = save_dir+'../Kepler_cartesian_trainings.csv'
    try:
        csv = pd.read_csv(csv_path,index_col=0)
    except:
        print('Could not open existing csv file')
        csv = pd.DataFrame()
    
    additional_data = {'date': datetime.now(),
                       'train_loss': train_loss,
                       'val_loss':val_loss,
                       'test_loss':test_loss,
                       'example_class': example_class,
                       'network': net_architecture,
                       'stopped_epoch': stopped_epoch
                       }
    
    all_args = data_args.copy()
    all_args.update(additional_data)
    all_args.update(train_args)
    all_args.update(ode_args)

    csv = csv.append(all_args, ignore_index=True)
    csv.to_csv(csv_path,decimal='.',sep=',')
    
    
def train_val_split(X,Y,val_size,save_dir):
    split_data_file_name = save_dir+f'/data_{val_size}_split'.replace('.','-')+'.npy'

    # Load existing splitted dataset if exists
    if os.path.exists(split_data_file_name):
        data_dict = np.load(split_data_file_name,allow_pickle=True).item()
        data = {'X_train':data_dict['X_train'],
                'X_val':data_dict['X_val'], 
                'X_test':data_dict['X_test'], 
                'Y_train':data_dict['Y_train'], 
                'Y_val':data_dict['Y_val'], 
                'Y_test':data_dict['Y_test']}
        return data

    # If no split dataset is found split the data and save
    if val_size>0:
        X_train, X_val, Y_train, Y_val = train_test_split(X,Y,
                                                      test_size=val_size,
                                                      shuffle=True)
        X_val, X_test, Y_val, Y_test = train_test_split(X_val,Y_val,
                                                      test_size=0.5,
                                                      shuffle=True)
        data = {'X_train':X_train, 'X_val':X_val, 'X_test':X_test, 
                'Y_train':Y_train, 'Y_val':Y_val, 'Y_test':Y_test}
    else: 
        data = {'X_train':X, 'X_val':None, 'X_test':None, 
                'Y_train':Y, 'Y_val':None, 'Y_test':None}
    
    np.save(split_data_file_name,data)

    return data


def wrap_data_into_dataloader(X_train,Y_train,X_val,Y_val,batch_size,num_workers,size_multiplier=100):
    train_data = DataLoader(WrapDataset(X_train,Y_train, size_multiplier = size_multiplier),
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory = True,
                        shuffle=True)
    val_data = DataLoader(WrapDataset(X_val,Y_val),
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory = True,
                        shuffle=False)

    return train_data, val_data