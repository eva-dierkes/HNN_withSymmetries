import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

import torch
import torch.nn as nn
import pandas as pd

import Utils.dataUtils as dataUtils
import Utils.torchUtils as torchUtils
import Utils.odeUtils as odeUtils
import Utils.lossUtils as lossUtils
from Nets import DenseNet, HnnNet, SymHnnNet


#%%
def solve_for_all_models(ode_list, t_start, t_end, t_span, x0, title, use_symplectic_midpoint=False):
    x = []
    
    for i, ode in enumerate(ode_list):
        print(f'For {x0} solve '+ title[i])
        if not use_symplectic_midpoint:
            sol = solve_ivp(ode,[t_start,t_end],x0,t_eval=t_span,rtol=1e-10,atol=1e-10)
            sol = sol.y
        else: 
            sol = symplectic_midpoint(ode,t_span,x0=x0)
        x.append(sol)
    
    return x


def symplectic_midpoint(ode,t_span,x0):
    qp_dim = x0.shape[0]
    sol = np.empty((qp_dim,t_span.shape[0]))

    sol[:,0] = x0
    for i, h in enumerate(np.diff(t_span[:])):
        sym_midpoint = lambda x_n1 : x_n1 - sol[:,i] - h * ode(0,(x_n1+sol[:,i])/2)  #ode = J^-1 @ dH
        sol[:,i+1] = fsolve(sym_midpoint,x0=sol[:,i])

    return sol


def generate_solution_for_inital_value(run_index, x0, ode_list, t_start, t_end, t_span, labels, model):
        print(f"{run_index}-run:\nInitial value: {x0}") 
        x = solve_for_all_models(ode_list,
                                 t_start=t_start,
                                 t_end=t_end,
                                 t_span=t_span,
                                 x0=x0,
                                 title=labels,
                                 use_symplectic_midpoint=True)
    
        x_err = [x_-x[0] for x_ in x[1:]]
        h = [model.hamiltonian(x_) for x_ in x]
        h_err = [h_-h[0] for h_ in h[1:]]
        return [x,h,x_err,h_err,x0]


def get_model_folder_and_file(data_config, net_config, nbr_states):
    model_file_suffix = dataUtils.create_model_file_name(net_config['train_args'])

    # chosse appropiate model structure
    label = net_config['net']
    if label=='NN':
        model = DenseNet.DenseNet(input_dim=nbr_states,output_dim=nbr_states)
    elif label=='HNN':
        model = HnnNet.HnnNet(input_dim=nbr_states)
    elif label=='SymHnn':
        model = SymHnnNet.SymHnnNet(input_dim=nbr_states)
        label += '_' +''.join(net_config['symmetry_type'])

    if net_config['loss_tag']=='mse':
        loss_tag = '_mse'
    else:
        loss_tag = ''

    if data_config['example'] == 'PendCart_rotated':
        data_config['example'] = data_config['example'] + str(data_config['ode_args']['rotation_angle'] if 'rotation_angle' in data_config['ode_args'] else '')

    tag = net_config['tag'] if 'tag' in net_config else ''

    save_dir = data_config['save_dir']
    model_prefix = f'model_{data_config["example"]}_{label}{loss_tag}' + model_file_suffix + tag
    file_name = save_dir + f'Models/{model_prefix}.ckpt'
    model_name = f'{data_config["example"]}_'+label+loss_tag+model_file_suffix+tag

    return model, save_dir, file_name, model_prefix, model_name


def load_model(data_config, net_config, nbr_states):    
    
    model, save_dir, file_name, model_prefix, model_name = get_model_folder_and_file(data_config, net_config, nbr_states)
    print(f'load Model: {file_name}')
    model = model.load_from_checkpoint(file_name)
    # Select lossfunction if defined in net config
    if net_config['loss_tag']=='mse':
        model.lossf = lambda model,x,y,status: lossUtils.my_mse_loss(model,x,y,status)[0]

    evalModel = torchUtils.EvalModel(model,requires_grad=False)

    return evalModel, model_name, model


def get_all_models_for_criteria(criteria, criteria_values, net_config, data_config, ref_model, dict_with_criteria='net_config'):
    model_list = []
    ode_list = []
    label_list = []
    criteria_list = []
    save_dir = data_config['save_dir'] 
    if criteria == None:
                evalModel, model_name, model = load_model(data_config, net_config, nbr_states=ref_model.nbr_states)

                model_list.append(model)
                ode_list.append(evalModel.identified_ode)
                label_list.append(model_name)
    else:
        for criteria_value in criteria_values:
            if dict_with_criteria=='net_config':    
                net_config['train_args'][criteria] = criteria_value
            elif dict_with_criteria == 'data_config':
                data_config['save_dir'] = save_dir
                data_config['data_args'][criteria] = criteria_value
                data_config['train_data_file_path'] = dataUtils.create_train_data_file_path(data_config['example'], data_config['ode_args'], 
                                                                                            data_config['data_args'])
                data_config['save_dir'] = '/'.join(data_config['save_dir'].split('/')[:2])+'/'+ data_config['train_data_file_path']+'/'
                

            evalModel, model_name, model = load_model(data_config, net_config, nbr_states=ref_model.nbr_states)

            model_list.append(model)
            ode_list.append(evalModel.identified_ode)
            label_list.append(f'{model_name}_{criteria}_{criteria_value}')
            criteria_list.append(f'{criteria}{criteria_value}')

    return model_list, ode_list, label_list, criteria_list


def create_model(data_config):
    exampleClass = None
    if data_config['example']=='PendCart':
        exampleClass = odeUtils.Pendulum_on_a_cart(data_config['ode_args'])
    elif data_config['example']=='Kepler_cartesian':
        exampleClass = odeUtils.Kepler_cartesian(data_config['ode_args'])
    
    return exampleClass



def l_sym_term_for_model_on_grid(grid, dH, model, model_tag, symmetry, dimq):
    if isinstance(model,nn.Module):
        grid = torch.from_numpy(grid).float()
        dH_grid = dH(grid).detach().numpy() #q_dot = dH/dp, p_dot = -dH/dq
        grid = grid.detach().numpy()
    else:
        dH_grid = dH(t=0,x=grid.T)          #dH/dp, -dH/dq
        dH_grid = dH_grid.T

    if isinstance(symmetry[0],torch.Tensor):
        rotation = symmetry[0].detach().numpy()
        translation = symmetry[1].detach().numpy()
    else:
        rotation = symmetry[0]
        translation = symmetry[1]
    v_hat = np.stack([-(rotation   @ grid[i,:dimq ] + translation) @ -dH_grid[i, dimq:] + \
                           (rotation.T @ grid[i, dimq:]).T               @  dH_grid[i, :dimq ] for i in range(grid.shape[0])]) 
    
    sym_norm = (np.linalg.norm(translation) + np.linalg.norm(rotation,ord='fro'))
    loss_v_hat = np.linalg.norm(v_hat[~np.isnan(v_hat)])/sym_norm

    df = pd.DataFrame({'x': grid[:,0],
                        'y': grid[:,1],
                        'v_hat': v_hat,
                        'Net': [model_tag]*grid.shape[0]
                        })
    return df, loss_v_hat