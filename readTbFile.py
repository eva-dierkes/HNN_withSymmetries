import torch

import Config.EvalConfigurations_PendCart as EvalConfig_PendCart
import Config.EvalConfigurations_KeplerCartesian as EvalConfig_KeplerCartesian
import Utils.dataUtils as dataUtils
import Utils.evalUtils as evalUtils


if __name__ == "__main__":
    data_config_list = [
                        'PendCart',
                        'Kepler_cartesian'
                        ] 
    net_config_list = [ 
                        'NN',
                        'HNN',
                        'SymHnn',
                        ]
    
    folder = 'Models/'

    #If models for different criteria sets should be compared
    criteria = None             #criteria as named in netConfig['train_args']
    criteria_values =  None     #values that should be used for criteria

    save_tikz = False

    for data_config_tag in data_config_list:
        if data_config_tag == 'PendCart':
            EvalCfg = EvalConfig_PendCart
        elif data_config_tag == 'Kepler_cartesian':
            EvalCfg = EvalConfig_KeplerCartesian

        traindata_config = EvalCfg.data_cfg[data_config_tag]

        traindata_config['save_dir'] = folder + traindata_config['save_dir']
        ref_model = evalUtils.create_model(traindata_config)

        model_list = [ref_model]
        hamiltonian_list = [ref_model.hamiltonian]
        ode_list = [ref_model.ode]
        label_list = ['exact']

        #load data set
        x,dx,exampleClass,all_train_traj = dataUtils.prepare_training_data(traindata_config)

        for net_config_tag in net_config_list: 
            net_config = EvalCfg.Netcfg[net_config_tag]

            net_config = EvalCfg.Netcfg[net_config_tag]
            current_model_list, current_ode_list, \
                current_label_list, criteria_list  = evalUtils.get_all_models_for_criteria(criteria, criteria_values, net_config, traindata_config, ref_model, dict_with_criteria='data_config')
            hamiltonian_list += [model.forward for model in current_model_list]
            model_list += current_model_list
            ode_list += current_ode_list
            label_list += [f'{data_config_tag}_{net_config_tag}'] if criteria==None else [f'{data_config_tag}_{net_config_tag}_{criteria_tag}' for criteria_tag in criteria_list]#current_label_list
            
        #load data set
        loss_values = {}
        for model,label in zip(model_list[1:],label_list[1:]):
            data = dataUtils.train_val_split(x, dx, val_size=net_config['train_args']['val_size'], save_dir=traindata_config['save_dir'])

            loss_values[label] = [model.lossf(model,torch.Tensor(data[f'X_{data_tag}']),torch.Tensor(data[f'Y_{data_tag}']),'eval') for data_tag in ['train', 'val', 'test' ]]

        for label in label_list[1:]:
            print(f'{label:<25}\t\t{"train":<35}{"val":<35}{"test":<35}')
            if 'SymHnn' in label:
                print(f'\t{"L":<25}{loss_values[label][0][0].detach().numpy():<35}{loss_values[label][1][0].detach().numpy():<35}{loss_values[label][2][0].detach().numpy():<35}')
                print(f'\t{"L_dyn":25}{loss_values[label][0][1].detach().numpy():<35}{loss_values[label][1][1].detach().numpy():<35}{loss_values[label][2][1].detach().numpy():<35}')
                print(f'\t{"L_sym":<25}{loss_values[label][0][2].detach().numpy():<35}{loss_values[label][1][2].detach().numpy():<35}{loss_values[label][2][2].detach().numpy():<35}')
            else:
                print(f'\t{"L":<25}{loss_values[label][0].detach().numpy():<35}{loss_values[label][1].detach().numpy():<35}{loss_values[label][2].detach().numpy():<35}')
