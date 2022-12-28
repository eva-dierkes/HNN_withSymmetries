import matplotlib.pyplot as plt

import Config.NetConfigurations as NetCfg
import Config.DataConfigurations as DataCfg

import Utils.dataUtils as dataUtils
import Utils.torchUtils as torchUtils
import Utils.plotUtils as plotUtils


if __name__ == "__main__":
    data_config_list = [
                        'PendCart',
                        'Kepler_cartesian'
                        ] 
    net_config_list = [ 
                        'NN',
                        'HNN',
                        'SymHnn_RotTrans'
                        ]
    
    folder = 'Models/'

    for data_config_tag in data_config_list:
        data_config = DataCfg.cfg[data_config_tag]
        data_config['save_dir'] = folder + data_config['save_dir']
        save_dir = data_config['save_dir'] 

        print(f'Dataclass: {data_config["example"]}')
        x,dx,exampleClass,all_train_traj = dataUtils.prepare_training_data(data_config)
        # plotUtils.plot_training_data(x, all_train_traj, data_config, exampleClass)

        for net_config_tag in net_config_list: 
            net_config = NetCfg.cfg[net_config_tag]

            data = dataUtils.train_val_split(x, dx, val_size=net_config['train_args']['val_size'], save_dir=data_config['save_dir'])
            # plotUtils.plot_train_val_test_data(data['X_train'],data['X_val'],data['X_test'])
            # plt.show()
            torchUtils.train(data,exampleClass,data_config,net_config)
