import torch 

cfg = {
    'Kepler_cartesian': {
        't_start': 0,
        't_end': 200,
        'sampling_rate': 500,
        'region_tag' : ''
    },
}

data_cfg = {
    'Kepler_cartesian' : {
        'example': 'Kepler_cartesian',
        'save_dir': '',#'Kepler_cartesian/',
        'ode_args' : {
            'm': 1,
            'g': 1.016895192894334*1e3
        },

        'data_args' : {
            't_start': 0,
            't_end': 10,
            'nbr_points_per_traj': 10,
            'nbr_of_traj': 5000,
            'noise':1e-2,# 0,#                             # data + noise*np.random.randn
            'seed': 0
        }
    }
}
Netcfg = {
    'NN' : {
        'net' : 'NN',
        'loss_tag' : 'mse',
        'seed' : 0,
        'train_args' : {
            'hidden' : [256,256,256],
            'activation' : torch.nn.Softplus(),
            'epochs'   : 20000,
            'size_multiplier' : 1,            #augments the trainingsdata to speed up efficiency (effectif epochs = n_epoch*size_multiplier)
            'batch_size' : 2048,
            'early_stop' : 10000,
            'dropout'    : 0,
            'lr'         : 5e-3,
            'weight_decay' : 0,
            'val_size': 0.3,
            'normalized_inputdata': 0,
            'init_weights': None,               #None,'orthogonal','lagrangian'
            'gpus': 1,
            'scheduler_dict' : {
                'reduceLRonPlateau' :{
                            'factor': 0.95,
                            'patience':50,
                            'min_lr': 1e-9
                },
            },
        },
        'tag': ''#'_last-v1'                                       #tag can be used for different versions e.g. '-v2'
    },

    'HNN' : {
        'net' : 'HNN',
        'loss_tag' : 'mse',
        'seed' : 0,
        'train_args' : {
            'hidden' : [256,256,256],
            'activation' : torch.nn.Softplus(),
            'epochs'   : 20000,
            'size_multiplier' : 1,           #augments the trainingsdata to speed up efficiency (effectif epochs = n_epoch*size_multiplier)
            'batch_size' : 2048,
            'early_stop' : 10000,
            'dropout'    : 0,
            'lr'         : 5e-3,
            'weight_decay' : 0,
            'val_size': 0.3,
            'normalized_inputdata': 0,
            'init_weights': None,   #None,'orthogonal','lagrangian'
            'gpus': 1,
            'scheduler_dict' : {
                'reduceLRonPlateau' :{
                            'factor': 0.95,
                            'patience': 50,
                            'min_lr': 1e-9
                },
            },
        },
        'tag': ''                                       #tag can be used for different versions e.g. '-v2'
    },

    'SymHnn' : {
        'net' : 'SymHnn',
        'loss_tag' : '',
        'symmetry_type': ['Rot','Trans'],
        'seed' : 0,
        'train_args' : {
            'hidden' : [256,256,256],
            'activation' : torch.nn.Softplus(),
            'epochs'   : 20000,
            'size_multiplier' : 1,           #augments the trainingsdata to speed up efficiency (effectif epochs = n_epoch*size_multiplier)
            'batch_size' : 2048,
            'early_stop' : 10000,
            'dropout'    : 0,
            'lr'         : 5e-3,
            'weight_decay' : 0,
            'val_size': 0.3,
            'normalized_inputdata': 0,
            'init_weights': None,   #None,'orthogonal','lagrangian'
            'gpus': 1,
            'scheduler_dict' : {
                'reduceLRonPlateau' :{
                            'factor': 0.95,
                            'patience': 50,
                            'min_lr': 1e-9
                },
            },
        },
        'tag': '_last'
    },

}