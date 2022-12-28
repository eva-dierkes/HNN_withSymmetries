import torch 

cfg = {
    'Pend': {
        't_start': 0,
        't_end': 60,
        'sampling_rate': 15,
        'region_tag' : 'Evaluation'  #wenn leer, dann gleicher Datenbereich wie Trainingsdaten
    },

    'PendCart': {
        't_start': 0,
        't_end': 30,
        'sampling_rate': 15,
        'region_tag' : ''
    },

    'PendCart_rotated': {
        't_start': 0,
        't_end': 30,
        'sampling_rate': 15,
        'region_tag' : ''
    },

    'Kepler': {
        't_start': 0,
        't_end': 2.5,
        'sampling_rate': 15,
        'region_tag' : ''
    },

}

Netcfg = {
    'NN' : {
        'net' : 'NN',
        'loss_tag' : 'mse',
        'seed' : 0,
        'train_args' : {
            'hidden' : [512,512,512],
            'activation' : torch.nn.Softplus(),#torch.tanh,
            'epochs'   : 100,
            'size_multiplier' : 100,            #augments the trainingsdata to speed up efficiency (effectif epochs = n_epoch*size_multiplier)
            'batch_size' : 8192,
            'early_stop' : 500,
            'dropout'    : 0,
            'lr'         : 1e-3,
            'weight_decay' : 1e-4,
            'val_size': 0.3,
            'normalized_inputdata': 0,
            'init_weights': None,               #None,'orthogonal','lagrangian'
            'gpus': 1,
            'scheduler_dict' : {
                # 'stepLR': {'step_size':1,
                #             'gamma':0.8
                #             },
                'multistepLR' : {
                            'milestones': [20,40,60],#[i for i in range(20)],
                            'gamma': 0.95
                            },
                'reduceLRonPlateau' :{
                            'factor': 0.8,
                            'patience': 10,
                            'min_lr': 1e-4
                },
            },
        }
    },

    'HNN' : {
        'net' : 'HNN',
        'loss_tag' : 'mse',
        'seed' : 0,
        'train_args' : {
            'hidden' : [512,512,512],
            'activation' : torch.nn.Softplus(),#torch.tanh,
            'epochs'   : 100,
            'size_multiplier' : 100,            #augments the trainingsdata to speed up efficiency (effectif epochs = n_epoch*size_multiplier)
            'batch_size' : 8192,
            'early_stop' : 500,
            'dropout'    : 0,
            'lr'         : 1e-3,
            'weight_decay' : 1e-4,
            'val_size': 0.3,
            'normalized_inputdata': 0,
            'init_weights': None,   #None,'orthogonal','lagrangian'
            'gpus': 1,
            'scheduler_dict' : {
                # 'stepLR': {'step_size':1,
                #             'gamma':0.8
                #             },
                'multistepLR' : {
                            'milestones': [20,40,60],#[i for i in range(20)],
                            'gamma': 0.95
                            },
                'reduceLRonPlateau' :{
                            'factor': 0.8,
                            'patience': 10,
                            'min_lr': 1e-4
                },
            },
        },
        'tag': ''                                       #tag can be used for different versions e.g. '-v2'
    },

    'SymHnn' : {
        'net' : 'SymHnn',
        'loss_tag' : '',
        'symmetry_type': ['Rot','Trans'],           #['Rot','Trans'] 
        'seed' : 0,
        'train_args' : {
            'hidden' : [512,512,512],
            'activation' : torch.nn.Softplus(),#torch.tanh,
            'epochs'   : 100,
            'size_multiplier' : 100,            #augments the trainingsdata to speed up efficiency (effectif epochs = n_epoch*size_multiplier)
            'batch_size' : 8192,
            'early_stop' : 500,
            'dropout'    : 0,
            'lr'         : 1e-3,
            'weight_decay' : 1e-4,
            'val_size': 0.3,
            'normalized_inputdata': 0,
            'init_weights': None,   #None,'orthogonal','lagrangian'
            'gpus': 1,
            'scheduler_dict' : {
                # 'stepLR': {'step_size':1,
                #             'gamma':0.8
                #             },
                'multistepLR' : {
                            'milestones': [20,40,60],#[i for i in range(20)],
                            'gamma': 0.95
                            },
                'reduceLRonPlateau' :{
                            'factor': 0.8,
                            'patience': 10,
                            'min_lr': 1e-4
                },
            },
        },
        'tag': '-v1'
    },

    'SymLib_Hnn' : {
        'net' : 'SymLib_Hnn',
        'loss_tag' : 'SymLib',
        'seed' : 0,
        'train_args' : {
            'hidden' : [512],
            'activation' : torch.tanh,
            'epochs'   : 2,
            'size_multiplier' : 2,            #augments the trainingsdata to speed up efficiency (effectif epochs = n_epoch*size_multiplier)
            'batch_size' : 8192     ,
            'early_stop' : 500,
            'dropout'    : 0,
            'lr'         : 1e-3,
            'weight_decay' : 1e-4,
            'val_size': 0.3,
            'normalized_inputdata': 0,
            'init_weights': 'lagrangian',   #None,'orthogonal','lagrangian'
            'gpus': 1,
            'scheduler_dict' : {
                # 'stepLR': {'step_size':1,
                #             'gamma':0.8
                #             },
                # 'multistepLR' : {
                #             'milestones': [3],#[i for i in range(20)],
                #             'gamma': 0.1
                #             },
                'reduceLRonPlateau' :{
                            'factor': 0.8,
                            'patience': 10,
                            'min_lr': 1e-4
                },
            },
        },
        'tag': ''
    },
}