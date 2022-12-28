import torch

train_args = {
            'hidden' : [256,256,256],
            'activation' : torch.nn.Softplus(), 
            'epochs'   : 20000,
            'size_multiplier' : 1,           #augments the trainingsdata to speed up efficiency (effective epochs = n_epoch*size_multiplier)
            'batch_size' : 2048,
            'early_stop' : 10000,
            'dropout'    : 0,
            'lr'         : 5e-3 , 
            'weight_decay' : 0,
            'val_size': 0.3,
            'normalized_inputdata': 0,
            'init_weights': None,             #None,'orthogonal','lagrangian'
            'gpus': 1,
            'scheduler_dict' : {
                'reduceLRonPlateau' :{
                            'factor': 0.95,
                            'patience': 50,#
                            'min_lr': 1e-9
                },
            },
        }

cfg = {
    'NN' : {
        'net' : 'NN',
        'loss_tag' : 'mse',             #if empty use othe lossf of the Network
        'seed' : 0,
        'train_args' : train_args
    },

    'HNN' : {
        'net' : 'HNN',
        'loss_tag' : 'mse',
        'seed' : 0,
        'train_args' : train_args
    },

    'SymHnn_RotTrans' : {
        'net' : 'SymHnn',
        'loss_tag' : '',    
        'symmetry_type': ['Rot','Trans'],
        'seed' : 0,
        'train_args' : train_args
    },
    
}