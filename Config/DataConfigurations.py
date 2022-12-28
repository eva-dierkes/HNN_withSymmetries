
cfg = {
    'PendCart' : {
        'example': 'PendCart',
        'save_dir': '',
        'ode_args' : {
            'm': 1,
            'M': 1,
            'l': 1,
            'g': 9.81
        },

        'data_args' : {
            't_start': 0,
            't_end': 1,
            'nbr_points_per_traj': 45, 
            'nbr_of_traj': 1000, 
            'noise': 1e-2,                             # data + noise*np.random.randn
            'seed': 0
        }
    },

    
    'Kepler_cartesian' : {
        'example': 'Kepler_cartesian',
        'save_dir': '',
        'ode_args' : {
            'm': 1,
            'g': 1.016895192894334*1e3
        },

        'data_args' : {
            't_start': 0,
            't_end': 10,
            'nbr_points_per_traj': 10,
            'nbr_of_traj': 5000,
            'noise': 1e-2,                             # data + noise*np.random.randn
            'seed': 0
        }
    }
}