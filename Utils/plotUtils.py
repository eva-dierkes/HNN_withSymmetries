from cmath import nan
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import seaborn as sn

import Utils.odeUtils as odeUtils


def plot_all_states(x_list,t_span,labels,color=None,title=None,y_label='',n=1,legend=False):
    if not color:
        color = sn.color_palette(n_colors=len(x_list)) 
    q_dim = int(x_list[0].shape[0]/2)
    
    fig1, ax1 = plt.subplots(2,q_dim)
    figs = [fig1]
    #Phaseplot if only one q
    if q_dim ==1 and not y_label=='err':
        fig3, ax3 = plt.subplots()
        figs.append(fig3)
        ax3.set_xlabel('q')
        ax3.set_ylabel('p')
        if legend:
            ax3.legend()
        if y_label == 'err':
            fig3.suptitle(title+'diff plot')
        else:
            fig3.suptitle(title)

    column_names = ['q','p']

    for i,x in enumerate(x_list):
        for column_index in range(x.shape[0]):
            if ax1.shape == (2,):
                this_ax = ax1[column_index]
            else:
                this_ax = ax1[int(column_index/q_dim), column_index%q_dim]
            this_ax.plot(t_span[::n],x[column_index,::n],'*',color=color[i],label=labels[i])

            #getting right ylabels
            if y_label == 'err':
                this_ax.set_ylabel(f'|{column_names[int(column_index/q_dim)]}{column_index%q_dim}_ref-{column_names[int(column_index/q_dim)]}{column_index%q_dim}|')
            else:
                this_ax.set_ylabel(f'{column_names[int(column_index/q_dim)]}{column_index%q_dim}')
            if int(column_index/q_dim) == 2:
                this_ax.set_xlabel('time [s]')

        if q_dim ==1  and not y_label == 'err':
            ax3.plot(x[0,::n],x[1,::n],color=color[i],label=labels[i])

    if legend:
        if ax1.shape == (2,):
            ax1[0].legend()
        else:
            ax1[0,0].legend()
    fig1.suptitle(title)
    fig1.tight_layout()
    return figs


def plot_q_states(x_list,labels,color=None,title=None,legend=False, fig=None, ax=None, n=1):
    if not color:
        color = sn.color_palette(n_colors=len(x_list)) 
    q_dim = int(x_list[0].shape[0]/2)

    if fig==None:
        fig, ax = plt.subplots()

    if q_dim==2:
        ax.set_xlabel('q0')
        ax.set_ylabel('q1')
    
        for i,x in enumerate(x_list):
            ax.plot(x[0,::n],x[1,::n],'*',color=color[i],label=labels[i])
        
        if legend:
            ax.legend()
        else:
            ax.set_title(title)


def plot_training_data(x, all_train_traj, data_config, example_class):
    nbr_traj = data_config['data_args']['nbr_of_traj']
    t_span = np.linspace(data_config['data_args']['t_start'], data_config['data_args']['t_end'], 
                    int(data_config['data_args']['nbr_points_per_traj']))
    figures = []
    if len(t_span)>1:
        x_reshaped = x.reshape((nbr_traj,int(x.shape[0]/nbr_traj),x.shape[1]))
        x_reshaped = [xi.T for xi in x_reshaped] 
        figures = plot_all_states(x_reshaped,t_span,[None]*nbr_traj,sn.color_palette(None, nbr_traj),f'All {data_config["example"]} training data',y_label='',legend=False)

    if data_config['example']=='PendCart':
        cm = plt.get_cmap('gist_rainbow')
    elif data_config['example']=='Kepler_cartesian':
        fig, ax = plt.subplots()
        figures.append(fig)
        cm = plt.get_cmap('gist_rainbow')
        for i,traj in enumerate(all_train_traj):
            ax.plot(traj[0,0],traj[0,1],'*',color=cm(i/len(all_train_traj)))
            ax.quiver(traj[0,0],traj[0,1],traj[0,2],traj[0,3],color=cm(i/len(all_train_traj)))
            ax.plot(traj[:,0],traj[:,1],color=cm(i/len(all_train_traj)))
        ax.set_aspect('equal','box')

    return figures


def plot_train_val_test_data(X_train,X_val,X_test):
    fig,ax = plt.subplots()
    ax.plot(X_train[:,0],X_train[:,1],'*r',label='train')
    ax.plot(X_val[:,0],X_val[:,1], '*b', label='val')
    ax.plot(X_test[:,0],X_test[:,1], '*g', label='test')
    ax.legend()
    return fig


def plot_level_sets_constant_p(exampleClass, 
                                model, 
                                hamiltonian, 
                                constant_p=0, 
                                q_grid=None,
                                color=None,
                                label=None, 
                                line_list=[], 
                                label_list=[],
                                positions_of_level_sets=None,
                                fig=None, ax=None, minmax=None):
    n = 50                             
    if q_grid is None:
        q = np.empty((model.dimq,n))
        q[0,] = np.linspace(-20,20,n)
        q[1,] = np.linspace(-20,20,n)
        q1, q2 = np.meshgrid(q[0,],q[1,])
        mesh_shape = q1.shape
        p = constant_p * np.ones((model.dimq, n,n))     #p=0 for potential
        grid = np.vstack([np.stack([q1,q2]),p]).reshape(2*model.dimq,-1)
    else:
        q1, q2 = np.meshgrid(q_grid[0,],q_grid[1,])
        mesh_shape = q1.shape
        p = constant_p * np.ones((model.dimq, n,n))     #p=0 for potential
        grid = np.vstack([np.stack([q1,q2]),p]).reshape(2*model.dimq,-1)

    if isinstance(exampleClass, odeUtils.Kepler_cartesian):
        not_in_grid = np.sqrt(q1**2+q2**2) < 2 
    else:
        not_in_grid = np.full(mesh_shape,False)

    if isinstance(model,nn.Module):
        grid = torch.tensor(grid)
        H = hamiltonian(grid.T.float()).detach().numpy()
    else:
        H = hamiltonian(grid)
        
    H = H.reshape(*mesh_shape)
    H[not_in_grid]=nan
    if fig is None:
        fig, ax = plt.subplots()
    
    if not minmax:
        minmax = [np.min(H), np.max(H)]

    if color is None:
        H-=H[0]
        cf = ax.contourf(q1,q2,H,30)
        fig.colorbar(cf,ax=ax)
        ax.set_title(label)
    else:
        if positions_of_level_sets is None:
            cf = ax.contour(q1,q2,H,colors=color)
        else: 
            positions_of_level_sets = np.hstack((positions_of_level_sets, constant_p * np.ones_like(positions_of_level_sets)))
            if isinstance(model,nn.Module):
                level_heights = hamiltonian(torch.Tensor(positions_of_level_sets)).detach().numpy().reshape(-1,)
                level_heights = np.sort(level_heights)
            else:
                level_heights = hamiltonian(positions_of_level_sets.T)
                level_heights = np.sort(level_heights)
            cf = ax.contour(q1,q2,H,levels=level_heights, colors=color,linestyles='dashed')
        line_list.append(cf.collections[0])
        label_list.append(label)
        ax.legend(line_list,label_list)
        ax.set_title(f'p={constant_p}')
    
    ax.set_xlabel('q1')
    ax.set_ylabel('q2')
    ax.set_aspect('equal', 'box')
    
    return line_list, label_list


def plot_level_sets_constant_q(model, 
                                hamiltonian, 
                                constant_q=0, 
                                p_grid=None,
                                color=None, 
                                label=None, 
                                line_list=[], 
                                label_list=[],
                                positions_of_level_sets=None,
                                fig=None, ax=None, minmax=None):
    n = 50
    if p_grid is None:
        p = np.empty((model.dimq,n))
        p[0,] = np.linspace(-10,10,n)
        p[1,] = np.linspace(-10,10,n)
        p1, p2 = np.meshgrid(p[0,],p[1,])
        q = constant_q*np.ones((model.dimq, n,n))     #p=0 for potential
        grid = np.vstack([q,np.stack([p1,p2])]).reshape(2*model.dimq,-1)
    else:
        p1, p2 = np.meshgrid(p_grid[0,],p_grid[1,])
        q = constant_q*np.ones((model.dimq, n,n))     #p=0 for potential
        grid = np.vstack([q,np.stack([p1,p2])]).reshape(2*model.dimq,-1)

    if isinstance(model,nn.Module):
        grid = torch.tensor(grid)
        H = hamiltonian(grid.T.float()).detach().numpy()
    else:
        H = hamiltonian(grid)
    H = H.reshape(n,n)

    if fig is None:
        fig, ax = plt.subplots()
    if not minmax:
        minmax = [np.min(H), np.max(H)]

    if color is None:
        H -= H[0]
        cf = ax.contourf(p1,p2,H,30)
        fig.colorbar(cf,ax=ax)
        ax.set_title(label)
    else:
        if positions_of_level_sets is None:
            cf = ax.contour(p1,p2,H,colors=color)
        else: 
            positions_of_level_sets = np.hstack((constant_q * np.ones_like(positions_of_level_sets),positions_of_level_sets))
            if isinstance(model,nn.Module):
                level_heights = hamiltonian(torch.Tensor(positions_of_level_sets)).detach().numpy().reshape(-1,)
                level_heights = np.sort(level_heights)
            else:
                level_heights = hamiltonian(positions_of_level_sets.T)
                level_heights = np.sort(level_heights)
            cf = ax.contour(p1,p2,H,levels=level_heights, colors=color,linestyles='dashed')

        line_list.append(cf.collections[0])
        label_list.append(label)
        ax.legend(line_list,label_list)
        ax.set_title(f'q={constant_q}')

    ax.set_xlabel('p1')
    ax.set_ylabel('p2')
    ax.set_aspect('equal', 'box')
    return line_list, label_list


def plot_conserved_quantity(model,  trajectory, t_span, color=None, linestyle='-',label=None, fig=None, ax=None, n=1, title=None):
    [q,p] = np.array_split(trajectory,indices_or_sections=2,axis=0)
    symmetries = zip(model.translation_factor, model.rotation_factor)
    nbr_symmetries = model.translation_factor.shape[0]
    for symmetry in symmetries:
        translation, rotation = symmetry
        if isinstance(model,nn.Module):
            rotation = rotation.detach().numpy()
            translation = translation.detach().numpy()
            
        I = np.array([-p[:,i].T @ ((rotation @ q[:,i]) + translation) for i in range(q.shape[-1])])
        I -= I[0,]
        if fig is None:
            fig, ax = plt.subplots()
        cf = ax.plot(t_span[::n],I[::n],label=label, color=color, linestyle=linestyle)
        ax.legend()
        ax.set_xlabel('time')
        ax.set_ylabel('I = -p.T (R q+T)')
        if title: ax.set_title(title)


def plot_H_of_traj(model, hamiltonian, trajectory, t_span, color=None, label=None, fig=None, ax=None,n=1,title=None):
    if isinstance(model,nn.Module):
        trajectory = torch.from_numpy(trajectory).T.float()
        H = hamiltonian(trajectory).detach().numpy()
    else: 
        H = hamiltonian(trajectory)
    H -= H[0,]
    ax.plot(t_span[::n], H[::n], label=label,color=color)
    ax.legend()
    if title: ax.set_title(title)


def plot_l_sym_overq(df,fig=None, ax=None,title=None):
    s = ax.scatter(df.x, df.y, s=20, c=df.v_hat)
    plt.colorbar(s,ax=ax)
    ax.set_xlabel('q_0')
    ax.set_ylabel('q_1')
    ax.set_aspect('equal', 'box')
    if title: ax.set_title(f'{title}')

def plot_l_sym_over_points(df, color=None,fig=None, ax=None,title=None,label=None):
    ax.plot(df.v_hat, '*', color=color,label=label)
    ax.set_ylabel('l_sym')
    if title: ax.set_title(f'{title}')


