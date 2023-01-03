import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import tikzplotlib
import pandas as pd

import Config.EvalConfigurations_PendCart as EvalConfig_PendCart
import Config.EvalConfigurations_KeplerCartesian as EvalConfig_KeplerCartesian
import Utils.evalUtils as evalUtils
import Utils.plotUtils as plotUtils
import Utils.dataUtils as dataUtils

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

    save_tikz = True

    for data_config_tag in data_config_list:
        if data_config_tag == 'PendCart':
            EvalCfg = EvalConfig_PendCart
        elif data_config_tag == 'Kepler_cartesian':
            EvalCfg = EvalConfig_KeplerCartesian

        traindata_config = EvalCfg.data_cfg[data_config_tag]
        traindata_config['train_data_file_path'] = dataUtils.create_train_data_file_path(traindata_config['example'],
                                                            traindata_config['ode_args'],
                                                            traindata_config['data_args'])
        traindata_config['save_dir'] = folder + traindata_config['save_dir']+traindata_config['train_data_file_path']+'/'
        ref_model = evalUtils.create_model(traindata_config)

        model_list = [ref_model]
        hamiltonian_list = [ref_model.hamiltonian]
        ode_list = [ref_model.ode]
        label_list = ['exact']

        for net_config_tag in net_config_list: 
            net_config = EvalCfg.Netcfg[net_config_tag]
            current_model_list, current_ode_list, \
                current_label_list, criteria_list  = evalUtils.get_all_models_for_criteria(criteria, criteria_values, net_config, traindata_config, ref_model, dict_with_criteria='data_config')
            hamiltonian_list += [model.forward for model in current_model_list]
            model_list += current_model_list
            ode_list += current_ode_list
            label_list += [f'{data_config_tag}_{net_config_tag}'] if criteria==None else [f'{data_config_tag}_{net_config_tag}_{criteria_tag}' for criteria_tag in criteria_list]#current_label_list

        ## get uniform colors used for all plots
        # red = [(1.0,0.0,0.0)]
        # color = sn.color_palette(n_colors=len(model_list))
        # color.remove(max(color,key=lambda tup:tup[0]))      #remove most redist color
        # color = red + color
        color = [(1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0), (1.0,0.5,0.0)]

        ############  Plot level sets ##############################################################
        nbr_models = len(label_list)
        const_p = {'Kepler_cartesian': [-10,0,10],
                    'PendCart':[-5,0,5]}
        const_q = {'Kepler_cartesian': [-10,3,10],
                    'PendCart':[-5,0,5]}

        fig3,ax3 = plt.subplots(2,len(const_q[data_config_tag]))
        fig3.suptitle(f'hamiltonian level sets')
        line_list_constant_p,label_list_constant_p = [[]],[[]]
        line_list_constant_q,label_list_constant_q = [[]],[[]]
        
        positions_of_level_sets = {
                                    'Kepler_cartesian': np.array([[2.5, 2.5],
                                                                    [4.0, 4.0],
                                                                    [5.5, 5.5],
                                                                    [7.0, 7.0],
                                                                    [8.5, 8.5],
                                                                    [10.0,10.0],
                                                                    [12.5, 12.5],
                                                                    [14.0, 14.0],
                                                                    [15.5, 15.5],
                                                                    [17.0, 17.0],
                                                                    [18.5, 18.5],
                                                                    [20.0,20.0],]),
                                    'PendCart': np.array([[0.0, 0.0],
                                                            [0.0, 1.0],
                                                            [0.0, 2.0],
                                                            [0.0, 3.0],
                                                            [0.0, 4.0],
                                                            [0.0, 5.0],
                                                            [0.0, 6.0]]),
                                                                    }
        n=50
        q_grid_PendCart = np.empty((ref_model.dimq,n))
        q_grid_PendCart[0,] = np.linspace(-10,10,n)
        q_grid_PendCart[1,] = np.linspace(-2*np.pi,2*np.pi,n)
        q_grid_Kepler = np.empty((ref_model.dimq,n))
        q_grid_Kepler[0,] = np.linspace(-20,20,n)
        q_grid_Kepler[1,] = np.linspace(-20,20,n)
        q_grid= {'PendCart':         q_grid_PendCart,
                 'Kepler_cartesian': q_grid_Kepler
                }

        p_grid_PendCart = np.empty((ref_model.dimq,n))
        p_grid_PendCart[0,] = np.linspace(-2,2,n)
        p_grid_PendCart[1,] = np.linspace(-2*np.pi,2*np.pi,n)
        p_grid_Kepler = np.empty((ref_model.dimq,n))
        p_grid_Kepler[0,] = np.linspace(-20,20,n)
        p_grid_Kepler[1,] = np.linspace(-20,20,n)
        p_grid= {'PendCart':            p_grid_PendCart,
                 'Kepler_cartesian':    p_grid_Kepler
                }

        for i, (model, hamiltonian) in enumerate(zip(model_list,hamiltonian_list)):
            if '_NN' in label_list[i]:
                continue
            for j,(q,p) in enumerate(zip(const_q[data_config_tag],const_p[data_config_tag])):
                # for each column add a ne label and line list
                if j < len(line_list_constant_p):
                    line_list_constant_p.append([None])
                    label_list_constant_p.append([None])
                    line_list_constant_q.append([None])
                    label_list_constant_q.append([None])

                
                line_list_constant_p[j],label_list_constant_p[j] = plotUtils.plot_level_sets_constant_p(ref_model, model,
                                                                                                    hamiltonian,
                                                                                                    constant_p=const_p[data_config_tag][j],
                                                                                                    q_grid=q_grid[data_config_tag],
                                                                                                    color=[color[i]],
                                                                                                    label=label_list[i],
                                                                                                    line_list=line_list_constant_p[j], 
                                                                                                    label_list=label_list_constant_p[j],
                                                                                                    positions_of_level_sets=positions_of_level_sets[data_config_tag],
                                                                                                    fig=fig3,ax=ax3[0,j]
                                                                                                    )
                line_list_constant_q[j],label_list_constant_q[j] = plotUtils.plot_level_sets_constant_q(  model,
                                                                                                    hamiltonian,
                                                                                                    constant_q=const_q[data_config_tag][j],
                                                                                                    p_grid = p_grid[data_config_tag],
                                                                                                    color=[color[i]],
                                                                                                    label=label_list[i],
                                                                                                    line_list=line_list_constant_q[j], 
                                                                                                    label_list=label_list_constant_q[j],
                                                                                                    positions_of_level_sets=positions_of_level_sets[data_config_tag],
                                                                                                    fig=fig3,ax=ax3[1,j]
                                                                                                    )


        if save_tikz:
            tikzplotlib.save(f'Results/{data_config_tag}_LevelSets.tex')
            plt.close()

        ############  Eval models on single trajectory #######################################################
        evalData_cfg = EvalCfg.cfg[data_config_tag]
        t_span = np.linspace(evalData_cfg['t_start'],evalData_cfg['t_end'],evalData_cfg['sampling_rate']*(evalData_cfg['t_end']-evalData_cfg['t_start']))
        # start values PendCart
        if data_config_tag == 'PendCart':
            # x0 = ref_model.generate_random_inital_value()
            x0 = np.array([0.46873577, 2.33147689, 0.48886349, 1.54074325])
        # start values Kepler cartesian 
        elif data_config_tag == 'Kepler_cartesian':
            # x0 = ref_model.generate_random_inital_value(region_tag='Evaluation')
            # x0 = np.array([-7.00950459,  5.00748932,  5.98324499,  8.09541063])
            # x0 = np.array([ 11.09712163,  2.46248389, -1.50878849,  6.7993173]) 
            k = 1.016895192894334*1e3
            x0 = np.array([7,0,np.sqrt(k/7.)/7,np.sqrt(k/7.)]) #Kreis
        else: 
            x0 = ref_model.generate_random_inital_value()
        trajectory,_,trajectory_err,_,_ = evalUtils.generate_solution_for_inital_value(None, x0=x0 , ode_list=ode_list,t_start=t_span[0], t_end=t_span[-1], t_span=t_span, labels=label_list, model=ref_model)
        
        ############  Plot all states for trajectory ##############################################################
        
        plotUtils.plot_all_states(trajectory,t_span, label_list,color=color, title='Example trajectory for different models', y_label='',legend=True)
        
        if len(label_list)>1:
            plotUtils.plot_all_states(trajectory_err,t_span, label_list[1:],color=color[1:], title='Example trajectory for different models', y_label='err',legend=True)
        
        fig2, ax2 = plt.subplots(1,len(trajectory))
        fig2.suptitle('Example trajectory')
        for i,(traj,label) in enumerate(zip(trajectory,label_list)):
            plotUtils.plot_q_states([traj],[label],color=color,title=label,legend=False,fig=fig2, ax=ax2[i])    

        if save_tikz:
            tikzplotlib.save(f'Results/{data_config_tag}_S_err.tex')
            plt.close()
            tikzplotlib.save(f'Results/{data_config_tag}_S.tex')
            plt.close()
            tikzplotlib.save(f'Results/{data_config_tag}_S_on_qPlane.tex')
            plt.close()

        ############  Plot H's for trajectory ##############################################################
        fig3, ax3 = plt.subplots(3,1)
        for i, (model, hamiltonian) in enumerate(zip(model_list,hamiltonian_list)):
            plotUtils.plot_H_of_traj(ref_model, ref_model.hamiltonian, trajectory[i], t_span, color=color[i], label=label_list[i],fig=fig3,ax=ax3[1],title='H_exact(traj_theta)' if i==0 else None)   
            if '_NN' in label_list[i]:
                continue
            plotUtils.plot_H_of_traj(model, hamiltonian, trajectory[0], t_span, color=color[i], label=None,fig=fig3,ax=ax3[0],title='H_theta(traj_exact)' if i==0 else None)   
            plotUtils.plot_H_of_traj(model, hamiltonian, trajectory[i], t_span, color=color[i], label=None,fig=fig3,ax=ax3[2],title='H_theta(traj_theta)' if i==0 else None) 

        if save_tikz:
            tikzplotlib.save(f'Results/{data_config_tag}_H.tex')
            plt.close()

        # ############  Plot conserved quantity ##############################################################
        fig3, ax3 = plt.subplots(1,1)
        for i, (model, hamiltonian) in enumerate(zip(model_list,hamiltonian_list)):
            plotUtils.plot_conserved_quantity(model_list[0],trajectory[i], t_span, color= color[i],label=f'I(v_exact,traj_{label_list[i]})', fig=fig3, ax=ax3)
            if 'SymHnn' in label_list[i]:
                plotUtils.plot_conserved_quantity(model_list[i],trajectory[i], t_span, color= color[i],linestyle='--',label='I(v_SymHNN,traj_SymHNN)', fig=fig3, ax=ax3)
                plotUtils.plot_conserved_quantity(model_list[i],trajectory[0], t_span, color= color[i],linestyle='-.',label=f'I(v_SymHNN,traj_exact)', fig=fig3, ax=ax3)
        
        if save_tikz:
            tikzplotlib.save(f'Results/{data_config_tag}_ConservedQuantity.tex')
            plt.close()

        # ############  Print/Save learned symmetry ##############################################################
        print(f'SymHnn learned translation vector: {model_list[-1].translation_factor},\n learned matrix {model_list[-1].rotation_factor}')
        if save_tikz:
            with open(f'Results/{data_config_tag}_Symmetries.txt', 'w') as f:
                print(f'{data_config_tag} SymHnn learned translation vector: {model_list[-1].translation_factor},\n learned matrix {model_list[-1].rotation_factor}',file=f)

        ##############  Plot l_sym Term on a grid zero p############################################################
        fig4, ax4 = plt.subplots(2,3)
        fig5, ax5 = plt.subplots()
        fig6, ax6 = plt.subplots()
        n=1000
        grid = ref_model.generate_random_inital_value(quantity=n,region_tag='Evaluation')
        df_v_hat, loss_v_hat = evalUtils.l_sym_term_for_model_on_grid(grid, dH=model_list[0].ode, model=model_list[0], model_tag=f'v_hat({label_list[0]},v_exact)',symmetry=(ref_model.rotation_factor[0], ref_model.translation_factor[0]), dimq=ref_model.dimq)
        plotUtils.plot_l_sym_overq(df_v_hat,fig=fig4,ax=ax4[0,0],title='v_hat(H_exact,v_exact)\n{ref_model.rotation_factor[0]}\n{ref_model.translation_factor[0]}')
        plotUtils.plot_l_sym_over_points(df_v_hat, color=color[0],fig=fig5, ax=ax5,title=None,label=f'v_hat({label_list[0]},v_theta)')
        print(f'v_hat_norm(H_exact,v_exact)={loss_v_hat}')

        df_all_v_hats = pd.DataFrame()
        fig_index = 1 
        for i, (model,label) in enumerate(zip(model_list[1:],label_list[1:])):
            if 'SymHnn' in label:
                df_v_hat, loss_v_hat = evalUtils.l_sym_term_for_model_on_grid(grid, dH=model.output, model=model, model_tag=f'v_hat({label},v_exact)', symmetry=(ref_model.rotation_factor[0], ref_model.translation_factor[0]), dimq=ref_model.dimq)
                plotUtils.plot_l_sym_overq(df_v_hat,fig=fig4,ax=ax4[0,fig_index],title=f'{label}\nv_hat(H_theta,v_exact)\n{ref_model.rotation_factor[0]}\n{ref_model.translation_factor[0]}')
                plotUtils.plot_l_sym_over_points(df_v_hat, color=color[i+1],fig=fig5, ax=ax5,title=None,label=f'v_hat({label},v_exact)')
                print(f'{label}:\tloss_v_hat(H_theta,v_exact)={loss_v_hat}')
                df_all_v_hats = pd.concat([df_all_v_hats,df_v_hat])             

                df_v_hat, loss_v_hat = evalUtils.l_sym_term_for_model_on_grid(grid, dH=model.output, model=model, model_tag=f'v_hat({label},v_theta)', symmetry=(model.rotation_factor[0], model.translation_factor[0]), dimq=ref_model.dimq)
                plotUtils.plot_l_sym_overq(df_v_hat, fig=fig4,ax=ax4[1,fig_index],title=f'v_hat(H_theta,v_theta)\n{model.rotation_factor[0]}\n{model.translation_factor[0]}')
                plotUtils.plot_l_sym_over_points(df_v_hat, color=tuple(np.min([item + 0.5,1]) for item in color[i+1]),fig=fig5, ax=ax5,title=None,label=f'v_hat({label},v_theta)')
                print(f'{label}:\tloss_v_hat(H_theta,v_theta)={loss_v_hat}')
                df_all_v_hats = pd.concat([df_all_v_hats,df_v_hat])     
                fig_index += 1
            elif 'HNN' in label:
                df_v_hat, loss_v_hat = evalUtils.l_sym_term_for_model_on_grid(grid, dH=model.output, model=model, model_tag=f'v_hat({label},v_exact)', symmetry=(ref_model.rotation_factor[0], ref_model.translation_factor[0]), dimq=ref_model.dimq)
                plotUtils.plot_l_sym_overq(df_v_hat, fig=fig4,ax=ax4[0,fig_index],title=f'{label}\nv_hat(H_theta,v_exact)\n{ref_model.rotation_factor[0]}\n{ref_model.translation_factor[0]}')
                plotUtils.plot_l_sym_over_points(df_v_hat, color=color[i+1],fig=fig5, ax=ax5,title=None,label=f'v_hat({label},v_exact)')
                print(f'{label}:\tloss_v_hat(H_theta,v_exact)={loss_v_hat}')
                df_all_v_hats = pd.concat([df_all_v_hats,df_v_hat])     
                fig_index +=1

        sn.violinplot(data=df_all_v_hats,x='Net',y='v_hat',hue='Net',split=False,inner='quartile',ax=ax6)
        ax5.legend()

        if save_tikz:
            tikzplotlib.save(f'Results/{data_config_tag}_v_hat_violine.tex')
            plt.close()
            tikzplotlib.save(f'Results/{data_config_tag}_v_hat_over_points.tex')
            plt.close()
            tikzplotlib.save(f'Results/{data_config_tag}_v_hat_over_q.tex')
            plt.close()

        ############## Lsym for trajectory ########################
        fig7, ax7 = plt.subplots(2,3)
        df_v_hat, loss_v_hat = evalUtils.l_sym_term_for_model_on_grid(trajectory[0].T, dH=model_list[0].ode, model=model_list[0], model_tag=f'v_hat({label_list[0]},v_exact)',symmetry=(ref_model.rotation_factor[0], ref_model.translation_factor[0]), dimq=ref_model.dimq)
        plotUtils.plot_l_sym_overq(df_v_hat,fig=fig7,ax=ax7[0,0],title=f'v_hat(H_exact,v_exact)\n{ref_model.rotation_factor[0]}\n{ref_model.translation_factor[0]}')
        fig_index = 1

        for i, (model,label,traj) in enumerate(zip(model_list[1:],label_list[1:],trajectory[1:])):
            if 'SymHnn' in label:
                df_v_hat, loss_v_hat = evalUtils.l_sym_term_for_model_on_grid(traj.T, dH=model.output, model=model, model_tag=f'v_hat({label},v_exact)', symmetry=(ref_model.rotation_factor[0], ref_model.translation_factor[0]), dimq=ref_model.dimq)
                plotUtils.plot_l_sym_overq(df_v_hat,fig=fig7,ax=ax7[0,fig_index],title=f'{label}\nv_hat(H_theta,v_exact)\n{ref_model.rotation_factor[0]}\n{ref_model.translation_factor[0]}')   

                df_v_hat, loss_v_hat = evalUtils.l_sym_term_for_model_on_grid(traj.T, dH=model.output, model=model, model_tag=f'v_hat({label},v_theta)', symmetry=(model.rotation_factor[0], model.translation_factor[0]), dimq=ref_model.dimq)
                plotUtils.plot_l_sym_overq(df_v_hat, fig=fig7,ax=ax7[1,fig_index],title=f'v_hat(H_theta,v_theta)\n{model.rotation_factor[0].detach().numpy()}\n{model.translation_factor[0].detach().numpy()}')   
                fig_index += 1
            elif 'HNN' in label:
                df_v_hat, loss_v_hat = evalUtils.l_sym_term_for_model_on_grid(traj.T, dH=model.output, model=model, model_tag=f'v_hat({label},v_exact)', symmetry=(ref_model.rotation_factor[0], ref_model.translation_factor[0]), dimq=ref_model.dimq)
                plotUtils.plot_l_sym_overq(df_v_hat, fig=fig7,ax=ax7[0,fig_index],title=f'{label}\nv_hat(H_theta,v_exact)\n{ref_model.rotation_factor[0]}\n{ref_model.translation_factor[0]}')
                fig_index +=1

        fig7.suptitle('L_sym Term for evaulated trajectories')
        plt.show()

        if save_tikz:
            tikzplotlib.save(f'Results/{data_config_tag}_v_hat_of_traj.tex')
            plt.close()