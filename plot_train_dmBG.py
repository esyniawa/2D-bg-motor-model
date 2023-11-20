import numpy as np
import os


def load_monitors(simID, goal, data_type):
    from parameters import create_results_folder

    if data_type == 'r' or data_type == 'w':
        folder, _ = create_results_folder(simID)
        sub_folder = f'Goal[{goal}]/'

        monitors = {}

        # get all rates
        files = [x for x in os.listdir(folder + sub_folder) if x.startswith(data_type)]
        for file in files:
            monitors[file.rsplit('.', maxsplit=1)[0] + f'_{goal}'] = np.load(folder + sub_folder + file)

        return monitors
    else:
        raise AttributeError


def plot_rates_training_dm(simID, goal, populations, save_name=None):
    import matplotlib.pyplot as plt
    from parameters import create_results_folder

    rates = load_monitors(simID=simID, goal=goal, data_type='r')

    keys = []
    for population in populations:
        keys.extend([key for key in rates.keys() if population in key])

    # plot populations in subplot row wise
    plt.figure(f'rates_BG_goal[{goal}]', figsize=(50, 10))
    num_rows = len(keys)
    for row, key in enumerate(keys):
        plt.subplot(num_rows, 1, row+1)
        plt.title(key)
        plt.plot(rates[key])

    # save
    if save_name is not None:
        results_folder, _ = create_results_folder(simID)
        save_folder = 'figures/' + results_folder.split(os.sep)[1]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        plt.savefig(save_folder + '/' + save_name)

    plt.show()


def plot_w_training_dm(simID, goal, cons, save_name=None):
    import matplotlib.pyplot as plt
    from parameters import create_results_folder

    weights = load_monitors(simID=simID, goal=goal, data_type='w')

    keys = []
    for con in cons:
        keys.extend([key for key in weights.keys() if con in key])

    # plot populations in subplot row wise
    plt.figure(f'weights_BG_goal[{goal}]', figsize=(50, 10))
    num_rows = len(keys)
    for row, key in enumerate(keys):
        plt.subplot(num_rows, 1, row+1)
        plt.title(key)
        n_trials, n_post_cons, n_pre_cons = weights[key].shape
        for n_post_con in range(n_post_cons):
          plt.plot(weights[key][:, n_post_con, :])

    # save
    if save_name is not None:
        results_folder, _ = create_results_folder(simID)
        save_folder = 'figures/' + results_folder.split(os.sep)[1]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        plt.savefig(save_folder + '/' + save_name)

    plt.show()


def plot_w_matrix_training(simID, goal, save_name=None):
    import matplotlib.pyplot as plt
    from parameters import create_results_folder, model_params

    dimGP = list(model_params['dim_medial_BG'])
    dimPFC = model_params['num_goals']
    weights = load_monitors(simID, goal, 'w')

    direct_pathway_cons = ['PFCdStrD1', 'StrD1GPi']
    keys_direct = []
    for con in direct_pathway_cons:
        keys_direct.extend([key for key in weights.keys() if con in key])

    indirect_pathway_cons = ['PFCdStrD2', 'StrD2GPe']
    keys_indirect = []
    for con in indirect_pathway_cons:
        keys_indirect.extend([key for key in weights.keys() if con in key])

    # plotting
    timestamps = [4, 8, 12]
    nt = len(timestamps)

    # figure attributes
    minimum = 0.0
    maximum = 0.5
    fig = plt.figure(num=5, figsize=(15, 15))

    # norms
    norm_w_direct = np.matmul(np.transpose(weights[keys_direct[0]][0, :, :]),
                              np.transpose(weights[keys_direct[1]][0, :, :]))

    norm_w_indirect = np.matmul(np.transpose(weights[keys_indirect[0]][0, :, :]),
                                np.transpose(weights[keys_indirect[1]][0, :, :]))

    # reshape to (1, #body positions, #tactile points)
    for i, timestamp in enumerate(timestamps):

        w_direct = np.matmul(np.transpose(weights[keys_direct[0]][timestamp - 1, :, :]),
                             np.transpose(weights[keys_direct[1]][timestamp - 1, :, :])) - norm_w_direct

        w_indirect = np.matmul(np.transpose(weights[keys_indirect[0]][timestamp - 1, :, :]),
                               np.transpose(weights[keys_indirect[1]][timestamp - 1, :, :])) - norm_w_indirect

        # reshape to (1, #body positions, #tactile points)
        w_direct = w_direct.reshape([dimPFC] + dimGP)
        w_indirect = w_indirect.reshape([dimPFC] + dimGP)

        plt.figtext(0.07, 0.75 - 0.2725 * i, f"Trial = {timestamp}", fontsize=24, ha='center')

        # direct pathway
        plt.subplot(nt, 2, i * 2 + 1)
        pcm = plt.imshow(w_direct[goal, :, :], vmin=minimum, vmax=maximum, cmap='Purples')
        plt.ylabel('dPFC', fontsize=20)
        plt.xlabel('GPi', fontsize=20)
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.title('Direct pathway', fontsize=24)
        if i == nt - 1:
            ax = fig.gca()
            cax = ax.inset_axes([0.1, -0.3, 0.8, 0.1], transform=ax.transAxes)
            cbar = fig.colorbar(pcm, ax=ax, cax=cax, ticks=[minimum, maximum], orientation='horizontal')
            cbar.ax.tick_params(labelsize=18)
            cbar.set_label('$ \delta w $', labelpad=-15, fontsize=20, weight='bold')

        # plot lines to diffentiate the different body positions
        for j in range(dimGP[0]):
            plt.plot([- 0.5, -0.5 + dimGP[1]], [0.5 + j, 0.5 + j], color='gray', linewidth=1)

        # indirect pathway
        plt.subplot(nt, 2, i * 2 + 2)
        pcm = plt.imshow(w_indirect[goal, :, :], vmin=minimum, vmax=maximum, cmap='Purples')
        plt.ylabel('dPFC', fontsize=20)
        plt.xlabel('GPe', fontsize=20)
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.title('Indirect pathway', fontsize=24)
        if i == nt - 1:
            ax = fig.gca()
            cax = ax.inset_axes([0.1, -0.3, 0.8, 0.1], transform=ax.transAxes)
            cbar = fig.colorbar(pcm, ax=ax, cax=cax, ticks=[minimum, maximum], orientation='horizontal')
            cbar.ax.tick_params(labelsize=18)
            cbar.set_label('$ \delta w $', labelpad=-15, fontsize=20, weight='bold')

        # plot lines to diffentiate the different body positions
        for j in range(dimGP[0]):
            plt.plot([- 0.5, -0.5 + dimGP[1]], [0.5 + j, 0.5 + j], color='gray', linewidth=1)

    # save
    if save_name is not None:
        results_folder, _ = create_results_folder(simID)
        save_folder = 'figures/' + results_folder.split(os.sep)[1]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        plt.savefig(save_folder + '/' + save_name)

    plt.show()


if __name__ == '__main__':
    # plot_rates_training_dm(0, 0, ['PFC', 'StrD1', 'GPi', 'VA', 'SNc'], save_name='direct_Test_trial_complete_1.png')
    # plot_rates_training_dm(0, 0, ['PFC', 'StrD2', 'GPe', 'StrThal', 'VA', 'SNc'], save_name='indirect_Test_trial_complete_1.png')
    # plot_w_training_dm(0, 0, ['PFC', 'GPe', 'GPi'], save_name='Test_trial_wDA+_2.png')
    plot_w_matrix_training(0, 0, 'BG_learning.pdf')
