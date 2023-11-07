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

