import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def load_monitors(simID, bodypos, data_type):
    from parameters import create_results_folder

    if data_type == 'r' or data_type == 'w':
        folder, _ = create_results_folder(simID)
        sub_folder = f'Layer[{bodypos}]/'

        monitors = {}

        # get all rates
        files = [x for x in os.listdir(folder + sub_folder) if x.startswith(data_type)]
        for file in files:
            monitors[file.rsplit('.', maxsplit=1)[0]] = np.load(folder + sub_folder + file)

        return monitors
    else:
        raise AttributeError


def plot_overview(simID, bodypos, figure_size=(25,15), save_name=None):
    from parameters import model_params
    from functions import create_state_space

    y_dim, x_dim, _ = create_state_space().shape
    tInit = 0

    rates = load_monitors(simID=simID, bodypos=bodypos, data_type='r')

    # plot populations in subplot row wise
    fig = plt.figure(f'rates_BG_goal[{bodypos}]', figsize=figure_size)

    # PM
    nt, _ = rates['rPM'].shape
    rates['rPM'] = rates['rPM'].reshape((nt, model_params['num_init_positions'], y_dim, x_dim))

    # dPFC
    ax2 = fig.add_subplot(4, 3, 1)
    plt.title('dPFC', fontsize=10)
    lPFC = ax2.bar(x=[str(x) for x in range(model_params['num_goals'])], height=rates['rdPFC'][tInit, :], width=.5)
    ax2.set_ylim(0.0, 2.0)

    lVA = []
    lM1 = []
    lPM = []

    rates['rVA'] = rates['rVA'].reshape((nt, model_params['num_init_positions'], model_params['num_goals']))
    rates['rM1'] = rates['rM1'].reshape((nt, model_params['num_init_positions'], model_params['num_trajectories']))

    for layer in range(model_params['num_init_positions']):
        ax1 = fig.add_subplot(4, 3, 4 + (layer*3))
        plt.title(f'PM_layer[{layer}]', fontsize=8)
        lPM.append(ax1.imshow(rates['rPM'][tInit, bodypos, :, :], vmin=0, vmax=2.0, cmap='Blues', origin='lower'))
        plt.xticks([])
        plt.yticks([])

        ax3 = fig.add_subplot(3, 3, 2 + (layer*3))
        plt.title(f'VA_layer[{layer}]', fontsize=8)
        lVA.append(ax3.bar(x=[str(x) for x in range(model_params['num_goals'])], height=rates['rVA'][tInit, layer, :], width=.6))
        ax3.set_ylim(0.0, 4.0)

        # M1
        ax4 = fig.add_subplot(3, 3, 3 + (layer*3))
        plt.title(f'M1_layer[{layer}]', fontsize=8)
        lM1.append(ax4.bar(x=[str(x) for x in range(model_params['num_trajectories'])], height=rates['rM1'][tInit, layer, :]))
        ax4.set_ylim(0.0, 2.0)

    axslider = plt.axes([0.25, 0.05, 0.5, 0.03])
    time_slider = Slider(
        ax=axslider,
        label='Time [ms]',
        valmin=0,
        valmax=nt-1,
        valinit=0,
    )

    def update(val):
        t = int(time_slider.val)
        for i, plot in enumerate(lPM):
            plot.set_data(rates['rPM'][t, i, :, :])

        for i, bar in enumerate(lPFC):
            bar.set_height(rates['rdPFC'][t, i])

        for i, plot in enumerate(lVA):
            for j, bar in enumerate(plot):
                bar.set_height(rates['rVA'][t, i, j])

        for i, plot in enumerate(lM1):
            for j, bar in enumerate(plot):
                bar.set_height(rates['rM1'][t, i, j])

    time_slider.on_changed(update)

    plt.show()

if __name__ == '__main__':
    from functions import goal_coordinates, create_state_space, return_goal_indeces

    simID = 9
    plot_overview(simID, 2)
