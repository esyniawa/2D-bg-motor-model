import matplotlib.pyplot as plt
import numpy as np


def load_monitors(simID, load_VA_data=False, load_PM_data=False):
    from parameters import create_results_folder, model_params

    results_folder, _ = create_results_folder(simID)
    num_layers = model_params['num_init_positions']

    monitors = {}

    for layer in range(num_layers):

        sub_folder = f'Layer[{layer}]/'

        if load_VA_data:
            try:
                monitors[f'rVA_{layer}'] = np.load(results_folder + sub_folder + f'rVA.npy')
            except:
                print(f'VA data for layer {layer} is missing.')
                monitors['rVA_{layer}'] = None

        if load_PM_data:
            try:
                monitors[f'rPM_{layer}'] = np.load(results_folder + sub_folder + f'rPM.npy')
            except:
                print(f'PM data for layer {layer} is missing.')
                monitors[f'rPM_{layer}'] = None

        try:
            monitors[f'rStrD1_{layer}'] = np.load(results_folder + sub_folder + f'rStrD1_putamen.npy')
        except:
            print(f'Lateral striatum D1 data for layer {layer} is missing.')
            monitors[f'rStrD1_{layer}'] = None

        try:
            monitors[f'rSNr_{layer}'] = np.load(results_folder + sub_folder + f'rSNr.npy')
        except:
            print(f'SNr data for layer {layer} is missing.')
            monitors[f'rSNr_{layer}'] = None

        try:
            monitors[f'rVL_{layer}'] = np.load(results_folder + sub_folder + f'rVL.npy')
        except:
            print(f'VL data for layer {layer} is missing.')
            monitors[f'rVL_{layer}'] = None

        try:
            monitors[f'rM1'] = np.load(results_folder + sub_folder + f'rM1.npy')
        except:
            print(f'M1 data for layer {layer} is missing.')
            monitors[f'rM1_{layer}'] = None

        try:
            monitors[f'rSNc_{layer}'] = np.load(results_folder + sub_folder + f'rSNc_put.npy')
        except:
            print(f'Lateral SNc data for layer {layer} is missing.')
            monitors[f'rSNc_{layer}'] = None

    return monitors


def load_weights(simID):
    from parameters import create_results_folder, model_params

    results_folder, _ = create_results_folder(simID)

    num_layers = model_params['num_init_positions']

    weights = {}

    for layer in range(num_layers):
        sub_folder = f'Layer[{layer}]/'
        try:
            weights[f'wPMD1_{layer}'] = np.load(results_folder + sub_folder + f'wPMStrD1_put_{layer}.npy')
        except:
            print("Weights from PM to D1 is missing.")
            weights[f'wPMD1_{layer}'] = None

        try:
            weights[f'DAprediction_{layer}'] = np.load(results_folder + sub_folder + f'wStrD1SNc_put.npy')
        except:
            print("Weights from D1 to SNc is missing.")
            weights[f'DAprediction_{layer}'] = None

    return weights


if __name__ == '__main__':
    import os
    from parameters import model_params

    simID = 1

    figures_folder = f'figures/train_motor_network[{simID}]/'
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)

    monitors = load_monitors(simID, load_PM_data=False)
    weights = load_weights(simID)

    for layer in range(model_params['num_init_positions']):

        # plot monitors
        plt.figure(f'BG_goal[{layer}]', figsize=(50, 10))

        # Striatum
        l = plt.subplot(5, 1, 1)
        plt.title('StriatumD1')
        plt.plot(monitors[f'rStrD1_{layer}'])
        # SNr
        l = plt.subplot(5, 1, 2)
        plt.title('SNr')
        plt.plot(monitors[f'rSNr_{layer}'])
        # VA
        l = plt.subplot(5, 1, 3)
        plt.title('VL')
        plt.plot(monitors[f'rVL_{layer}'])
        # PM
        l = plt.subplot(5, 1, 4)
        plt.title('M1')
        plt.plot(monitors[f'rM1_{layer}'])
        # SNc
        l = plt.subplot(5, 1, 5)
        plt.title('SNc')
        plt.plot(monitors[f'rSNc_{layer}'])

        plt.legend()
        plt.subplots_adjust(left=0.01, right=0.99)
        plt.savefig(figures_folder + f'BG_activities_train_dl[{layer}].pdf')
        plt.close()

        # plot weights
        fig = plt.figure(f'Weights_goal[{layer}]', figsize=(10, 10))

        # Cortex-Striatum
        ax = fig.add_subplot(1, 2, 1)
        for movement in range(model_params['num_trajectories']):
            lw_CorStrD1 = ax.plot(weights[f'wPMD1_{layer}'][:, movement, :])
        
        # DA Prediction Error
        ax = fig.add_subplot(1, 2, 2)
        lw_DA = ax.plot(weights[f'DAprediction'][:, 0, :])

        plt.savefig(figures_folder + f'BG_weights_train_dl[{layer}].pdf')
        plt.close()
