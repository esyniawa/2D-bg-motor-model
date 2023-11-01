import matplotlib.pyplot as plt
import numpy as np


def load_monitors(simID, layer, load_VA_data=False, load_PM_data=False):
    from parameters import create_results_folder

    results_folder = create_results_folder(simID)
    sub_folder = f'Layer[{layer}]/'

    monitors = {}

    if load_VA_data:
        try:
            monitors['rVA'] = np.load(results_folder + sub_folder + f'rVA.npy')
        except:
            print('VA data is missing.')
            monitors['rVA'] = None

    if load_PM_data:
        try:
            monitors['rPM'] = np.load(results_folder + sub_folder + f'rPM.npy')
        except:
            print('PM data is missing.')
            monitors['rPM'] = None

    try:
        monitors['rStrD1'] = np.load(results_folder + sub_folder + f'rlatStrD1.npy')
    except:
        print('Lateral striatum D1 data is missing.')
        monitors['rStrD1'] = None

    try:
        monitors['rSNr'] = np.load(results_folder + sub_folder + f'rSNr.npy')
    except:
        print('SNr data is missing.')
        monitors['rSNr'] = None

    try:
        monitors['rVL'] = np.load(results_folder + sub_folder + f'rVL.npy')
    except:
        print('VL data is missing.')
        monitors['rVL'] = None

    try:
        monitors['rM1'] = np.load(results_folder + sub_folder + f'rM1.npy')
    except:
        print('M1 data is missing.')
        monitors['rM1'] = None

    try:
        monitors['rSNc'] = np.load(results_folder + sub_folder + f'rlatSNc.npy')
    except:
        print('Lateral SNc data is missing.')
        monitors['rSNc'] = None

    return monitors


def load_weights(simID, layer):
    from parameters import create_results_folder

    results_folder = create_results_folder(simID)
    sub_folder = f'Layer[{layer}]/'

    weights = {}
    try:
        weights['wCortexD1'] = np.load(results_folder + sub_folder + f'wproj2.npy')
    except:
        print("Weights from Cortex to D1 is missing.")
        weights['wCortexD1'] = None

    try:
        weights['DAprediction'] = np.load(results_folder + sub_folder + f'wproj10.npy')
    except:
        print("Weights from Cortex to D1 is missing.")
        weights['DAprediction'] = None

    return weights


if __name__ == '__main__':
    import sys, os
    from parameters import model_params

    simID = int(sys.argv[1])
    num_layers = model_params['num_init_positions']

    figures_folder = f'figures/train_motor_network[{simID}]/'
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)

    for layer in range(num_layers):
        monitors = load_monitors(simID, layer, load_PM_data=True)
        weights = load_weights(simID, layer)

        # plot monitors
        plt.figure(f'BG_goal[{layer}]', figsize=(50, 10))

        # Striatum
        l = plt.subplot(5, 1, 1)
        plt.title('StriatumD1')
        plt.plot(monitors['rStrD1'])
        # SNr
        l = plt.subplot(5, 1, 2)
        plt.title('SNr')
        plt.plot(monitors['rSNr'])
        # VA
        l = plt.subplot(5, 1, 3)
        plt.title('VL')
        plt.plot(monitors['rVL'])
        # PM
        l = plt.subplot(5, 1, 4)
        plt.title('M1')
        plt.plot(monitors['rM1'])
        # SNc
        l = plt.subplot(5, 1, 5)
        plt.title('SNc')
        plt.plot(monitors['rSNc'])

        plt.legend()
        plt.subplots_adjust(left=0.01, right=0.99)
        plt.savefig(figures_folder + f'BG_activities_goal[{layer}].pdf')
        plt.close()

        # plot weights
        fig = plt.figure(f'Weights_goal[{layer}]', figsize=(10, 10))

        # Cortex-Striatum
        ax = fig.add_subplot(1, 2, 1) 
        for movement in range(num_layers):
            lw_CorStrD1 = ax.plot(weights['wCortexD1'][:, movement, :])
        
        # DA Prediction Error
        ax = fig.add_subplot(1, 2, 2)
        lw_DA = ax.plot(weights['DAprediction'][:, 0, :])

        plt.savefig(figures_folder + f'BG_weights_goal[{layer}].pdf')
        plt.close()
