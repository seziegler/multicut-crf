import numpy as np
import matplotlib.pyplot as plt

def visualize_cycles(cycle_counts_rounded, cycle_counts, n_epochs, save, file=None):

    invalids_rounded_mean = np.zeros(n_epochs,dtype=int)
    invalids_mean = np.zeros(n_epochs,dtype=int)

    for batch in range(len(cycle_counts_rounded[0])):
        invalids_rounded = []
        invalids = []
        for e in range(n_epochs):
            
            invalids_rounded.append(cycle_counts_rounded[e][batch][1])
            invalids.append(cycle_counts[e][batch][1])
            
        invalids_rounded_mean += np.array(invalids_rounded).reshape(-1) 
        invalids_mean += np.array(invalids).reshape(-1)

    invalids_rounded_mean = invalids_rounded_mean/len(cycle_counts_rounded[0])
    invalids_mean = invalids_mean/len(cycle_counts_rounded[0])

    plt.figure()
    plt.plot(invalids_rounded_mean, label='rounded')
    #plt.title('Invalid Cycles Count - Mean')
    plt.plot(invalids_mean, label='not rounded')
    plt.legend(loc="upper right")

    if save:

        plt.savefig(file)

