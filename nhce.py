# Author: Yanyan He, Mingyu Kang
# Date: 2024-10-1
# Description: This code implements the Neural Higher-order Causal Entropy Algorithm for inferring causal relationships between variables.


import os
import time
import numpy as np
import pandas as pd
import netCDF4 as nc
import multiprocessing as mp

from config import args
from mci_test import mci_test
from utils import *
tmp = []
def CEMCI(i, samples, max_tau, shared_q):

    T, N, P = samples.shape
    Ni = []  # Ni set, the set of real causal parents

    # discover causal parents
    samples_k = samples[max_tau - 1: T - 1]  # the samples of other nodes
    samples_i = samples[max_tau:, i].reshape([T - max_tau, 1, -1])  # the samples of node i
    for tau in range(1, max_tau + 1):
        if tau > 1:
            samples_k = np.concatenate([samples_k, samples[max_tau - tau: T - tau]], axis=1)
        samples_ki = np.concatenate([samples_k, samples_i], axis=1)

        K = Ni + [_ + N * (tau - 1) for _ in range(N)]
        I = N * tau
        
        # discover causal parents
        for j in K[-N:]:
            # conditional set K-j
            Kj = [_ for _ in K if _ != j]
            mci, p = mci_test(
                samples=samples_ki,
                i=I,
                j=j,
                K=Kj,
                alpha=args.alpha
            )

            if mci:
                K.remove(j)
            else:
                Ni.append(j)
    samples_ki = np.concatenate([samples_k, samples_i], axis=1)
    Nii = []

    for j in Ni:
        K = [_ for _ in Ni if _ != j]

        mci, p = mci_test(
            samples=samples_ki,
            i=samples_ki.shape[1] - 1,
            j=j,
            K=K,
            alpha=args.beta
        )
        
        
        if not mci:
            Nii.append(j)
            # print('-'*20)
            # print(f'i = {i} j = {j}')
            a = samples_ki[:, samples_ki.shape[1] - 1]
            b = samples_ki[:, j]
    # print(f'i & Nii {i}, {Nii}')
    shared_q.put([i, Nii])


def NHCE(samples):
    """
    Neural Higher-order Causal Entropy Algorithm,

    :param samples: the samples of all nodes
    :return: causal parents at multiple time lags
    """

    #
    N = samples.shape[1]  # the number of all nodes
    causal_network = np.zeros([N, N, args.max_tau])  

    results = mp.Manager().Queue()

    mp_pool = mp.Pool(processes=os.cpu_count() // 4 * 3)
    for i in range(N):
        mp_pool.apply_async(CEMCI, args=(i, samples, args.max_tau, results))

    mp_pool.close()
    mp_pool.join()

    while not results.empty():
        result = results.get()

        for _, node in enumerate(result[1]):
            causal_network[node % N, result[0], node // N] = 1
    return causal_network


def print_parents(parents):
    for i in range(parents.shape[1]):
        print('\nnode', i, ':')
        for j in range(parents.shape[0]):
            time_lags = [str(-t - 1) + ' ' for t, _ in enumerate(parents[j, i]) if _]
            if time_lags:
                print(j, '->', i, ':', ''.join(time_lags))



if __name__ == '__main__':
    print(f'Args: {args}')

    t0 = time.time()
    print(f'Loading Data...')

    files = os.listdir('/public/heyanyan/tmp/NHCE/data/')
    loop = len(files)
    tprs = np.zeros(shape=loop)
    fprs = np.zeros(shape=loop)
    files = sorted(files, key=lambda x: int(x.split('.')[0]))
    
    for l, file in enumerate(files):
        print(f'Running {file}.')
        data = nc.Dataset('/public/heyanyan/tmp/NHCE/data/' + file, 'r')
        samples, net = data['samples'][:], data['net'][:]
        print(samples.shape,net.shape)

        # if data dimension is 2, add a dimension on Feature Dimension
        # samples = np.expand_dims(samples, axis=2)
        causal_network = NHCE(samples)
        print_parents(causal_network)

        tprs[l] = recall(net, causal_network)
        fprs[l] = false_positive_rate(net, causal_network)
        print('recall:\t', tprs[l])
        print('false positive rate:\t', fprs[l])

    # save results
    if not os.path.exists('./results/'):
        os.mkdir('./results/')

    tpr_fpr = np.concatenate([tprs, fprs]).reshape([2, loop]).transpose([1, 0])
    pd.DataFrame(tpr_fpr).to_csv(args.save_path)

    # time
    print('Time cost (s) ==', time.time() - t0)
