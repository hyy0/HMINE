import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

class HB_MINE(nn.Module):
    def __init__(self, NP, hidden_size=100):
        super(HB_MINE, self).__init__()
        self.nets = nn.Sequential(
            nn.Flatten(),
            nn.Linear(NP, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )


    def preprocess(self, x):
        x = torch.Tensor(x)
        y = torch.zeros_like(x)
        for _ in range(x.shape[1]):
            idx = torch.randperm(x.shape[0])
            y[:, _] = x[idx, _]

        return x, y

    def forward(self, x):
        # preprocess
        joint, marginal = self.preprocess(x)

        pred_joint = self.nets(joint)
        pred_marginal = self.nets(marginal)
        
        loss = - torch.mean(pred_joint) + torch.log(torch.mean(torch.exp(pred_marginal)))
        
        return loss

# true mutual information
def mi_gauss(cov, dim=1):
    return -0.5 * np.log(np.linalg.det(cov)) *dim

def hb_mine(*variables, epochs = 500, plot_figure = False, true_mi = None):
    # data
    x = np.concatenate([*variables], axis=1)
    T,N,P = x.shape
    # model
    model = HB_MINE(NP=N * P, hidden_size=100)  # .to(device)20
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # train
    MI = []
    for _ in range(epochs):
        loss = model(x)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        MI.append(-loss.item())

    
    mis_smooth = pd.Series(MI).ewm(span=100).mean()

    # plot
    if plot_figure:
        fig, ax = plt.subplots()
        ax.plot(range(len(MI)), MI, alpha=0.4)
        ax.plot(range(len(MI)), mis_smooth, label='MINE Estimation')
        if not true_mi is  None:
            ax.plot(range(len(MI)), [true_mi]*len(MI), label='True Mutual Information')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Mutual Information')
        ax.legend(loc='best')
        plt.show()
    return np.mean(mis_smooth[-10:])



if __name__ == '__main__':
    # dimension
    T = 5000
    P = 1

    # data
    mean = [0, 0, 0]
    cov = [[1, 0.7, 0.3],
        [0.7, 1, 0.4],
        [0.3, 0.4, 1],]
    # cov = [[1, 0, 0],
    #         [0., 1, 0.],
    #         [0., 0., 1],]
    data = np.random.multivariate_normal(mean=mean, cov=cov, size=[T, P]).transpose([0, 2, 1])
    x = torch.tensor(data[:, 0], dtype=torch.float32).unsqueeze(dim=1)
    y = torch.tensor(data[:, 1], dtype=torch.float32).unsqueeze(dim=1)
    z = torch.tensor(data[:, 2], dtype=torch.float32).unsqueeze(dim=1)

    # record time
    Time = time.time()
    true_mi = mi_gauss(cov,dim=P)
    mi = hb_mine(x, y, z, plot_figure=True, true_mi=true_mi)
    print('MI_hat  == ', np.abs(mi))
    print('mi_true == ', true_mi, 'diff == ', np.abs(mi - true_mi))
    print('Total time ==', time.time() - Time)
