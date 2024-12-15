import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree


# true mutual information
def mi_gauss(cov):
    return -0.5 * np.log(np.linalg.det(cov))


def sample_batch(data, batchsize=100, mode='joint', knn_neighbors=10, knn_radius=1000):
    X = data[0]
    Y = data[1]
    Z = np.concatenate([*data[2:]], axis=1)
    T = X.shape[0]
    batch = None
    if mode == 'joint':
        index = np.random.choice(T, size=batchsize, replace=False)
        batch = np.concatenate([X[index], Y[index], Z[index]], axis=1)
    elif mode == 'prod':
        index = np.random.choice(T, size=batchsize // knn_neighbors, replace=False)

        index_rest = np.delete(np.arange(T), index)
        X_ = X[index_rest]
        Z_ = Z[index_rest]

        a, b = np.array(Z_).reshape(len(index_rest), -1), np.array(Z[index]).reshape(len(index), -1)
        kdt = KDTree(a, metric='euclidean')
        _, Neighbor_indices = kdt.query(b, k=knn_neighbors)

        index_x = Neighbor_indices.flatten()
        index_y = index_z = index.repeat(knn_neighbors)  # [1,2,3] -> [1,...,1,2,...,2,3,...,3]
        batch = np.concatenate([X_[index_x], Y[index_y], Z[index_z]], axis=1)

    return batch.reshape(batchsize, -1)


class HC_MINE(nn.Module):
    def __init__(self, input_size=3, hidden_size=100):
        super(HC_MINE, self).__init__()
        self.nets = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, joint, prod):
        pred_joint = self.nets(joint)
        pred_prod = self.nets(prod)

        pred_joint = torch.squeeze(pred_joint)
        pred_prod = torch.squeeze(pred_prod)

        second_term = torch.exp(pred_prod)
        loss = - torch.mean(pred_joint) + torch.log(torch.mean(second_term))

        return loss


def hc_mine(data, epochs=1000, plot_figure=False, mi_true=None):
    input_size = sum([_.shape[1] for _ in data])
    if data[0].ndim > 2:
        input_size = input_size * data[0].shape[2]

    # model
    model = HC_MINE(input_size=input_size, hidden_size=300)  # 150,100,300
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 0.001

    # train
    MI = []
    for _ in range(epochs):
        joint = torch.FloatTensor(sample_batch(data))
        prod = torch.FloatTensor(sample_batch(data, mode='prod'))
        loss = model(joint, prod)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        MI.append(-loss.item())

    mis_smooth = pd.Series(MI).ewm(span=100).mean()

    if plot_figure:
        fig, ax = plt.subplots()
        plt.plot(range(len(MI)), MI, alpha=0.4)
        plt.plot(range(len(MI)), mis_smooth, label='CMI Estimation')
        if not mi_true is None:
            plt.plot(range(len(MI)), [mi_true] * len(MI), label='True Mutual Information')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Mutual Information')
        ax.legend(loc='best')
        plt.show()
    del model
    return np.mean(mis_smooth.values[-10:])



# """
#    ✔， 测试4变量
# """
# if __name__ == '__main__':
#     # dimension
#     T = 10000
#     # N = 4
#     P = 1
#
#     # data
#     mean = [0, 0, 0, 0]
#     cov = np.array(
#         [[1, 0.7, 0.3, 0.4],
#          [0.7, 1, 0.8, 0.5],
#          [0.3, 0.8, 1, 0.6],
#          [0.4, 0.5, 0.6, 1]]
#     )
#     data = np.random.multivariate_normal(mean=mean, cov=cov, size=[T, P]).transpose([0, 2, 1])
#     x = torch.tensor(data[:, 0], dtype=torch.float32).unsqueeze(dim=1)
#     y = torch.tensor(data[:, 1], dtype=torch.float32).unsqueeze(dim=1)
#     z = torch.tensor(data[:, 2], dtype=torch.float32).unsqueeze(dim=1)
#     w = torch.tensor(data[:, 3], dtype=torch.float32).unsqueeze(dim=1)
#     print(x.shape, y.shape, z.shape)
#
#     # record time
#     mi_true = mi_gauss(cov) - mi_gauss(cov[[0, 2, 3]][:, [0, 2, 3]]) - mi_gauss(cov[[1, 2, 3]][:, [1, 2, 3]]) \
#               + mi_gauss(cov[[2, 3]][:, [2, 3]])
#     print('mi_true ==', mi_true)
#
#     Time = time.time()
#     cmi = mi_chmine([x, y, z, w], plot_figure=True, mi_true=mi_true)
#     print('mi_chmine ==', cmi, 'diff ==', abs(cmi - mi_true))
#     print('Total time ==', time.time() - Time)
