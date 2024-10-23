'''
Competition URL: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
Author: Zhouzhou Shen
Reference: Dive into deep learning
'''

# Step0: encapsulate device getting function.
import torch 

def try_gpu(i = 0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")

def try_all_gpus():
    devices = [
        torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
    ]
    return devices if devices else [torch.device("cpu")]

# Step1: download data.
import os
import requests
import hashlib

DATA_HUB = dict()
DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"

def download(fname, cache_dir = os.path.join(".", "house_pred", "data")):
    assert fname in DATA_HUB, f"{fname} is not in DATA_HUB."
    os.makedirs(cache_dir, exist_ok = True)

    url, sha1_value = DATA_HUB[fname]
    fpath = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(fpath):
        sha1 = hashlib.sha1()
        with open(fpath, "rb") as f:
            while True:
                data = f.read(65536)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_value:
            return fpath

    print(f"Downloading \"{fname}\"...")
    r = requests.get(url, stream = True, verify = True)
    with open(fpath, "wb") as f:
        f.write(r.content)
    return fpath

DATA_HUB["kaggle_house_pred_train"] = (
    DATA_URL + "kaggle_house_pred_train.csv",
    "585e9cc93e70b39160e7921475f9bcd7d31219ce"
)
DATA_HUB["kaggle_house_pred_test"] = (
    DATA_URL + "kaggle_house_pred_test.csv",
    "fa19780a7b011d9b009e8bff8e99922a8ee2eb90"
)

# Step2: preprocess data.
import pandas as pd
import numpy as np

train_raw = pd.read_csv(download("kaggle_house_pred_train"))
test_raw = pd.read_csv(download("kaggle_house_pred_test"))

all_features = pd.concat((train_raw.iloc[:, 1:-1], test_raw.iloc[:, 1:]), axis = 0)
numeric_features = all_features.dtypes[all_features.dtypes != "object"].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / x.std()
)
all_features[numeric_features] = all_features[numeric_features].fillna(
    all_features[numeric_features].mean(), axis = 0
)
all_features = pd.get_dummies(all_features, dummy_na = True)
all_features = all_features.astype(float)

num_train = train_raw.shape[0]
train_features = torch.tensor(all_features[:num_train].values, dtype = torch.float32)
test_features = torch.tensor(all_features[num_train:].values, dtype = torch.float32)
train_labels = torch.tensor(train_raw.SalePrice.values.reshape((-1, 1)), dtype = torch.float32)

print("Data preproduction finished.")

# Step3: infer using NN.
from utils import d2l
from torch import nn
import time

# Step3.1: define net.
def get_net():
    net = nn.Sequential(nn.Linear(train_features.shape[1], 1))
    return net

def init_weight(layer):
    if type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean = 0, std = 0.01)

# Step3.2: define loss.
def get_loss():
    return nn.MSELoss()

def log_rmse(pred, labels):
    clipped_preds = torch.clamp(pred, 1, float("inf"))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

# Step3.3: define trainer.
def get_optim_func():
    return torch.optim.Adam

# Step3.4: define data, oh it's already generated in step 2.

# Step3.5: define train functions.
def to_gpu(*args):
    ret = []
    for arg in args:
        if arg is None:
            continue
        elif isinstance(arg, nn.Module):
            arg = arg.to(device = try_gpu(0))
        else:
            arg = arg.cuda(0)    
        ret.append(arg)
    return ret
        
def train(net, loss, optim_func, train_features, train_labels, test_features, test_labels,
          lr, weight_decay, batch_size, num_epochs):
    to_gpu_list = [net, train_features, train_labels, test_features, test_labels]
    if test_features is not None:
        net, train_features, train_labels, test_features, test_labels = to_gpu(*to_gpu_list)
    else:
        net, train_features, train_labels = to_gpu(*to_gpu_list)

    train_loss, test_loss = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = optim_func(net.parameters(), lr = lr, weight_decay = weight_decay)
    
    st_time = time.time()
    for _ in range(num_epochs):
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        net.eval()
        train_loss.append(log_rmse(net(train_features), train_labels))
        if test_features is not None:
            test_loss.append(log_rmse(net(test_features), test_labels))
    ed_time = time.time()
    print(f"It takes {(ed_time - st_time):.5f}s to train {num_epochs} epochs.")
    
    return train_loss, test_loss

# Use k fold validation to select hyperparameters. K must satisfy k | X.shape[0].
def get_k_fold_data(k, i, X, y):
    assert k > 1, "K must be larger than 1!"
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx_range = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx_range, :], y[idx_range]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], axis = 0)
            y_train = torch.cat([y_train, y_part], axis = 0)            
    return X_train, y_train, X_valid, y_valid

def k_fold_valid(k, net, loss, optim_func, train_features, train_labels,
                 lr, weight_decay, batch_size, num_epochs):
    train_l_sum, valid_l_sum = 0.0, 0.0
    for i in range(k):
        X_train, y_train, X_valid, y_valid =\
        get_k_fold_data(k, i, train_features, train_labels)
        train_loss, test_loss = train(net, loss, optim_func, X_train, y_train, X_valid, y_valid,
                                      lr, weight_decay, batch_size, num_epochs)
        train_l_sum += train_loss[-1]
        valid_l_sum += test_loss[-1]
        print(f"Fold {i}, train loss is {train_loss[-1]:.5f}, validation loss is {test_loss[-1]:.5f}")
    print(f"Average train loss is {(train_l_sum / k):.5f}. Average validation loss is {(valid_l_sum / k):.5f}")

def train_and_pred(net, loss, optim_func, train_features, train_labels, test_features,
                   lr, weight_decay, batch_size, num_epochs):
    train_ls, _ = train(net, loss, optim_func, train_features, train_labels, None, None,
                        lr, weight_decay, batch_size, num_epochs)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel = "epoch",
             ylabel = "log rmse", xlim = [1, num_epochs], yscale = "log")
    print(f"Train log rmse: {float(train_ls[-1]):.5f}")
    
    net.to(try_gpu(0))
    net.eval()
    test_features = test_features.cuda(0)
    preds = net(test_features).cpu().detach().numpy()
    test_raw["SalePrice"] = pd.Series(preds.reshape((1, -1))[0])
    submission = pd.concat([test_raw["Id"], test_raw["SalePrice"]], axis = 1)
    submission.to_csv("./house_pred/submission.csv", index = False)

# Step4: write script to run.
lr = 1
weight_decay = 0.001
batch_size = 64
num_epochs = 500
k = 5

net = get_net()
net.apply(init_weight)
loss = get_loss()
optim_func = get_optim_func()

k_fold_valid(k, net, loss, optim_func, train_features, train_labels,
             lr, weight_decay, batch_size, num_epochs)

# Use a new net to predict.
net = get_net()
net.apply(init_weight)
train_and_pred(net, loss, optim_func, train_features, train_labels, test_features,
               lr, weight_decay, batch_size, num_epochs)
d2l.plt.show()