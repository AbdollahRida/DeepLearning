# Importing libs
import numpy as np
from numpy.random import random
import torch
#import plotting
from scipy.stats import bernoulli
from scipy.special import expit

# global variables
data_size = 30
num_epochs = 100
learning_rate = 1e-6
dtype = torch.FloatTensor

# generate random input data
x = random((data_size,2))

# generate labels corresponding to input data x
y = np.dot(x, [2., -3.]) + 1.
w_source = np.array([2., -3.])
b_source  = np.array([1.])

# generate Z
Z = bernoulli.rvs(expit(y))

# randomly initialize learnable weights and bias
w_init = random(2)
b_init = random(1)

w = w_init
b = b_init
#print("initial values of the parameters:", w, b )

# Tensorize everything
x_t = torch.from_numpy(x).type(dtype)
y_t = torch.from_numpy(y).type(dtype).unsqueeze(1)
Z_t = torch.from_numpy(Z).type(dtype).unsqueeze(1)
w_init_t = torch.from_numpy(w_init).type(dtype)
b_init_t = torch.from_numpy(b_init).type(dtype)

# Defining the model
model = torch.nn.Sequential(
    torch.nn.Linear(2, 1),
)

for m in model.children():
    m.weight.data = w_init_t.clone().unsqueeze(0)
    m.bias.data = b_init_t.clone()

# Using Binary Cross Entropy
loss_fn = torch.nn.BCEWithLogitsLoss()

# switch to train mode
model.train()

for epoch in range(num_epochs):
    Z_pred = model(x_t)
    loss = loss_fn(Z_pred, Z_t)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param.data -= learning_rate * param.grad

    print("progress:", "epoch:", epoch, "loss", loss.data.item())

# After training
print("estimation of the parameters:")
for param in model.parameters():
    print(param)

model = torch.nn.Sequential(
    torch.nn.Linear(2, 1),
)

for m in model.children():
    m.weight.data = w_init_t.clone().unsqueeze(0)
    m.bias.data = b_init_t.clone()

loss_fn = torch.nn.BCEWithLogitsLoss()

model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    Z_pred = model(x_t)
    loss = loss_fn(Z_pred, Z_t)
    print("progress:", "epoch:", epoch, "loss", loss.item())
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# After training
print("estimation of the parameters:")
for param in model.parameters():
    print(param)