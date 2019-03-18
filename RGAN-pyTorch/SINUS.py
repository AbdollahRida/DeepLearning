import numpy as np
import torch

T = 20
L = 1000
N = 100

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float64')
torch.save(data, open('traindata.pt', 'wb'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

use_gpu = torch.cuda.is_available()
#use_gpu = False
def gpu(tensor, gpu=use_gpu):
    if gpu:
        return tensor.cuda()
    else:
        return tensor

z_dim = 1028
hidden_dim = 51

class GaussianNoise(nn.Module):
    def __init__(self, stddev = 0.1):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return (din + torch.autograd.Variable(torch.randn(din.size()).cuda().type(torch.DoubleTensor) * self.stddev)).type(torch.DoubleTensor)
        return din.type(torch.DoubleTensor)

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()

        self.lstm_G = nn.LSTM(input_size = z_dim, 
           hidden_size = 600,
           num_layers = 2,
           bias = False)

        self.relu = nn.ReLU()

        self.fc_G = nn.Linear(600, 999)

    def forward(self, x):
        x = x.unsqueeze(1)
        #print(x.type())
        #print(x.size())
        output, _ = self.lstm_G(x)
        output = torch.tanh(self.fc_G(self.relu(output)))
        return output

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()

        self.lstm_D = nn.LSTM(input_size = 999, 
                               hidden_size = 600,
                               num_layers = 2)
        
        self.relu = nn.ReLU()

        self.MLP = nn.Linear(600, 1)
        
        self.noise = GaussianNoise(.1)

    def forward(self, x):
        #print(x.size())
        #print(x.size())
        #x = self.noise(x)
        #x = self.bn(x)
        #x = x.unsqueeze(1)
        outputs, _ = self.lstm_D(x)
        #print(outputs.size())
        res = self.MLP(self.relu(outputs[:, -1, :]))
        #print(res)
        y_data = torch.sigmoid(res.narrow(0, 0, x[0].shape[0]))
        return y_data

from tqdm import tqdm

# set random seed to 0
np.random.seed(0)
torch.manual_seed(0)

batch_size = 1
lr = 1e-4
nb_epochs = 100

# load data and make training set
data = torch.load('traindata.pt')
input = gpu(torch.from_numpy(data[3:, :-1]))
target = gpu(torch.from_numpy(data[3:, 1:]))
test_input = gpu(torch.from_numpy(data[:3, :-1]))
test_target = gpu(torch.from_numpy(data[:3, 1:]))

# build the model
net_G = gpu(generator())
net_D = gpu(discriminator())
net_G.double()
net_D.double()

optimizer_G = torch.optim.Adam(net_G.parameters(),lr=lr)
optimizer_D = torch.optim.Adam(net_D.parameters(),lr=lr)

loss_D_epoch = []
loss_G_epoch = []

for e in range(nb_epochs):
    print('epoch: ', e)
    real_samples = input.type(torch.DoubleTensor)
    loss_G = 0
    loss_D = 0
    for t, real_batch in enumerate(tqdm(real_samples.split(batch_size))):
        #improving D
        z = gpu(torch.empty(batch_size,z_dim).normal_().type(torch.DoubleTensor))
        #print(z.type())
        fake_batch = net_G(z)
        real_batch = gpu(real_batch)
        real_batch = real_batch.unsqueeze(1)
        #print(real_batch.size())
        D_scores_on_real = net_D(real_batch)
        D_scores_on_fake = net_D(fake_batch)
            
        loss = -torch.mean(torch.log(1-D_scores_on_fake) + torch.log(D_scores_on_real))
        optimizer_D.zero_grad()
        loss.backward()
        optimizer_D.step()
        loss_D += loss
        #print('Discriminator loss: ', loss_D)
                    
        # improving G
        z = gpu(torch.empty(batch_size,z_dim).normal_().type(torch.DoubleTensor))
        fake_batch = net_G(z)
        D_scores_on_fake = net_D(fake_batch)
            
        loss = -torch.mean(torch.log(D_scores_on_fake))
        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()
        loss_G += loss
        #print('Generator loss: ', loss_G)
           
    loss_D_epoch.append(loss_D)
    loss_G_epoch.append(loss_G)

plt.plot(loss_D_epoch, color = 'b')
plt.plot(loss_G_epoch, color = 'r')
plt.savefig('Loss.png')

z = gpu(torch.empty(batch_size,z_dim).normal_().type(torch.DoubleTensor))

fake_samples = net_G(z)
fake_data = fake_samples.cpu().data.numpy()

for i in range(20):
    x = fake_data[0, 0]

    x = x.reshape(28, 28)

    plt.imshow(x, interpolation='nearest')
    plt.savefig('generated_{}.png'.format(i))

