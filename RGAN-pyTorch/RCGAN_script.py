import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
#from skimage.io import imsave
import os
#from tensorboardX import SummaryWriter

from tqdm import tqdm

use_gpu = torch.cuda.is_available()
def gpu(tensor, gpu=use_gpu):
    if gpu:
        return tensor.cuda()
    else:
        return tensor
    
img_height = 28
img_width = 28
img_size = img_height * img_width

to_train = True
to_restore = False
output_path = "output"

max_epoch = 1000

hg_size = 150
hd_size = 300
z_size = 100
batch_size = 256
seq_size=4
n_hidden=300
tr_data_num=60000;
g_num_layers=2;
d_num_layers=2;

root_dir = "/home/majrda/Scripts/data"

class GaussianNoise(nn.Module):
    def __init__(self, stddev = 0.1):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
        return din

class ModelG(nn.Module):
    def __init__(self, z_dim):
        self.z_dim = z_dim

        super(ModelG, self).__init__()

        self.lstm_G = nn.LSTM(input_size = z_dim + 10, 
                   hidden_size = n_hidden,
                   num_layers = g_num_layers,
                   bias = False,
                   dropout = .5)

        self.relu = nn.ReLU()

        self.fc_G = nn.Linear(n_hidden, img_size)

    def forward(self, x, label_onehot):
        x = torch.cat([x, label_onehot], 1)
        x = x.unsqueeze(1)
        #print(x.size())
        output, _ = self.lstm_G(x)
        output = torch.tanh(self.fc_G(self.relu(output)))
        return output

class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        self.lstm_D = nn.LSTM(input_size = img_size + 10, 
                               hidden_size = n_hidden,
                               num_layers = d_num_layers)
        
        self.relu = nn.ReLU()

        self.MLP = nn.Linear(n_hidden, 1)
        
        self.noise = GaussianNoise(.3)

    def forward(self, x, label_onehot):
        #print(x.size())
        x = torch.cat([x, label_onehot], 2)
        #print(x.size())
        x = self.noise(x)
        outputs, _ = self.lstm_D(x)
        #print(outputs.size())
        res = self.MLP(self.relu(outputs[:, -1, :]))
        #print(res)
        y_data = torch.sigmoid(res.narrow(0, 0, x[0].shape[0]))
        return y_data
    
mnist_trainset = datasets.MNIST(root=root_dir, train=True, download=False, transform=transforms.ToTensor())
mnist_testset = datasets.MNIST(root=root_dir, train=False, download=False, transform=transforms.ToTensor())

X_train = mnist_trainset.train_data.numpy()
Y_train = mnist_trainset.train_labels.numpy()

X_test = mnist_testset.test_data.numpy()
Y_test = mnist_testset.test_labels.numpy()

lr = 1e-4
nb_epochs = 100

loss_D_epoch = []
loss_G_epoch = []

z_dim = z_size
label_dim = 10

net_CG = gpu(ModelG(z_size))
net_CD = gpu(ModelD())

optimizer_CG = torch.optim.Adam(net_CG.parameters(),lr=lr)
optimizer_CD = torch.optim.Adam(net_CD.parameters(),lr=lr)

for e in range(nb_epochs):
    print("Epoch: ", e)
    rperm = np.random.permutation(X_train.shape[0]);
    np.take(X_train,rperm,axis=0,out=X_train);
    np.take(Y_train,rperm,axis=0,out=Y_train);
    real_samples = torch.from_numpy(X_train).type(torch.FloatTensor)
    real_labels = torch.from_numpy(Y_train).type(torch.LongTensor)
    loss_G = 0
    loss_D = 0
    for real_batch, real_batch_label in tqdm(zip(real_samples.split(batch_size),real_labels.split(batch_size))):
        
        #print(real_batch.size())
        #print(real_batch.shape[0])
        real_batch = real_batch.view(real_batch.shape[0], 1, img_size).cuda()
        #print(real_batch.size())
            
        #improving D
        z = gpu(torch.empty(real_batch.shape[0],z_dim).normal_())
        #print(real_batch_label)
        real_batch_label_ = torch.unsqueeze(real_batch_label, 1)
        #print(real_batch_label_.size())
        label_onehot = torch.FloatTensor(real_batch.shape[0], label_dim).zero_()
        #print(label_onehot.size())
        label_onehot = gpu(label_onehot.scatter(1, real_batch_label_, 1).type(torch.FloatTensor))
        #print(label_onehot.size())
        
        fake_batch = net_CG(z, label_onehot)
        #print("FB size: ", fake_batch.size())
        label_onehot = label_onehot.unsqueeze(1)
        #print(label_onehot.size())
        D_scores_on_real = net_CD(gpu(real_batch), label_onehot)
        D_scores_on_fake = net_CD(fake_batch, label_onehot)
        
        loss = -torch.mean(torch.log(1-D_scores_on_fake) + torch.log(D_scores_on_real))
        optimizer_CD.zero_grad()
        loss.backward()
        optimizer_CD.step()
        loss_D += loss
            
            # improving G
        z = gpu(torch.empty(real_batch.shape[0],z_dim).normal_())
        real_batch_label_ = torch.unsqueeze(real_batch_label, 1)
        #print(real_batch_label_.size())
        label_onehot = torch.FloatTensor(real_batch.shape[0], label_dim).zero_()
        #print(label_onehot.size())
        label_onehot = gpu(label_onehot.scatter(1, real_batch_label_, 1).type(torch.FloatTensor))
        #print(label_onehot)
        fake_batch = net_CG(z, label_onehot)
        
        label_onehot = label_onehot.unsqueeze(1)
        D_scores_on_fake = net_CD(fake_batch, label_onehot)
            
        loss = -torch.mean(torch.log(D_scores_on_fake))
        optimizer_CG.zero_grad()
        loss.backward()
        optimizer_CG.step()
        loss_G += loss
                    
    loss_D_epoch.append(loss_D)
    loss_G_epoch.append(loss_G)
    print("Loss on Generator this epoch: {}\nLoss on Discriminator this epoch: {}".format(loss_G, loss_D))
    
import matplotlib.pyplot as plt

plt.plot(loss_D_epoch, color ='b')
plt.plot(loss_G_epoch, color = 'r')
plt.savefig('Losses.png')
    
z = gpu(torch.empty(Y_test.size,z_size).normal_())
real_batch_label = torch.from_numpy(Y_test).type(torch.LongTensor)
real_batch_label_ = torch.unsqueeze(real_batch_label, 1)
#print(real_batch_label_.size())
label_onehot = torch.FloatTensor(Y_test.size, label_dim).zero_()
#print(label_onehot.size())
label_onehot = gpu(label_onehot.scatter(1, real_batch_label_, 1).type(torch.FloatTensor))
#print(label_onehot.size())

fake_samples = net_CG(z, label_onehot)
fake_data = fake_samples.cpu().data.numpy()

for i in range(20):
    x = fake_data[0, 0]

    x = x.reshape(28, 28)

    from matplotlib import pyplot as plt
    plt.imshow(x, interpolation='nearest')
    plt.savefig('generated_{}.png'.format(i))
    
    
    
