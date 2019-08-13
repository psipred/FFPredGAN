__author__ = 'cenwan'

#-----This is a PyTorch implementation of FFPred-GAN synthetic protein feature samples generation method. 
#-----This code was adapted from an implementation of WGAN-GP by caogang/wgan-gp@GitHub repository.
#-----The real training data can be downloaded via http://bioinfadmin.cs.ucl.ac.uk/downloads/FFPredGAN.

import os, sys
sys.path.append(os.getcwd())
import random
import numpy as np
import sklearn.datasets
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import decomposition
torch.manual_seed(1)

GOTermID='GO0034613'

with open(".../"+GOTermID+"_Real_Training_Positive.txt") as f:
    MatrixFeaturesPositiveSamples = [list(x.split(",")) for x in f]
FeaturesPositiveSamples = [line[1:259] for line in MatrixFeaturesPositiveSamples[:]]

GDIM = 512
DDIM = 86 
FIXED_GENERATOR = False  
LAMBDA = .1  
CRITIC_ITERS = 5  
BATCH_SIZE = len(FeaturesPositiveSamples)
ITERS = 100000  
use_cuda = False

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(258, GDIM),
            nn.ReLU(True),
            nn.Linear(GDIM, GDIM),
            nn.ReLU(True),
            nn.Linear(GDIM, GDIM),
            nn.Tanh(),
            nn.Linear(GDIM, 258),
        )
        self.main = main

    def forward(self, noise, real_data):
        if FIXED_GENERATOR:
            return noise + real_data
        else:
            output = self.main(noise)
            return output

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.fc1=nn.Linear(258, DDIM)
        self.relu=nn.LeakyReLU()
        self.fc2=nn.Linear(DDIM, DDIM)
        self.relu=nn.LeakyReLU()
        self.fc3 = nn.Linear(DDIM, DDIM)
        self.relu = nn.LeakyReLU()
        self.fc4 = nn.Linear(DDIM, 1)

    def forward(self, inputs):

        out=self.fc1(inputs)
        out=self.relu(out)
        out=self.fc2(out)
        out=self.relu(out)
        out=self.fc3(out)
        out=self.relu(out)
        out=self.fc4(out)

        hidden1 = self.relu(self.fc1(inputs))
        hidden2 = self.relu(self.fc2(self.relu(self.fc1(inputs))))
        hidden3 = self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(inputs))))))

        return out.view(-1), hidden1, hidden2, hidden3


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def inf_train_gen():

    with open(".../"+GOTermID+"_Real_Training_Positive.txt") as f:
        MatrixFeaturesPositive = [list(x.split(",")) for x in f]
    proteinList = [line[0:1] for line in MatrixFeaturesPositive[:]]
    FeaturesPositive = [line[1:259] for line in MatrixFeaturesPositive[:]]
    dataset2 = np.array(FeaturesPositive, dtype='float32')

    return dataset2
  
def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, hidden_output_1, hidden_output_2, hidden_output_3 = netD(interpolates) 

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

netG = Generator()
netD = Discriminator()
netD.apply(weights_init)
netG.apply(weights_init)


if use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()

data = inf_train_gen()


for iteration in range(ITERS):

    for p in netD.parameters():  
        p.requires_grad = True  

    data = inf_train_gen()
    real_data = torch.FloatTensor(data)
    if use_cuda:
        real_data = real_data.cuda()
    real_data_v = autograd.Variable(real_data)
    
    noise = torch.randn(BATCH_SIZE, 258)
    if use_cuda:
        noise = noise.cuda()
    noisev = autograd.Variable(noise, volatile=True)  
    fake = autograd.Variable(netG(noisev, real_data_v).data)

    fake_output=fake.data.cpu().numpy()

    for iter_d in range(CRITIC_ITERS):

        netD.zero_grad()

        D_real, hidden_output_real_1, hidden_output_real_2, hidden_output_real_3 = netD(real_data_v)
        D_real = D_real.mean()
        D_real.backward(mone)

        noise = torch.randn(BATCH_SIZE, 258)
        if use_cuda:
            noise = noise.cuda()
        noisev = autograd.Variable(noise, volatile=True)  
        fake = autograd.Variable(netG(noisev, real_data_v).data)
        
        inputv = fake
        D_fake, hidden_output_fake_1, hidden_output_fake_2, hidden_output_fake_3 = netD(inputv)
        D_fake = D_fake.mean()
        D_fake.backward(one)

        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
        gradient_penalty.backward()

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()

    if iteration%200 == 0:
       fake_writer = open(".../"+GOTermID+"_Iteration_"+str(iteration)"_Synthetic_Training_Positive.txt","w")
      
       for rowIndex in range(len(fake_output)):
           for columnIndex in range(len(fake_output[0])):
                fake_writer.write(str(fake_output[rowIndex][columnIndex]) + ",")
           fake_writer.write("\n")
       fake_writer.flush()
       fake_writer.close()


    if not FIXED_GENERATOR:

        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()

        real_data = torch.Tensor(data)
        if use_cuda:
            real_data = real_data.cuda()
        real_data_v = autograd.Variable(real_data)

        noise = torch.randn(BATCH_SIZE, 258)
        if use_cuda:
            noise = noise.cuda()
        noisev = autograd.Variable(noise)
        fake = netG(noisev, real_data_v)
        G, hidden_output_ignore_1, hidden_output_ignore_2, hidden_output_ignore_3 = netD(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()



