__author__ = 'cenwan'

# The Python implementation of generating synthetic samples by using the trained generator network of FFPred-GAN.
# Please download the trained models via http://bioinfadmin.cs.ucl.ac.uk/downloads/FFPredGAN/TestingData/Trained_FFPredGAN_Models/.

import numpy as np
import random
import torch
import torch.autograd as autograd
import torch.nn as nn


#-----Define the generator network-----
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(258, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 258),
        )
        self.main = main

    def forward(self, noise):
            output = self.main(noise)
            return output

#-----Change the GO ID if generating synthetic samples for other GO terms.
GOTerm='GO0034613'
number_of_samples=1

#-----Load the trained generator network-----
the_generator_model = Generator()
the_generator_model.load_state_dict(torch.load(".../"+GOTerm+"_positive_model.pt"))
the_generator_model.eval()
noise = torch.randn(number_of_samples, 258)
noisev = autograd.Variable(noise)
fake = autograd.Variable(the_generator_model(noisev).data)
fake_samples=fake.data.cpu().numpy()

#-----Save the generated synthetic samples-----
fakedataset = np.array(fake_samples, dtype='float32')
fileWriter_Synthetic_sample = open(".../Synthetic_samples.txt", "w")
for index1 in range(len(fakedataset)):
	for index2 in range(len(fakedataset[0])):
		fileWriter_Synthetic_sample.write(str(fakedataset[index1][index2])+",")
	fileWriter_Synthetic_sample.write("\n")	
fileWriter_Synthetic_sample.flush()
fileWriter_Synthetic_sample.close()


        
        
        
        
        
