__author__ = 'cenwan'

#-----The Python implementation of a approach to conduct the CTST on synthetic and real testing protein feature samples.
#-----Please download data via http://bioinfadmin.cs.ucl.ac.uk/downloads/FFPredGAN/TestingData/.

import numpy as np
import math
import glob
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer, roc_auc_score
import numpy as np
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


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

#-----Please download data via http://bioinfadmin.cs.ucl.ac.uk/downloads/FFPredGAN/GOTerm_List.txt
with open(".../GOTerm_List.txt") as ScriptFile:
    GOTermsRaw=ScriptFile.readlines()

GOTerms=[]
for index in range(0,len(GOTermsRaw)):
    GOTerms.append(GOTermsRaw[index].strip())

for index_GOID in range(len(GOTerms)):
    GOTerm=GOTerms[index_GOID]
    #-----Please download data via http://bioinfadmin.cs.ucl.ac.uk/downloads/FFPredGAN/TestingData/GOTermBased/.
    with open(".../"+GOTerm+"-Class-Test.txt") as f:
        testingClass1 = f.readline()
    splitClass=testingClass1.split(",")
    testingClass=[]
    for loopIndex in range(len(testingClass1.split(","))):
        testingClass.append(float(testingClass1.split(",")[loopIndex]))

    #-----Please download data via http://bioinfadmin.cs.ucl.ac.uk/downloads/FFPredGAN/TestingData/GOTermBased/.
    with open(".../"+GOTerm+"-Feature-Test.txt") as f:
        originalMatrixFeaturesTest = [list(x.split(",")) for x in f]
    originalFeaturesTest = [line[0:258] for line in originalMatrixFeaturesTest[:]]
    datasetOriginalTest = np.array(originalFeaturesTest, dtype='float')

    datasetOriginalTest_Positive=[]
    datasetOriginalTest_Negative=[]
    for index1 in range(len(testingClass)):
        if str(testingClass[index1])=="1.0":
        #----change to str(testingClass[index1])=="0.0", if testing with negative samples
            datasetOriginalTest_Positive.append(datasetOriginalTest[index1])
        else:
            datasetOriginalTest_Negative.append(datasetOriginalTest[index1])

    number_of_samples_Positive=len(datasetOriginalTest_Positive)
    the_generator_model = Generator()
    #-----change to ".../"+GOTerm+"_negative_model.pt", if testing with negative samples.
    #-----Please download data via http://bioinfadmin.cs.ucl.ac.uk/downloads/FFPredGAN/TestingData/Trained_FFPredGAN_Models/Positive_samples/.
    the_generator_model.load_state_dict(torch.load(".../"+GOTerm+"_positive_model.pt"))
    the_generator_model.eval()
    noise_positive = torch.randn(number_of_samples_Positive, 258)
    noisev_positive = autograd.Variable(noise_positive)
    fake_positive = autograd.Variable(the_generator_model(noisev_positive).data)
    fake_samples_positive=fake_positive.data.cpu().numpy()
    fakedataset_positive = np.array(fake_samples_positive, dtype='float')

    label=[]
    for rowIndex in range(len(datasetOriginalTest_Positive)):
        label.append(1)
    for rowIndex in range(len(fakedataset_positive)):
        label.append(0)
    labelArray=np.asarray(label)

    realFakeFeatures=np.vstack((datasetOriginalTest_Positive, fakedataset_positive))

    prediction_list=[]
    real_list=[]
    loo = LeaveOneOut()
    loo.get_n_splits(realFakeFeatures)
    for train_index, test_index in loo.split(realFakeFeatures):
        X_train, X_test = realFakeFeatures[train_index], realFakeFeatures[test_index]
        y_train, y_test = labelArray[train_index], labelArray[test_index]
        knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
        predicted_y = knn.predict(X_test)
        prediction_list.append(predicted_y)
        real_list.append(y_test)

    accuracy=accuracy_score(real_list, prediction_list)
    print(GOTerm+"%"+str(accuracy))
    testingClass.clear()
    originalMatrixFeaturesTest.clear()
    prediction_list.clear()
    real_list.clear()
    label.clear()
    testingClass.clear()
    datasetOriginalTest_Positive.clear()
    datasetOriginalTest_Negative.clear()






