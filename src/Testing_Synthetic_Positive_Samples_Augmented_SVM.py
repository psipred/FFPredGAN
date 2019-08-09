__author__ = 'cenwan'

import numpy as np
import math
import glob
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer, roc_auc_score

GOTerm="GO0034613"

MCCValue=make_scorer(matthews_corrcoef)

#-----Loading testing data-----
with open(".../"+GOTerm+"-Class-Test.txt") as f:
    testingClass1 = f.readline()
splitClass=testingClass1.split(",")
testingClass=[]
for loopIndex in range(len(testingClass1.split(","))):
    testingClass.append(float(testingClass1.split(",")[loopIndex]))

with open(".../"+GOTerm+"-Feature-Test.txt") as f:
    originalMatrixFeaturesTest = [list(x.split(",")) for x in f]
originalFeaturesTest = [line[0:258] for line in originalMatrixFeaturesTest[:]]
datasetOriginalTest = np.array(originalFeaturesTest, dtype='float')

#-----Loading real training data-----
with open(".../"+GOTerm+"_Real_Training_Negative.txt") as GONameFileNegative:
    TrainingNegative = [list(x.split(",")) for x in GONameFileNegative]
originalFeaturesTrainingNegative = [line[1:259] for line in TrainingNegative[:]]
with open(".../"+GOTerm+"_Real_Training_Positive.txt") as GONameFilePositive:
    TrainingPositive = [list(x.split(",")) for x in GONameFilePositive]
originalFeaturesTrainingPositive = [line[1:259] for line in TrainingPositive[:]]

datasetOriginalTrain = np.vstack((originalFeaturesTrainingNegative, originalFeaturesTrainingPositive))

trainingClass = []
for loopIndex in range(len(TrainingNegative)):
  trainingClass.append(0)
for loopIndex in range(len(TrainingPositive)):
  trainingClass.append(1)

#-----Loading synthetic training data-----
with open(".../"+GOTerm+"_Synthetic_Training_Positive.txt") as f:
    matrixFeaturesFakePositive = [list(x.split(",")) for x in f]
matrixFeaturesFakePositive = [line[0:258] for line in matrixFeaturesFakePositive[:]]
matrixFeaturesFakePositive = np.array(matrixFeaturesFakePositive, dtype='float')

newTrainingFeatures=np.vstack((datasetOriginalTrain, matrixFeaturesFakePositive))

for loopIndex in range(len(matrixFeaturesFakePositive)):
    trainingClass.append(1)

#-----Grid-search for SVM hyper-parameter optimisation-----
param_grid = [
    {'C': [1, 10, 0.1, 100, 0.01, 1000, 0.001, 1e4, 1e-4], 'kernel': ['linear']},
    {'C': [1, 10, 0.1, 100, 0.01, 1000, 0.001, 1e4, 1e-4],
      'gamma': [1, 0.5, 3, 0.2, 10, 0.1, 0.03, 0.01, 0.001, 1e-4], 'kernel': ['rbf']}
]

svc = SVC()
clf = GridSearchCV(estimator=svc, param_grid=param_grid, scoring=MCCValue, cv=5)
clf.fit(newTrainingFeatures, trainingClass)

#-----Training SVM with optimal hyper-parameters-----
para = clf.best_params_
c_value = float(para['C'])
kernel_value = para['kernel']
if kernel_value == 'rbf':
    gamma_value = float(para['gamma'])
    svc_trained = SVC(kernel=kernel_value, C=c_value, gamma=gamma_value, probability=True)
else:
    svc_trained = SVC(kernel=kernel_value, C=c_value, probability=True)

svc_trained.fit(newTrainingFeatures, trainingClass)
resultsTestingProb = svc_trained.predict_proba(datasetOriginalTest)

resultsTesting=[]
for indexResults in range(len(resultsTestingProb)):
    if float(resultsTestingProb[indexResults][1])>0.5 or float(resultsTestingProb[indexResults][1])==0.5:
       resultsTesting.append(1)
    else:
       resultsTesting.append(0)

mccTesting = matthews_corrcoef(testingClass, resultsTesting)
print("MCC: "+GOTerm+": "+str(mccTesting))

aurocTesting = roc_auc_score(testingClass, resultsTestingProb[:,1])
print("AUROC: "+GOTerm+": "+str(aurocTesting))

