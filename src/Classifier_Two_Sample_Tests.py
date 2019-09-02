__author__ = 'cenwan'

# The Python implementation of Classifier Two-Sample Tests (CTST) for selecting the optimal synthetic protein feature samples.

import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

GOTermID='GO0034613'

with open(".../"+GOTermID+"_Real_Training_Positive.txt") as f:
    MatrixFeatures = [list(x.split(",")) for x in f]
proteinList = [line[0:1] for line in MatrixFeatures[:]]
realFeatures = [line[1:259] for line in MatrixFeatures[:]]
realDataset = np.array(realFeatures, dtype='float32')

# Adding equal numbers of binary labels
label=[]
for rowIndex in range(len(realDataset)):
    label.append(1)
for rowIndex in range(len(realDataset)):
    label.append(0)
labelArray=np.asarray(label)

opt_diff_accuracy_05=0.5
opt_Epoch=0
opt_accuracy=0
for indexEpoch in range(0, 500):
    epoch = indexEpoch * 200
    with open(".../"+GOTermID+"_Iteration_"+str(epoch)+"_Synthetic_Training_Positive.txt") as f:
         MatrixFeatures = [list(x.split(",")) for x in f]
    fakeFeatures = [line[0:258] for line in MatrixFeatures[:]]
    fakedataset = np.array(fakeFeatures, dtype='float32')

    realFakeFeatures=np.vstack((realDataset, fakedataset))

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
    diff_accuracy_05=abs(accuracy-0.5)
    if diff_accuracy_05 < opt_diff_accuracy_05:
       opt_diff_accuracy_05=diff_accuracy_05
       opt_Epoch=epoch
       opt_accuracy=accuracy
print(GOTermID+"%"+str(opt_Epoch)+"%"+str(opt_accuracy))

