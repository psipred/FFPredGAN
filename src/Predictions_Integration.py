__author__ = 'cenwan'

#-----The Python (version 0.22.2) implementation of a predictions integration approach.
#-----Please download data via http://bioinfadmin.cs.ucl.ac.uk/downloads/FFPredGAN/TestingData/Integration/

import numpy as np
import collections
import pickle
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import GridSearchCV

def set_Dict(key, Prob, Target_GOTerm_Prob_Dict):
    if key in Target_GOTerm_Prob_Dict:
        old_Prob = Target_GOTerm_Prob_Dict.get(key)
        if float(Prob) > float(old_Prob):
            Target_GOTerm_Prob_Dict[key] = Prob
    else:
        Target_GOTerm_Prob_Dict[key] = Prob

    return Target_GOTerm_Prob_Dict

#-----Loading the GO hierarchy file.
with open(".../GO_Path.txt") as ReaderAllPath:
    AllPathList=ReaderAllPath.read().splitlines()

#-----Loading the raw predictions made by FFPredGAN for BP terms. Replace this file for the purpose of back-propagating the predictions on MF and CC terms made by FFPredGAN or other methods such as FFPred.
with open(".../Predictions_FFPredGAN_BP_Raw.txt") as ReaderFFPredGANPredictionRaw:
    FFPredGANPredictionRawList=ReaderFFPredGANPredictionRaw.read().splitlines()
Target_List_FFPredGAN=[]
for index1 in range(len(FFPredGANPredictionRawList)):
    TempArray=FFPredGANPredictionRawList[index1].split("%")
    Target_List_FFPredGAN.append(TempArray[0])
Target_List_Unique_FFPredGAN=list(dict.fromkeys(Target_List_FFPredGAN))

Target_GOTerm_Prob_Dict_FFPredGAN={}
for index1 in range(len(Target_List_Unique_FFPredGAN)):
    Target=Target_List_Unique_FFPredGAN[index1]
    for index2 in range(len(FFPredGANPredictionRawList)):
        if FFPredGANPredictionRawList[index2].startswith(Target_List_Unique_FFPredGAN[index1]):
            GOTerm=FFPredGANPredictionRawList[index2].split("%")[1]
            Prob=FFPredGANPredictionRawList[index2].split("%")[2]

            AllParentGOTerms=[]
            for index11 in range(len(AllPathList)):
                if AllPathList[index11].startswith(GOTerm):
                    tempArrayAllPath = AllPathList[index11].split(",")
                    for index12 in range(len(tempArrayAllPath)):
                        AllParentGOTerms.append(tempArrayAllPath[index12])
            AllParentGOTerms_Unique = list(dict.fromkeys(AllParentGOTerms))

            for index21 in range(len(AllParentGOTerms_Unique)):
                key=Target+"%"+AllParentGOTerms_Unique[index21]
                Target_GOTerm_Prob_Dict_FFPredGAN=set_Dict(key, Prob, Target_GOTerm_Prob_Dict_FFPredGAN)

fileWriter_FFPredGAN_BPed = open(".../FFPredGAN_Predictions_BP_BPed.txt", "w")
for key, value in Target_GOTerm_Prob_Dict_FFPredGAN.items():
    fileWriter_FFPredGAN_BPed.write(key+"%"+value+"\n")
fileWriter_FFPredGAN_BPed.flush()
fileWriter_FFPredGAN_BPed.close()

#-----Loading the raw predictions made by NetGO for BP terms. Replace this file for the purpose of back-propagating the predictions on MF and CC terms.
with open(".../Predictions_NetGO_BP_Raw.txt") as ReaderNetGOPredictionRaw:
    NetGOPredictionRawList=ReaderNetGOPredictionRaw.read().splitlines()
Target_List_NetGO=[]
for index1 in range(len(NetGOPredictionRawList)):
    TempArray=NetGOPredictionRawList[index1].split("%")
    Target_List_NetGO.append(TempArray[0])
Target_List_Unique_NetGO=list(dict.fromkeys(Target_List_NetGO))

Target_GOTerm_Prob_Dict_NetGO={}
for index1 in range(len(Target_List_Unique_NetGO)):
    Target=Target_List_Unique_NetGO[index1]
    for index2 in range(len(NetGOPredictionRawList)):
        if NetGOPredictionRawList[index2].startswith(Target_List_Unique_NetGO[index1]):
            GOTerm=NetGOPredictionRawList[index2].split("%")[1]
            Prob=NetGOPredictionRawList[index2].split("%")[2]

            AllParentGOTerms=[]
            for index11 in range(len(AllPathList)):
                if AllPathList[index11].startswith(GOTerm):
                    tempArrayAllPath = AllPathList[index11].split(",")
                    for index12 in range(len(tempArrayAllPath)):
                        AllParentGOTerms.append(tempArrayAllPath[index12])
            AllParentGOTerms_Unique = list(dict.fromkeys(AllParentGOTerms))

            for index21 in range(len(AllParentGOTerms_Unique)):
                key=Target+"%"+AllParentGOTerms_Unique[index21]
                Target_GOTerm_Prob_Dict_NetGO=set_Dict(key, Prob, Target_GOTerm_Prob_Dict_NetGO)

fileWriter_NetGO_BPed = open(".../NetGO_Predictions_BP_BPed.txt", "w")
for key, value in Target_GOTerm_Prob_Dict_NetGO.items():
    fileWriter_NetGO_BPed.write(key+"%"+value+"\n")
fileWriter_NetGO_BPed.flush()
fileWriter_NetGO_BPed.close()

#-----Loading the true BP term annotation labels of targets. Replace this file for the purpose of back-propagating the predictions on MF and CC terms.
with open(".../True_Label_BP_Raw.txt") as ReaderLabelPredictionRaw:
    LabelPredictionRawList=ReaderLabelPredictionRaw.read().splitlines()
Target_List_Label=[]
for index1 in range(len(LabelPredictionRawList)):
    TempArray=LabelPredictionRawList[index1].split("%")
    Target_List_Label.append(TempArray[0])
Target_List_Unique_Label=list(dict.fromkeys(Target_List_Label))

Target_GOTerm_Prob_Dict_Label={}
for index1 in range(len(Target_List_Unique_Label)):
    Target=Target_List_Unique_Label[index1]
    for index2 in range(len(LabelPredictionRawList)):
        if LabelPredictionRawList[index2].startswith(Target_List_Unique_Label[index1]):
            GOTerm=LabelPredictionRawList[index2].split("%")[1]
            Prob=LabelPredictionRawList[index2].split("%")[2]

            AllParentGOTerms=[]
            for index11 in range(len(AllPathList)):
                if AllPathList[index11].startswith(GOTerm):
                    tempArrayAllPath = AllPathList[index11].split(",")
                    for index12 in range(len(tempArrayAllPath)):
                        AllParentGOTerms.append(tempArrayAllPath[index12])
            AllParentGOTerms_Unique = list(dict.fromkeys(AllParentGOTerms))

            for index21 in range(len(AllParentGOTerms_Unique)):
                key=Target+"%"+AllParentGOTerms_Unique[index21]
                Target_GOTerm_Prob_Dict_Label=set_Dict(key, Prob, Target_GOTerm_Prob_Dict_Label)

fileWriter_Label_BPed = open(".../True_Label_BP_BPed.txt", "w")
for key, value in Target_GOTerm_Prob_Dict_Label.items():
    fileWriter_Label_BPed.write(key+"%"+value+"\n")
fileWriter_Label_BPed.flush()
fileWriter_Label_BPed.close()

#-----This is the final output file including the integrated predictions based on FFPredGAN and NetGO.
fileWriterIntegration = open(".../Integration_NetGO_FFPredGAN_BP.txt", "w")

#-----Loading the 10-fold allocation index of targets.
with open(".../10_Fold_Allocation_BP.txt") as ReaderFoldAllocationIndex:
    FoldAllocationIndex=ReaderFoldAllocationIndex.read().splitlines()

dict_fold_allocation={}
all_target_list=[]
for index_fold in range(len(FoldAllocationIndex)):
    temp_array=FoldAllocationIndex[index_fold].split("%")
    dict_fold_allocation[temp_array[0]]=temp_array[1]
    all_target_list.append(temp_array[0])

for fold in range(0,10):
    fold_index=fold+1
    Target_Fold_2=[]
    Target_Fold_1=[]
    for index_target in range(len(all_target_list)):
        if dict_fold_allocation.get(all_target_list[index_target])==str(fold_index):
            Target_Fold_2.append(all_target_list[index_target])
        else:
            Target_Fold_1.append(all_target_list[index_target])

    with open(".../NetGO_Predictions_BP_BPed.txt") as ReaderNetGOBP:
        NetGOBPList=ReaderNetGOBP.read().splitlines()

    with open(".../FFPredGAN_Predictions_BP_BPed.txt") as ReaderFFPredGANBP:
        FFPredGANBPList=ReaderFFPredGANBP.read().splitlines()

    with open(".../True_Label_BP_BPed.txt") as ReaderTrueLabelBP:
        TrueLabelBPList=ReaderTrueLabelBP.read().splitlines()

    NetGO_Fold_1=[]
    FFPredGAN_Fold_1=[]
    TrueLabel_Fold_1=[]
    for index1 in range(len(Target_Fold_1)):
        for index2 in range(len(NetGOBPList)):
            if NetGOBPList[index2].startswith(Target_Fold_1[index1]):
                NetGO_Fold_1.append(NetGOBPList[index2])
        for index3 in range(len(FFPredGANBPList)):
            if FFPredGANBPList[index3].startswith(Target_Fold_1[index1]):
                FFPredGAN_Fold_1.append(FFPredGANBPList[index3])
        for index4 in range(len(TrueLabelBPList)):
            if TrueLabelBPList[index4].startswith(Target_Fold_1[index1]):
                TrueLabel_Fold_1.append(TrueLabelBPList[index4])

    os.mkdir(".../Fold_" + str(fold_index))

    fileWriter_Fold_1_NetGO = open(".../Fold_" + str(fold_index) + "/NetGO_Fold_1.txt", "w")
    fileWriter_Fold_1_FFPredGAN = open(".../Fold_" + str(fold_index) + "/FFPredGAN_Fold_1.txt", "w")
    fileWriter_Fold_1_TrueLabel = open(".../Fold_" + str(fold_index) + "/TrueLabel_Fold_1.txt", "w")

    for index5 in range(len(NetGO_Fold_1)):
        fileWriter_Fold_1_NetGO.write(NetGO_Fold_1[index5]+"\n")
    fileWriter_Fold_1_NetGO.flush()
    fileWriter_Fold_1_NetGO.close()

    for index6 in range(len(FFPredGAN_Fold_1)):
        fileWriter_Fold_1_FFPredGAN.write(FFPredGAN_Fold_1[index6]+"\n")
    fileWriter_Fold_1_FFPredGAN.flush()
    fileWriter_Fold_1_FFPredGAN.close()

    for index7 in range(len(TrueLabel_Fold_1)):
        fileWriter_Fold_1_TrueLabel.write(TrueLabel_Fold_1[index7]+"\n")
    fileWriter_Fold_1_TrueLabel.flush()
    fileWriter_Fold_1_TrueLabel.close()

    NetGO_Fold_2=[]
    FFPredGAN_Fold_2=[]
    TrueLabel_Fold_2=[]
    for index1 in range(len(Target_Fold_2)):
        for index2 in range(len(NetGOBPList)):
            if NetGOBPList[index2].startswith(Target_Fold_2[index1]):
                NetGO_Fold_2.append(NetGOBPList[index2])
        for index3 in range(len(FFPredGANBPList)):
            if FFPredGANBPList[index3].startswith(Target_Fold_2[index1]):
                FFPredGAN_Fold_2.append(FFPredGANBPList[index3])
        for index4 in range(len(TrueLabelBPList)):
            if TrueLabelBPList[index4].startswith(Target_Fold_2[index1]):
                TrueLabel_Fold_2.append(TrueLabelBPList[index4])

    fileWriter_Fold_2_NetGO = open(".../Fold_" + str(fold_index) + "/NetGO_Fold_2.txt", "w")
    fileWriter_Fold_2_FFPredGAN = open(".../Fold_" + str(fold_index) + "/FFPredGAN_Fold_2.txt", "w")
    fileWriter_Fold_2_TrueLabel = open(".../Fold_" + str(fold_index) + "/TrueLabel_Fold_2.txt", "w")

    for index5 in range(len(NetGO_Fold_2)):
        fileWriter_Fold_2_NetGO.write(NetGO_Fold_2[index5]+"\n")
    fileWriter_Fold_2_NetGO.flush()
    fileWriter_Fold_2_NetGO.close()

    for index6 in range(len(FFPredGAN_Fold_2)):
        fileWriter_Fold_2_FFPredGAN.write(FFPredGAN_Fold_2[index6]+"\n")
    fileWriter_Fold_2_FFPredGAN.flush()
    fileWriter_Fold_2_FFPredGAN.close()

    for index7 in range(len(TrueLabel_Fold_2)):
        fileWriter_Fold_2_TrueLabel.write(TrueLabel_Fold_2[index7]+"\n")
    fileWriter_Fold_2_TrueLabel.flush()
    fileWriter_Fold_2_TrueLabel.close()

    with open(".../Fold_" + str(fold_index) + "/NetGO_Fold_1.txt") as ReaderNetGOFold1:
        NetGOListFold1=ReaderNetGOFold1.read().splitlines()
    with open(".../Fold_" + str(fold_index) + "/FFPredGAN_Fold_1.txt") as ReaderFFPredGANFold1:
        FFPredGANListFold1=ReaderFFPredGANFold1.read().splitlines()
    with open(".../Fold_" + str(fold_index) + "/TrueLabel_Fold_1.txt") as ReaderTrueLabelFold1:
        TrueLabelListFold1=ReaderTrueLabelFold1.read().splitlines()

    Targets_List_Fold1_Int=[]
    for index1 in range(len(NetGOListFold1)):
        Targets_List_Fold1_Int.append(NetGOListFold1[index1].split("%")[0])
    Target_List_Fold1_Unique_Int = list(dict.fromkeys(Targets_List_Fold1_Int))

    fileWriter_Cooccure_Dict = open(".../Fold_" + str(fold_index) + "/fileWriter_Cooccure_Dict.txt", "w")
    fileWriter_Cooccure_Prob_NetGO = open(".../Fold_" + str(fold_index) + "/fileWriter_Cooccure_Prob_NetGO.txt", "w")
    fileWriter_Cooccure_Prob_FFPredGAN = open(".../Fold_" + str(fold_index) + "/fileWriter_Cooccure_Prob_FFPredGAN.txt", "w")

    Cooccure_Dict={}
    Cooccure_Prob_NetGO = {}
    Cooccure_Prob_FFPredGAN = {}
    for index1 in range(len(Target_List_Fold1_Unique_Int)):
        for index2 in range(len(NetGOListFold1)):
            if NetGOListFold1[index2].startswith(Target_List_Fold1_Unique_Int[index1]):
                temp_list=NetGOListFold1[index2].split("%")
                GOTerm_NetGO=temp_list[1]
                Pred_Prob=temp_list[2]
                check_cooccure=False
                for index3 in range(len(FFPredGANListFold1)):
                    if FFPredGANListFold1[index3].startswith(Target_List_Fold1_Unique_Int[index1]+"%"+GOTerm_NetGO):
                        temp_list_FFPredGAN=FFPredGANListFold1[index3].split("%")
                        print()
                        check_cooccure=True
                        Cooccure_Prob_FFPredGAN[Target_List_Fold1_Unique_Int[index1]+"%"+GOTerm_NetGO]=temp_list_FFPredGAN[2]
                        fileWriter_Cooccure_Prob_FFPredGAN.write(Target_List_Fold1_Unique_Int[index1]+"%"+GOTerm_NetGO+">"+temp_list_FFPredGAN[2]+"\n")
                        break
                if check_cooccure==True:
                    Cooccure_Prob_NetGO[temp_list[0]+"%"+temp_list[1]]=temp_list[2]
                    fileWriter_Cooccure_Prob_NetGO.write(temp_list[0]+"%"+temp_list[1]+">"+temp_list[2]+"\n")
                    Cooccure_Dict[temp_list[0]]=temp_list[1]
                    fileWriter_Cooccure_Dict.write(temp_list[0]+">"+temp_list[1]+"\n")

    fileWriter_Cooccure_Dict.flush()
    fileWriter_Cooccure_Dict.close()
    fileWriter_Cooccure_Prob_NetGO.flush()
    fileWriter_Cooccure_Prob_NetGO.close()
    fileWriter_Cooccure_Prob_FFPredGAN.flush()
    fileWriter_Cooccure_Prob_FFPredGAN.close()

    with open(".../Fold_" + str(fold_index) + "/fileWriter_Cooccure_Dict.txt") as Reader_Cooccure_Dict:
        Cooccure_Dict_List=Reader_Cooccure_Dict.read().splitlines()
    with open(".../Fold_" + str(fold_index) + "/fileWriter_Cooccure_Prob_NetGO.txt") as Reader_Cooccure_Prob_NetGO:
        Cooccure_Prob_NetGO_List=Reader_Cooccure_Prob_NetGO.read().splitlines()
    with open(".../Fold_" + str(fold_index) + "/fileWriter_Cooccure_Prob_FFPredGAN.txt") as Reader_Cooccure_Prob_FFPredGAN:
        Cooccure_Prob_FFPredGAN_List=Reader_Cooccure_Prob_FFPredGAN.read().splitlines()
    with open(".../Fold_" + str(fold_index) + "/TrueLabel_Fold_1.txt") as Reader_True_Label_BPed:
        True_Label_BPed_List=Reader_True_Label_BPed.read().splitlines()

    Cooccure_GO_Terms=[]
    Cooccure_Protein=[]
    for index1 in range(len(Cooccure_Dict_List)):
        temp=Cooccure_Dict_List[index1].split(">")
        Cooccure_GO_Terms.append(temp[1])
        Cooccure_Protein.append(temp[0])
    Cooccure_GO_Terms_unique=list(dict.fromkeys(Cooccure_GO_Terms))
    Cooccure_Protein_unique=list(dict.fromkeys(Cooccure_Protein))

    Valid_GOTerm=[]
    GOTerms_Model_Dict={}
    for index1 in range(len(Cooccure_GO_Terms_unique)):
        prob_NetGO=[]
        prob_FFPredGAN=[]
        true_Label=[]
        for index2 in range(len(Cooccure_Dict_List)):
            if Cooccure_GO_Terms_unique[index1] in Cooccure_Dict_List[index2]:
                prob_NetGO.append(Cooccure_Prob_NetGO_List[index2].split(">")[1])
                prob_FFPredGAN.append(Cooccure_Prob_FFPredGAN_List[index2].split(">")[1])
                protein_ID=Cooccure_Dict_List[index2].split(">")[0]
                BooleanFoundLabel=False
                for index3 in range(len(True_Label_BPed_List)):
                    if protein_ID+"%"+Cooccure_GO_Terms_unique[index1] in True_Label_BPed_List[index3]:
                        BooleanFoundLabel=True
                        break
                if BooleanFoundLabel==True:
                    true_Label.append(1)
                else:
                    true_Label.append(0)

        if 1 in true_Label and 0 in true_Label and true_Label.count(1)>4:
            training_features=[]
            for index11 in range(len(prob_NetGO)):
                temp_list_row=[]
                temp_list_row.append(prob_NetGO[index11])
                temp_list_row.append(prob_FFPredGAN[index11])
                training_features.append(temp_list_row)
            training_features=np.asarray(training_features,np.float)

            param_Grid = [
                {'penalty': ['l2', 'none'], 'solver': ['lbfgs'], 'class_weight':['balanced',None], 'C': [1, 10, 0.1, 100, 0.01, 1000, 0.001, 1e4, 1e-4]},
                {'penalty': ['l2', 'none'], 'solver': ['newton-cg'], 'class_weight':['balanced',None], 'C': [1, 10, 0.1, 100, 0.01, 1000, 0.001, 1e4, 1e-4]},
                {'penalty': ['l1', 'l2'], 'solver': ['liblinear'], 'class_weight':['balanced',None], 'C': [1, 10, 0.1, 100, 0.01, 1000, 0.001, 1e4, 1e-4]},
                {'penalty': ['l2', 'none'], 'solver': ['sag'], 'class_weight':['balanced',None], 'C': [1, 10, 0.1, 100, 0.01, 1000, 0.001, 1e4, 1e-4]},
                {'penalty': ['l1', 'l2', 'none'], 'solver': ['saga'], 'class_weight':['balanced',None], 'C': [1, 10, 0.1, 100, 0.01, 1000, 0.001, 1e4, 1e-4]},
            ]

            PRValue = make_scorer(average_precision_score)
            grid = GridSearchCV(estimator=LogisticRegression(random_state=2048), param_grid=param_Grid, scoring=PRValue)
            grid.fit(training_features, true_Label)
            para = grid.best_params_
            c_value = float(para['C'])
            pen = para['penalty']
            solv = para['solver']
            classw = para['class_weight']

            logisticRegr = LogisticRegression(random_state=2048, penalty=pen, solver=solv, C=c_value, class_weight=classw)
            trained_model=logisticRegr.fit(training_features, true_Label)
            GOTerms_Model_Dict[Cooccure_GO_Terms_unique[index1]]=trained_model
            filename = '.../Fold_' + str(fold_index) + '/'+Cooccure_GO_Terms_unique[index1].split(":")[0]+Cooccure_GO_Terms_unique[index1].split(":")[1]+'_logisticRegression_model.sav'
            joblib.dump(trained_model, filename)
            loaded_model = joblib.load(filename)
            debug_prediction=loaded_model.predict(training_features)
            prob_NetGO.clear()
            prob_FFPredGAN.clear()
            true_Label.clear()
            Valid_GOTerm.append(Cooccure_GO_Terms_unique[index1])
        else:
            prob_NetGO.clear()
            prob_FFPredGAN.clear()
            true_Label.clear()

    fileWriter_GOTerm_Model_List = open(".../Fold_" + str(fold_index) + "/GOTerm_Model_List.txt", "w")
    for index10 in range(len(Valid_GOTerm)):
        fileWriter_GOTerm_Model_List.write(Valid_GOTerm[index10]+"\n")
    fileWriter_GOTerm_Model_List.flush()
    fileWriter_GOTerm_Model_List.close()

    GOTerms_Model_Dict={}
    with open(".../Fold_" + str(fold_index) + "/GOTerm_Model_List.txt") as Reader_GOTerm_Model_List:
        GOTerm_Model_List=Reader_GOTerm_Model_List.read().splitlines()
    for index1 in range(len(GOTerm_Model_List)):
        GOID=GOTerm_Model_List[index1]
        GOID_1=GOTerm_Model_List[index1].split(":")[0]
        GOID_2=GOTerm_Model_List[index1].split(":")[1]
        filename = ".../Fold_" + str(fold_index) + "/"+GOID_1+GOID_2+"_logisticRegression_model.sav"
        loaded_model = joblib.load(filename)
        GOTerms_Model_Dict[GOID] = loaded_model

    with open(".../Fold_" + str(fold_index) + "/NetGO_Fold_2.txt") as Reader_Fold_2_NetGO:
        Fold_2_NetGO=Reader_Fold_2_NetGO.read().splitlines()
    with open(".../Fold_" + str(fold_index) + "/FFPredGAN_Fold_2.txt") as Reader_Fold_2_FFPredGAN:
        Fold_2_FFPredGAN=Reader_Fold_2_FFPredGAN.read().splitlines()

    fileWriter_IntegrateProb = open(".../Fold_" + str(fold_index) + "/fileWriter_IntegrateProb.txt", "w")

    for index1 in range(len(Fold_2_NetGO)):
        temp_array=Fold_2_NetGO[index1].split("%")
        if temp_array[1] in GOTerms_Model_Dict:
            prob_NetGO=float(temp_array[2])
            booleanMarkNotFoundFFPredGAN=True
            for index2 in range(len(Fold_2_FFPredGAN)):
                if Fold_2_FFPredGAN[index2].startswith(temp_array[0]+"%"+temp_array[1]):
                    prob_FFPredGAN=float(Fold_2_FFPredGAN[index2].split("%")[2])
                    Pred_Model=GOTerms_Model_Dict.get(temp_array[1])
                    testing_features=[]
                    testing_features2 = []
                    testing_features.append(prob_NetGO)
                    testing_features.append(prob_FFPredGAN)
                    testing_features2.append(testing_features)
                    testing_features2=np.asarray(testing_features2)
                    integrated_prob=Pred_Model.predict_proba(testing_features2)
                    fileWriter_IntegrateProb.write(temp_array[0]+"%"+temp_array[1] + "%" + str(integrated_prob[0][1]) + "\n")
                    booleanMarkNotFoundFFPredGAN=False
                    break
            if booleanMarkNotFoundFFPredGAN==True:
                fileWriter_IntegrateProb.write(Fold_2_NetGO[index1] + "\n")
        else:
            fileWriter_IntegrateProb.write(Fold_2_NetGO[index1]+"\n")

    fileWriter_IntegrateProb.flush()
    fileWriter_IntegrateProb.close()

    Target_Fold_1.clear()
    Target_Fold_2.clear()
    NetGO_Fold_1.clear()
    FFPredGAN_Fold_1.clear()
    TrueLabel_Fold_1.clear()
    NetGO_Fold_2.clear()
    FFPredGAN_Fold_2.clear()
    TrueLabel_Fold_2.clear()
    Targets_List_Fold1_Int.clear()
    Cooccure_Dict.clear()
    Cooccure_Prob_NetGO.clear()
    Cooccure_Prob_FFPredGAN.clear()
    Cooccure_GO_Terms.clear()
    Cooccure_Protein.clear()
    Valid_GOTerm.clear()
    GOTerms_Model_Dict.clear()

    with open(".../Fold_" + str(fold_index) + "/fileWriter_IntegrateProb.txt") as Reader_IntegrateProb_Raw:
        IntegrateProb_Raw=Reader_IntegrateProb_Raw.read().splitlines()

    for index1 in range(len(IntegrateProb_Raw)):
        tempArray=IntegrateProb_Raw[index1].split("%")
        TargetID=tempArray[0]
        GOTerm=tempArray[1]
        Prob=tempArray[2]
        Prob=str(round(float(Prob), 2))
        if Prob!="0.00" and Prob!="0.0":
                if Prob=="0.1":
                    fileWriterIntegration.write(TargetID + " " + GOTerm + " " + "0.10" + "\n")
                if Prob=="0.2":
                    fileWriterIntegration.write(TargetID + " " + GOTerm + " " + "0.20" + "\n")
                if Prob=="0.3":
                    fileWriterIntegration.write(TargetID + " " + GOTerm + " " + "0.30" + "\n")
                if Prob=="0.4":
                    fileWriterIntegration.write(TargetID + " " + GOTerm + " " + "0.40" + "\n")
                if Prob=="0.5":
                    fileWriterIntegration.write(TargetID + " " + GOTerm + " " + "0.50" + "\n")
                if Prob=="0.6":
                    fileWriterIntegration.write(TargetID + " " + GOTerm + " " + "0.60" + "\n")
                if Prob=="0.7":
                    fileWriterIntegration.write(TargetID + " " + GOTerm + " " + "0.70" + "\n")
                if Prob=="0.8":
                    fileWriterIntegration.write(TargetID + " " + GOTerm + " " + "0.80" + "\n")
                if Prob=="0.9":
                    fileWriterIntegration.write(TargetID + " " + GOTerm + " " + "0.90" + "\n")
                if Prob=="1.0":
                    fileWriterIntegration.write(TargetID + " " + GOTerm + " " + "1.00" + "\n")
                if Prob!="0.1" and Prob!="0.2" and Prob!="0.3" and Prob!="0.4" and Prob!="0.5" and Prob!="0.6" and Prob!="0.7" and Prob!="0.8" and Prob!="0.9" and Prob!="1.0":
                    fileWriterIntegration.write(TargetID+" "+GOTerm+" "+Prob+"\n")
fileWriterIntegration.flush()
fileWriterIntegration.close()

