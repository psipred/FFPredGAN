# FFPred-GAN

This is a Python implementation of FFPred-GAN method reported in a submitted paper:

Wan, C. and Jones, D.T. (2019) Improving protein function prediction with synthetic feature samples created by generative adversarial networks. bioRxiv: 10.1101/730143.

---------------------------------------------------------------
# Requirements

- Python 3.6 
- Numpy 
- PyTorch
- Scikit-learn
- Standard computer cluster

---------------------------------------------------------------
# Running 

* Step 1. Downloading real training feature samples via `http://bioinfadmin.cs.ucl.ac.uk/downloads/FFPredGAN/RealTrainingData/`.

* Step 2. Generating positive synthetic feature samples by using `./src/Generating_Synthetic_Positive_Samples_FFPred-GAN.py`.
 (This script is used to train a GAN for generating synthetic protein feature samples for single GO term. In order to generate synthetic samples for mutliple GO terms, this script should be duplicated with only changing the GO term ID.)
 
* Step 3. Running Classifier Two-Sample Tests to select the optimal synthetic feature samples by using `./src/Classifier_Two_Sample_Tests.py`.<br/> _* The selected optimal synthetic feature samples can be directly downloaded via `http://bioinfadmin.cs.ucl.ac.uk/downloads/FFPredGAN/SyntheticTrainingData/`_

* Step 4. Downloading testing feature samples and class labels via `http://bioinfadmin.cs.ucl.ac.uk/downloads/FFPredGAN/TestingData/`.

* Step 5. Training and testing support vector machine classifier with positive synthetic feature samples augmented training data by using `./src/Testing_Synthetic_Positive_Samples_Augmented_SVM.py`.


