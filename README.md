# FFPred-GAN

This is a Python implementation of FFPred-GAN method reported in a submitted paper:

Wan, C. and Jones, D.T. (2019) Improving protein function prediction with synthetic feature samples created by generative adversarial networks. bioRxiv: 10.1101/730143.

---------------------------------------------------------------
# Requirements

- Python 3.6 
- Numpy 
- PyTorch (CPU mode)
- Scikit-learn
- Standard computer cluster

---------------------------------------------------------------
# Running 
It is recommended to use a standard computer cluster that allows to run multiple jobs simultaneously (e.g. the SGE array job).

* Step 1. Downloading real training feature samples via <br/><br/>`http://bioinfadmin.cs.ucl.ac.uk/downloads/FFPredGAN/RealTrainingData/`.

* Step 2. Generating positive synthetic feature samples by using the template <br/><br/>`./src/Generating_Synthetic_Positive_Samples_FFPred-GAN.py`.<br/><br/>
_This template is used to train one GAN for generating synthetic protein feature samples for single GO term. In order to generate synthetic samples for mutliple GO terms, this script should be duplicated with only changing the GO term ID, or changing the absolute pathname of the training feature samples for corresponding GO terms._
 
* Step 3. Running Classifier Two-Sample Tests to select the optimal synthetic feature samples by using <br/><br/>`./src/Classifier_Two_Sample_Tests.py`.<br/><br/> 
_This template is used to select the optimal synthetic protein feature samples generated for single GO term. In order to select the optimal synthetic samples for mutliple GO terms, this script should be duplicated with only changing the GO term ID, or changing the absolute pathname of the real and synthetic feature samples for corresponding GO terms. The selected optimal synthetic feature samples can be directly downloaded via `http://bioinfadmin.cs.ucl.ac.uk/downloads/FFPredGAN/SyntheticTrainingData/`_

* Step 4. Downloading testing feature samples and class labels via <br/><br/>`http://bioinfadmin.cs.ucl.ac.uk/downloads/FFPredGAN/TestingData/`.

* Step 5. Training and testing support vector machine classifier with positive synthetic feature samples augmented training data by using<br/><br/> `./src/Testing_Synthetic_Positive_Samples_Augmented_SVM.py`.


