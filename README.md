# Machine-Learning-Basic
Implementation of some basic machine learning algorithms, which will be divided into 4 groups: _probabilistic model_, _supervised learning_, _unsupervised learning_ and _reinforcement learning (__TODO__)_. 

(Note that the _probabilistic model_ will mainly focus on some old-fashioned ML algorithms such as SVM and PCA, which theoretically should belong to supervised learning or unsupervised learning.)

## 1. Probabilistic Model

### 1.1 Hidden Markov Model (HMM)

* Description: Implement and apply the _forward algorithm_ and _Viterbi algorithm_ on a made-up question (with detailed mathematical derivation).
* Link: [Hidden Markov Model.ipynb](https://github.com/shuheng-cao/Machine-Learning-Basic/blob/master/Probabilistic-Model/HMM/Hidden%20Markov%20Model.ipynb)

### 1.2 Kernel Density Estimation (KDE)

* Description: Implement the kernel density estimation from scratch with three different kernels: _gaussian_, _tophat_ and _epanechnikov_.
* Link: [Kernel Density Estimation.ipynb](https://github.com/shuheng-cao/Machine-Learning-Basic/blob/master/Probabilistic-Model/KDE/Kernel%20Density%20Estimation.ipynb)

### 1.3 Multiclass Logistic Regression (LR)

* Description: Implement the multi-class logistic regression from scratch and apply it to MNIST dataset to get a heat map for weights.
* Link: [Multiclass Logistic Regression.ipynb](https://github.com/shuheng-cao/Machine-Learning-Basic/blob/master/Probabilistic-Model/LR/Multiclass%20Logistic%20Regression.ipynb)

### 1.4 Principal Component Analysis (PCA)

* Description: Implement the PCA from scratch and apply it on iris dataset (with detailed mathematical derivation).
* Link: [Principal Component Analysis.ipynb](https://github.com/shuheng-cao/Machine-Learning-Basic/blob/master/Probabilistic-Model/PCA/Principal%20Component%20Analysis.ipynb)

### 1.5 Support Vector Machine (SVM)

* Description: Implement a soft-margin SVM from scratch with Linear and Gaussian kernels (with detailed mathematical derivation).
* Link: [Support Vector Machine.ipynb](https://github.com/shuheng-cao/Machine-Learning-Basic/blob/master/Probabilistic-Model/SVM/Support%20Vector%20Machine.ipynb)

## 2. Supervised Learning

### 2.1 Convolutional Neural Network (CNN)

* Description: Implement major CNN layers (_Conv2d_, _MaxPool2d_, _Flatten_ etc.) and _Adam Optimizer_ from scratch. Build and train a CNN for MNIST dataset to achieve a test accuracy of 98.4%.
* Link: [Convolutional Neural Network.ipynb](https://github.com/shuheng-cao/Machine-Learning-Basic/blob/master/Supervised-Learning/CNN/Convolutional%20Neural%20Network.ipynb)

### 2.2 Deep Neural Network (DNN)

* Description: Implement a neural network with regulization and different activation/criterion functions from scratch. Build and train (binary/multiclass) classification and regression models on made-up data.
* Link: [Deep Neural Network.ipynb](https://github.com/shuheng-cao/Machine-Learning-Basic/blob/master/Supervised-Learning/DNN/Deep%20Neural%20Network.ipynb)

### 2.3 Long Short Term Memory (LSTM)

* Description: Implement and train a character-based LSTM on tweet data to predict sentiments.
* Link: [Long Short Term Memory.ipynb](https://github.com/shuheng-cao/Machine-Learning-Basic/blob/master/Supervised-Learning/LSTM/Long%20Short%20Term%20Memory.ipynb)

### 2.4 Recurrent Neural Network (RNN)

* Description: Implement the RNN from scratch and train a character-based RNN on _origin of species_ to predict the next character/word.
* Link: [Recurrent Neural Network.ipynb](https://github.com/shuheng-cao/Machine-Learning-Basic/blob/master/Supervised-Learning/RNN/Recurrent%20Neural%20Network.ipynb)

### 2.5 k-Nearest Neighbour (kNN)

* Description: Implement a multi-thread kNN with different distance metric from scratch and train the kNN on MNIST dataset to achieve a test accuracy of 97%.
* Link: [k-Nearest Neighbour.ipynb](https://github.com/shuheng-cao/Machine-Learning-Basic/blob/master/Supervised-Learning/kNN/k-Nearest%20Neighbour.ipynb)

## 3. Unsupervised Learning

### 3.1 Autoencoder (AE)

* Description: Implement and train an autoecoder on MNIST dataset using the DNN from __2.2__ and visualize the hidden states in 3D.
* Link: [Autoencoder.ipynb](https://github.com/shuheng-cao/Machine-Learning-Basic/blob/master/Unsupervised-Learning/AE/Autoencoder.ipynb)

### 3.2 Gaussian Mixture Model (GMM)

* Description: Implement the GMM from scratch and visualize it using made-up data (with detailed mathematical derivation).
* Link: [Gaussian Mixture Model.ipynb](https://github.com/shuheng-cao/Machine-Learning-Basic/blob/master/Unsupervised-Learning/GMM/Gaussian%20Mixture%20Model.ipynb)

### 3.3 Self-organizing Map (SOM)

* Description: Implement a GPU-version SOM from scratch and train it on MNIST dataset to get a 30 by 30 mapping.
* Link: [Self-organizing Map.ipynb](https://github.com/shuheng-cao/Machine-Learning-Basic/blob/master/Unsupervised-Learning/SOM/Self-organizing%20Map.ipynb)

### 3.4 Variational Autoencoder (VAE)

* Description: Implement 3 versions (with DNN, CNN and ResNet as backbones separately) of VAES and train them on the galaxy dataset.
* Link: [Variational Autoencoder.ipynb](https://github.com/shuheng-cao/Machine-Learning-Basic/blob/master/Unsupervised-Learning/VAE/Variational%20Autoencoder.ipynb)

### 3.5 k-Mean Clustering (kMean)

* Description: Implement the k-mean clustering algorithm from scratch and visualize the evolution history of EM algorithm.
* Link: [k-Mean Clustering.ipynb](https://github.com/shuheng-cao/Machine-Learning-Basic/blob/master/Unsupervised-Learning/kMean/k-Mean%20Clustering.ipynb)
