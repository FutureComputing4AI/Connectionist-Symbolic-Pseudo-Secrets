<h2 align="center">CSPS<br>HRR NETWORK PSEUDO-ENCRYPTION</h2>

## Requirements
- [PyTorch](https://pytorch.org/get-started/locally/) 
- [HRR](https://github.com/MahmudulAlam/Holographic-Reduced-Representations)  ```pip install hrr```

## Intro 
<p align="justify">
Pseudo-encryption method to deploy convolutional networks on an untrusted platform for secure inferential steps and prevent model theft using 2D Holographic Reduced Representations (HRR). By leveraging HRR, we create a neural network with a pseudo-encryption style defense that empirically shows robustness to attack, even under threat models that unrealistically favor the adversary.
</p>

## Demo 
<p align="justify">
Here is an example of 2D HRR operating on the image shown in (a). The bound image of the original image and secret is shown in (b). The retrieved image from the bound image using the secret is shown in (c). 
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/170301330-c7c58023-f2cf-4da9-a08b-4212cf568d5a.png" width="600">
</p>

## Network 
<p align="justify">
The following block diagram illustrates the encryption process of the CNN using improved 2D HRR in three stages. Both of the orange regions are on the user-end. The secrets to unbind the images and outputs of the main network are only shared in these regions (dashed line). The red region indicates the untrusted third party who will run the main network after it has been trained.
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/170298928-b379d6bf-24e8-4e2f-8950-668733b59fa7.png" width="950">
</p>

## Datasets 🗃️
Experiments are performed on the following datasets: 

* MNIST: 28 x 28 - 10 classes
* SVHN: 32 x 32 - 10 classes 
* CIFAR-10: 32 x 32 - 10 classes 
* CIFAR-100: 32 x 32 - 100 classes
* Mini-Imagenet: 84 x 84 - 100 classes 

## Experiments 
Experiments regarding each dataset are separated by each folder by the name of the dataset. Base model experiments are separated by suffixing ```-base``` in the dataset name. Each folder contains separate files for ```network```, ```train```, and ```predict```. ```embed.py``` creates UMAP 2D representation image samples. ```clustering.py``` do the Kmeans, Spectral, Gaussian mixture, Birch, and HDBScan clustering experiments of the paper.  

<p align="justify">
The following table shows the accuracy of the Base model (without secret binding) and CSPS on the five datasets.
</p> 

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/170303129-aab0707e-5ba0-4657-a84a-b6cf5d9510ac.png" width="500">
</p>

<p align="justify">
The accuracy of CSPS drops compared to the Base model due to the extra security feature. However, the lost accuracy can be retrieved by sampling <em>k</em> predictions and taking their ensemble average. The following figure demonstrates how taking the ensemble average of <em>k (1...10)</em> predictions restores the lost accuracy.
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/170303310-922af26b-2d5c-44df-a4a3-6caed991ab06.png" width="550">
</p>

---
## Citations

[![Paper](https://img.shields.io/badge/paper-ICML22-90fc03.svg?longCache=true&style=flat)](http://arxiv.org/abs/2206.05893)

[Deploying Convolutional Networks on Untrusted Platforms Using 2D Holographic Reduced Representations @ ICML 2022](http://arxiv.org/abs/2206.05893)

If you use our work, please cite us using the below bibtex. We will update when the ICML proceedings provide the official citation style. Cite the paper as:

```bibtex
@inproceedings{Alam2022,
  archivePrefix = {arXiv},
  arxivId = {2206.05893},
  author = {Alam, Mohammad Mahmudul and Raff, Edward and Oates, Tim and Holt, James},
  booktitle = {International Conference on Machine Learning},
  eprint = {2206.05893},
  title = {{Deploying Convolutional Networks on Untrusted Platforms Using 2D Holographic Reduced Representations}},
  url = {http://arxiv.org/abs/2206.05893},
  year = {2022}
}

```
---