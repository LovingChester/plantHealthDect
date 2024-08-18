# Plant Health Detection
## Abstract
Early detection of plant diseases plays an important role in agriculture, as it is directly related to the quality and yield of the agricultural products. There are two sources contributing to the diseases, living organisms such as viruses, bacteria, fungi, etc, and environmental factors such as drought, heat, cold, etc. Despite significant efforts being made to overcome these diseases, the problem of quality and yield loss remains crucial. Even today, in most of the places in the world, disease inspections on plants are still done manually. Most of the time, farmers inspect the disease either based on the guideline, which requires a high level of experience and skills, or based on the requested technical support, which is very costly and time consuming. Hence, an automated way to identify the plant diseases is necessary for agricultural development.

## Setting Up
### Installation
We recommand use Anaconda. If you do not have Anaconda installed, please follow the instruction to install the latest version of Anaconda: [Anaconda Link](https://docs.anaconda.com/anaconda/install/index.html).  

After successfually installing Anaconda, open Anaoncda Prompt and install the following necessary packages by typing in the command.  
NumPy: 
`pip install numpy`  
Easydict: 
`pip install easydict`  
[PyTorch](https://pytorch.org/), install PyTorch based on your OS and CUDA version.  
Scikit-Learn: 
`pip install -U scikit-learn`  

### Dataset
The dataset is publicly visible at [PlantVillage](https://paperswithcode.com/dataset/plantvillage). After downloading all the files, unzip them and keep the file structure. Next, using the following command to split the data into training, validation, testing set: `Python train_test_split.py`.

Desired file structure:  
├── src
│   ├── Plant_leave_diseases_dataset_with_augmentation
|   |   |── Apple___Apple_scab
|   |   |   |── image1.jpg
|   |   |   ......
│   |── Plant_leave_diseases_dataset_without_augmentation
│   ├── PlantHealthDect
|   |   |── config.py
|   |   ......

## Training
In order to perform training, you can first go to `config.py` file to adjust the model, hyperparameters, and output directory. For model, you can choose between VGG-16 and Vision Transformer. For hyperparameter, you can adjust learning rate, L2 regularization, number of epoches, etc.  
If you are satisified with the model and hyperparameter, you can start training by using the command ''