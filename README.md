# COMP3419: Facial Attribute Analysis

Assignment 2 for Graphics and Multimedia at USyd Semester 2 2020


A python notebook to train, test and evaluate a model on the CelebA dataset. The best model has been provided as well as example code for inferencing

## Framework
- PyTorch 1.6.0
- CUDA 10.1
- Torchvision 0.7.0

## Provided 
- PyTorch model ```comp3419_celeba_model```
- list_attr_celeb.txt
- comp3419_a2.ipynb

## Missing
- img_align_celeb.zip 

The notebook provided contains 6 sections:
### Config and Setup

Here you can modify key parameters for training and inference. Specically the path to the model and list attribute text files need to be correct to run infernce with the provided model, depedning on your experimental setup. Training and validation was done via Google Colab and so much of the structure was set up for this such as using the __tmp__ folder to place training and validation data.

However, this should not be a problem for inferencing as the complete paths for both the label attribute and model path are required.

### Data Acquisition - Transfer Learning
Sections for loading the data from the provided location and training, using either an old model or the ResNet core.

### Inferencing ##
Inferencing allows you to load the best model (provided) and see the output on any image. Both the full paths to the model and the __list_attr_celeb.txt__ file are required. The function  ```evaluate``` can be called on an instance of this class with the full path to any image and the function will return the predicted labels as a list of stirngs.

