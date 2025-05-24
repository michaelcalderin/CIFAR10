# Project 3: Image Classification for CIFAR-10 Dataset

## Overview
In Section 1 of this project, an ANN was compared to random forest with PCA and random forest without dimensionality reduction for image classification on CIFAR-10. In Section 2, a transformer fetched from Hugging Face was used on CIFAR-10 and then improvements were attempted with data augmentation, regularization, and learning rate scheduling. As a general note, scripts/notebooks should be extracted to your working directory (removed from *Scripts* folder). For Section 1, it is usually expected that the *Models* folder is unzipped and present in that same working directory. For Section 2, each model is stored as a *..._vit_model* folder and a *.keras* file (both in the *Models* folder). Unlike in Section 1, these are expected to be placed in the working directory and should not stay in the *Models* folder. These are general guidelines and may vary depending on the notebook since they were written before the GitHub repository was properly structured.

## Gradio Interface Demo
- In Section 1, a Gradio interface was developed so that users could upload or draw images and see predictions made by the ANN.
- The Gradio code can be found in the last cell in *Scripts/S1_training.ipynb*. Ensure the notebook and unzipped Models folder is extracted to your working directory. Refer to the demo link for more details.
- Link: https://youtu.be/b1EBNGFZ0WU

## Main Files
- In the *Scripts* folder, files starting with *S1* represent training and testing notebooks used for Section 1.
- In the *Scripts* folder, files starting with *S2* represent training and testing notebooks used for Section 2. Notice that for training there is a "baseline" notebook and "improved" notebook. In the baseline notebook, a Dense output layer was simply added to the main transformer layer to give some sort of baseline for future improvements. In the improved notebook, a series of improvements were attempted to try to beat the baseline performance. All of these models, including baseline and improvement attempts, were evaluated on the test set in the testing notebook.
- A challenge was given to get the best CIFAR-10 accuracy and macro average F1 on an unlabeled test set, so files related to this have "challenge" or "submission" in their title.
- **Report.pdf**: report summarizing the full process and insights

## Running the Notebooks
- All notebooks are found in the *Scripts* folder.
- The models they use are in the *Models* zipped folder which needs to be unzipped.
- Section 1 models were typically *.pkl* files and required the preprocessing pipeline found in the *Pipelines* folder.
- Section 2 models were typically a *..._vit_model* folder and *.keras* file (the folder held information about the original transformer model and the keras file held information about the extra layers and elements adding during training). *general_training.py* had some utility functions to help speed up training and making predictions.

## Dependencies
- Install *TensorFlow, Scikit-learn, NumPy, Matplotlib, Joblib, Gradio, Transformers*

## Using the Models
- Refer to the testing notebooks to see the models loaded in and used for predictions.
- Section 1 models require the preprocessing pipeline in the *Pipelines* folder and the actual models are *.pkl* files.
- Section 2 models require the preprocessing pipeline provided by Hugging Face. The *.keras* file is the model but requires a custom ViTLayer class object to be loaded in which needs the *..._vit_model* folder to read from. Details can be found in the training or testing notebooks and reading *general_training.py* is recommended.

## Update 5/23/2025
- *Models.zip* was removed due to a lack of space. Follow the Jupyter Notebooks if interested in generating the models.
