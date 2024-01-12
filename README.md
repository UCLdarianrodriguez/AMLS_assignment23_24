# ELEC0134: Applied Machine Learning Systems Project

## Project Description

The project consists in a binary-class classification tasks and a multi-class classification task:

### Task A:

Classify an image onto "Normal" (no pneumonia) or "Pneumonia" using the PneumoniaMNIST dataset with 5,856 pediatric chest X-Ray images, that have been preprocessed by resizing and center-cropping to a standardized 28x28 grayscale format. 

Class labels:

* `Normal`: 0
* `Pneumonia`: 1

### Task B

Classify an image onto 9 different types of tissues, using the PathMNIST dataset derived from a study to predict survival from colorectal cancer histology slides. The images were resized from 3×224×224 to 3×28×28.

Class labels:

* `Adipose`: 0
* `Background`: 1
* `Debris`: 2
* `Lymphocytes`: 3
* `Mucus`: 4
* `Smooth muscle`: 5
* `Normal colon mucosa`: 6
* `Cancer-associated stroma`: 7
* `Colorectal adenocarcinoma epithelium`: 8


## Project Organization

The main.py calls the modules associated with each task, with each task residing in a separate file following an structure.


### Folder Structure 

* _init_.py: empty file to consider directory as a Python package

* helper_functions.py: contains all the functions to execute the main modules of the task

* task.py:  execute the corresponding task by using the modules or functions in helper_functions 

* "Figures" folder: folder to save all the plots generated while executing the task.

### Other Files

* environment.yml: file for the conda environment  configuration, useful for creating the required envrironment for this project.

* requirements.txt: list the packages that needs to be install for the environment.


## Setup

Run conda env create -f environment.yml && conda activate AML-final && python main.py
this will run the main file after creating the environment and installing the required libraries:

* numpy
* jupyter
* jupyterlab
* pandas
* matplotlib
* scikit-learn
* seaborn
* medmnist
* tensorflow
* keras

The main.py directly executes both tasks, and it may have a long runtime as all the tested models are trained with each run. If you wish to skip specific models, you can directly comment out the relevant code lines. By default, the lines related to hyperparameter tuning in SVM for task A are commented out to speed up execution. However, if you desire to run this part, simply uncomment it in task.py inside folder A. During execution, the code utilizes pre-selected hyperparameter values, as explained in the report.

For example the following lines are commented to skip PCA hyperparameter tuning:

    scores = hf.PCA_SVM(pneumonia, dimensions_list)

    hf.plot_hyper_tuning(dimensions_list,scores,"F1-Score over diffent PCA-SVM","Components",name="PCA components scores (SVM)")

The classification report, specifying the precision, recall, and F1-score for each model, is directly printed in the console.


The “load_data” method in class “MNISTDataManager”, downloads the dataset using medmnist library in .npz format, “split_dataset” divides the dataset into training, validation, and testing sets.


