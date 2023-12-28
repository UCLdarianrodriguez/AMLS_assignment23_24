""" Functions to control tasks"""

import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO
import numpy as np


class MNISTDataManager:
    """Class for loading and managing the MNIST dataset."""

    def __init__(self,data_flag):
        """Initialize attributes"""
        self.x_train, self.y_train = None, None
        self.x_val, self.y_val = None, None
        self.x_test, self.y_test = None, None
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.data_flag = data_flag

    def load_data(self):
        """Load the MNIST dataset and split it into train, validation, and test sets
        """

        download = True
        info = INFO[self.data_flag]
        data_class = getattr(medmnist, info['python_class'])

        # Download and load the PneumoniaMNIST dataset
        self.train_dataset = data_class(split='train', download=download)
        self.validation_dataset = data_class(split='val', download=download)
        self.test_dataset = data_class(split='test', download=download)

    def split_dataset(self,flat:bool = False):
        """ Split it into train, validation, and test sets.
        
            Arguments:
                Flat (bool): specify if the you want to flatten the image
        
        """
        self.x_train = self.train_dataset.imgs
        self.y_train = self.train_dataset.labels.flatten()

        self.x_val = self.validation_dataset.imgs
        self.y_val = self.validation_dataset.labels.flatten()

        self.x_test = self.test_dataset.imgs
        self.y_test = self.test_dataset.labels.flatten()


        # Flatten each image 
        if flat:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], -1) 
            self.x_val = self.x_val.reshape(self.x_val.shape[0], -1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)

    def barplot_categories(self,name:str):
        """Create a bar plot showing the distribution of classes
           comparing training, validation and testing sets.

           Args:
            name (str) : specify the file name
        """
        fig, ax = plt.subplots(1,3,figsize=(30, 10))

        ys = [self.y_train,self.y_val,self.y_test]
        datasets_dict = [self.train_dataset.__dict__,self.validation_dataset.__dict__,self.test_dataset.__dict__]

        for i in range(3):

            # Calculating value counts
            unique_values, value_counts = np.unique(ys[i], return_counts=True)
            get_dict = datasets_dict[i]
            labels_dict = get_dict['info']['label']

            # Extract labels
            value_labels  = [labels_dict[str(value)] for value in unique_values]

            # Plot bar
            bar_container = ax[i].bar(value_labels,value_counts)
            
            # Plot number of each bar
            ax[i].bar_label(bar_container, fmt='{:,.0f}',fontsize=12)

        #Setting axis titles
        ax[0].set_title("Training set",fontsize=14)
        ax[1].set_title("Validation set",fontsize=14)
        ax[2].set_title("Testing set",fontsize=14)
        ax[0].set_ylabel("Number of samples")

        # Define the path where you want to save the plot
        folder_path = ("./A/figures" )

        plt.savefig(f"{folder_path}/{name}.png")
      

    def plot_images(self):

        num_total = len(self.train_dataset.imgs)
        #class_names = self.y_train
        images = self.train_dataset.imgs

        #Select color mapping
        if self.train_dataset.__dict__['info']['n_channels'] == 1:
            mapping = 'grey'
        else:
            mapping = 'rgb'

        plt.subplots(3,3,figsize=(5,5))

        for i,k in enumerate(np.random.randint(num_total, size=9)):  # Take a random sample of 9 images 
            im = images[k]                    
            plt.subplot(3,3,i+1)
            #plt.xlabel(classNames[imageClass[k]])
            plt.imshow(im,cmap=mapping)
            plt.axis('off') # Turn off axis labels
        plt.tight_layout()
        plt.show()
