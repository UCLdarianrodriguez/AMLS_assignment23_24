""" Functions to control tasks"""

# Importing modules
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
import pandas as pd

# Set seeds for reproducibility
tf.random.set_seed(42)

def model_neural_network(params):

    """
    Generate a simple neural network model

    Arguments:
        params (dict): Dictionary containing parameters for model training. It should contain the necessary arguments
                     for the `model.fit()` method, such as `x`, `y`, `epochs`, `batch_size`, etc.

    Returns:
        historic_data: history object from model training
        model: trained model

    """
    # Define the model without batch normalization
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28, 3)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(9, activation='softmax')  # 9 output classes (0-8) with softmax
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model 
    historic_data = model.fit(**params)

    return historic_data,model

def model_nn_regularizer(dropout_value,params):
    """
    Generate a neural network model with batch normalization and  regularizer

    Arguments:
        dropout value (int): Dropout rate value between 0 and 1 
        params (dict): Dictionary containing parameters for model training. It should contain the necessary arguments
                     for the `model.fit()` method, such as `x`, `y`, `epochs`, `batch_size`, etc.

    Returns:
        historic_data: history object from model training
        model: trained model

    """

    # Define the model with normalization
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28, 3)),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_value),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_value),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_value),
        layers.Dense(9, activation='softmax')  # 9 output classes (0-8) with softmax
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model 
    historic_data = model.fit(**params)

    return historic_data,model

def model_cnn_simple(params):

    """
    Generate a simple CNN 

    Arguments:
        params (dict): Dictionary containing parameters for model training. It should contain the necessary arguments
                     for the `model.fit()` method, such as `x`, `y`, `epochs`, `batch_size`, etc.

    Returns:
        historic_data: history object from model training
        model: trained model
    """

    model = models.Sequential([
        # Convolutional layers 1
        layers.BatchNormalization(),
        layers.Conv2D(64, (2, 2), activation='relu', input_shape=(28, 28, 3)),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional layers 2
        layers.Conv2D(64, (2, 2), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten the output of the last convolutional layer
        layers.Flatten(),
        
        # Fully connected layers
        layers.Dense(512, activation='relu'),
        
        # Output layer
        layers.Dense(9, activation='softmax')  
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model 
    historic_data = model.fit(**params)
    
    return historic_data,model

def model_cnn_bn(params):
    """
    Generate a CNN  with bath normalization

    Arguments:
        params (dict): Dictionary containing parameters for model training. It should contain the necessary arguments
                     for the `model.fit()` method, such as `x`, `y`, `epochs`, `batch_size`, etc.

    Returns:
        historic_data: history object from model training
        model: trained model
    """

    model = models.Sequential([
        # Convolutional layers with BatchNormalization
        layers.BatchNormalization(),
        layers.Conv2D(64, (2, 2), input_shape=(28, 28, 3)),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        
        layers.Conv2D(64, (2, 2)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten the output of the last convolutional layer
        layers.Flatten(),
        
        # Fully connected layers with BatchNormalization
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        # Output layer
        layers.BatchNormalization(),
        layers.Dense(9, activation='softmax')  
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model 
    historic_data = model.fit(**params)
    
    return historic_data,model

def plot_accuracy_epochs(history,name:str):
    """
    Generate plot of the model accuracy across the epochs

    Arguments:
        history: history object from model training
        name (str) : specify the file name

    Returns:
    """
     
    plt.figure(figsize=(20,5))

    # Create a list of the epochs
    epochs = range(1,len(history.history['loss'])+1)

    plt.plot(epochs,history.history['accuracy'], label='Training Accuracy',marker='o')
    plt.plot(epochs,history.history['val_accuracy'], label='Validation Accuracy',marker='o')
    plt.xticks(epochs)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Define the path where you want to save the plot
    folder_path = "./B/figures"

    # Save plot in the figures folder
    plt.savefig(f"{folder_path}/{name}.png")


def report_bar(y_pred,y_true,name:str):

    """
    Generate F1 Score per class in bar plot

    Arguments:
        y_pred (array-like): Predicted labels
        y_real (array-like): True labels
        name (str): filename

    Returns:
        report_df(dataframe): pandas df with the evaluation metrics

    """

    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)

    # Create pandas dataframe with the report information
    report_df = pd.DataFrame({'F1 score': fscore, 'Recall': recall, 'Precision': precision})

    # Create a new figure
    plt.figure(figsize = (20,8))

    # Create the bar plot
    bar_container = plt.bar(report_df.index,report_df['F1 score']*100,edgecolor='black', color = sns.color_palette('pastel', 8))

    # Specify the classes
    plt.xticks(range(0,9))

    # Plot number of each bar
    plt.bar_label(bar_container, fmt='{:.2f}%',fontsize=12,fontweight='bold')

    plt.title("F1-Score per class",fontsize=18)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Accuracy%', fontsize=14)

    # Define the path where you want to save the plot
    folder_path = "./B/figures"

    # Save plot in the figures folder
    plt.savefig(f"{folder_path}/{name}.png")

    return report_df

def report_multi_results(y_pred,y_real,labels,path):

    """
    Generate report for the Classifier.

    Args:
        y_pred (array-like): Predicted labels generated by the classifier.
        y_real (array-like): Ground truth (real) labels.
        labels(str): xticks labels
        path (str) : Path or file location to report the results.
    """

    # Generate classification report
    report = classification_report(y_real, y_pred)
    print("Classification Report:")
    print(report)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_real, y_pred)

    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # Save plot in the path
    plt.savefig(f"{path}.png")
