""" Functions to control tasks"""

# Importing modules
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns

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
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
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
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_value),
        layers.Dense(64, activation='relu'),
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

def report_multi_results(y_pred,y_real,labels,name:str):
    """
    Generate a classification report and confusion matrix for a multi-class classification problem.

    Arguments:
        y_pred (array-like): Predicted labels
        y_real (array-like): True labels
        labels (list): List of class labels
        name (str): filename

    Returns:
    """

    # Generate classification report
    report = classification_report(y_real, y_pred)
    print("Classification Report:")
    print(report)

    # Confusion Matrix
    print("Confusion Matrix:")

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_real, y_pred)
    print(confusion_matrix)

    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
   
    # Define the path where you want to save the plot
    folder_path = "./B/figures"

    # Save plot in the figures folder
    plt.savefig(f"{folder_path}/{name}.png")