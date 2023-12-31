""" Executing task B"""

import tensorflow as tf
from B import helper_functions as hf
from A.helper_functions import MNISTDataManager
import numpy as np


def multiclass_task():
    """ Execute the whole code related to task B
        PathMNIST classification
    """

    # Load pathmnist dataset
    pathmnist = MNISTDataManager("pathmnist")
    pathmnist.load_data()
    pathmnist.split_dataset(flat=False) #choose not to flatten the image

    # Define the path where you want to save the plot
    folder_path = "./B/figures"
    filename = f"{folder_path}/Pathmnist classes distribution"

    pathmnist.barplot_categories(angle=90,name=filename)

    # Parameter values
    batch_size = 64
    epochs =  12
    #dropout_rates = [0, 0.2, 0.3, 0.4, 0.5] #list of dropout values used to evaluate model

    # Dictionary containing model fit parameters
    params = {
        'x': pathmnist.x_train,
        'y': pathmnist.y_train,
        'epochs': epochs,
        'batch_size': batch_size,
        'validation_data': (pathmnist.x_val, pathmnist.y_val)
    }

    # Create instance of the simple neural network
    historic_nn,model_nn = hf.model_neural_network(params)

    # Create instance of the neural network with regularizer and normalization
    historic_reg,model_reg = hf.model_nn_regularizer(0.3,params)

    # Plot model performance during epochs
    hf.plot_accuracy_epochs(historic_nn,"Epochs Accuracy simple NN")
    hf.plot_accuracy_epochs(historic_reg,"Epochs Accuracy NN Regularized")

    # Prediction from the validation set for the simple neural network
    class_labels = np.unique(pathmnist.y_train)
    val_predictions = model_nn.predict(pathmnist.x_val)
    y_pred = tf.argmax(val_predictions, axis=1)


    # Plot Report simple NN
    hf.report_multi_results(y_pred,pathmnist.y_val,class_labels,"Confusion matrix simple nn")

    # Prediction from the validation set for the neural network with regularizer
    val_predictions = model_reg.predict(pathmnist.x_val)
    y_pred = tf.argmax(val_predictions, axis=1)

    # Plot Report regularized NN
    hf.report_multi_results(y_pred,pathmnist.y_val,class_labels,"Confusion matrix regularized nn")

    # Create instance of the simple CNN
    historic_cnn,model_cnn = hf.model_cnn_simple(params)

    # Create instance of the CNN with batch normalization
    historic_cnn_bn,model_cnn_bn = hf.model_cnn_bn(params)

    # Plot model performance during epochs
    hf.plot_accuracy_epochs(historic_cnn,"Epochs Accuracy simple CNN")
    hf.plot_accuracy_epochs(historic_cnn_bn,"Epochs Accuracy CNN BN")

    # Prediction from the validation set for the simple CNN
    val_predictions = model_cnn.predict(pathmnist.x_val)
    y_pred = tf.argmax(val_predictions, axis=1)

    # Plot Report simple CNN
    hf.report_multi_results(y_pred,pathmnist.y_val,class_labels,"Confusion matrix CNN BN")

    # Prediction from the validation set for the CNN Batch normalization
    val_predictions = model_cnn_bn.predict(pathmnist.x_val)
    y_pred = tf.argmax(val_predictions, axis=1)

    # Plot Report CNN Batch normalization
    hf.report_multi_results(y_pred,pathmnist.y_val,class_labels,"Confusion matrix CNN BN")

    # Report ROC Curve on the testing set for the different propose models
    models = [model_nn,model_reg,model_cnn,model_cnn_bn]
    labels = ["NN","NN regularizer","CNN","CNN normalization"]
    class_labels = np.unique(pathmnist.y_train)

    for index,model in enumerate(models):

        # Prediction from test set
        test_predictions = model.predict(pathmnist.x_test)
        y_pred = tf.argmax(test_predictions, axis=1)

        # Plots to save on test set
        chart = hf.report_bar(pathmnist.y_test,y_pred,f"F1 score {labels[index]} TS")
        hf.report_multi_results(y_pred,pathmnist.y_test,class_labels,f"{labels[index]} Confusion matrix TS")

        hf.roc_multiclass(test_predictions,pathmnist.y_test,labels[index],f"{labels[index]} ROC Curve")









