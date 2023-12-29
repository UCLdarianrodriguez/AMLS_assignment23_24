
""" Executing task A"""

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from A import helper_functions as hf


def pneumonia_task():

    """ Execute the whole code related to task A
        pneumonia classification
    """
    
    # Load pneumonia dataset
    pneumonia = hf.MNISTDataManager("pneumoniamnist")
    pneumonia.load_data()
    pneumonia.split_dataset(flat=True) #choose to flatten the image
    pneumonia.barplot_categories("Pneumonia classes distribution")

    # Using SVM with polynomial kernel
    degrees = range(1, 10)  # try degrees from 1 to 10
    scores = hf.iterate_hyper(pneumonia,degrees,kernel_type= 'poly')

    # Save plot evaluating the performance over different degrees
    hf.plot_hyper_tuning(degrees,scores,"F1-Score over polynomial degree (SVM)","Degree",name="Polynomial degrees scores")

    # Using SVM with RBF kernel
    c = [0.01, 0.1, 0.2, 0.3,0.5,0.8,1,2,20,100] # SVM regularization parameter
    scores = hf.iterate_hyper(pneumonia,c,kernel_type= 'rbf')

    # Save plot evaluating the performance over different C values
    hf.plot_hyper_tuning(c,scores,"F1-Score over RBF C parameter (SVM)","C",name="C Regularisation scores (SVM)",log=True)

    # Train final models SVM one polynomial and another rbf
    svm_poly = hf.train_svm(pneumonia.x_train, pneumonia.y_train,degree = 3,kernel_type= 'poly',gamma='scale')
    svm_rbf = hf.train_svm(pneumonia.x_train, pneumonia.y_train,C = 2,kernel_type= 'rbf',gamma='scale')

    # Compute ROC Curve for the best polynomial and rbf SVM to choose one
    models = [svm_poly,svm_rbf]
    labels = ["svm_poly","svm_rbf"]
    hf.models_roc(pneumonia.x_val,pneumonia.y_val,models,labels,name="ROC Curve SVM")

    # Compute confusion matrix for rbf SVM on validation set
    y_pred = svm_rbf.predict(pneumonia.x_val)
    y_real = pneumonia.y_val
    hf.report_results(y_pred,y_real,"./A/figures/RBF confusion matrix validation set")

    # Using PCA with SVM rbf
    dimensions_list = range(1,15) #List of dimensions to try
    scores = hf.PCA_SVM(pneumonia, dimensions_list) # score per dimension

    # Save plot evaluating the performance over different PCA components
    hf.plot_hyper_tuning(dimensions_list,scores,"F1-Score over diffent PCA-SVM","Components",name="PCA components scores (SVM)")

    # Selecting only the first 9 components based on the past graph
    svm_pca,pca,scaler = hf.PCA_model(pneumonia,9)
    x_val_scaled = scaler.transform(pneumonia.x_val)
    x_val_pca = pca.transform(x_val_scaled)

    # Predict using the trained classifier
    y_pred = svm_pca.predict(x_val_pca)

    # Report on validation set the PCA results
    hf.report_results(y_pred,pneumonia.y_val,"./A/figures/PCA confusion matrix validation set")

    # Create decision tree model
    tree = hf.tree_model(pneumonia)

    # Predict on the validation set
    y_pred = tree.predict(pneumonia.x_val)

    # Save plot evaluating the performance
    hf.report_results(y_pred,pneumonia.y_val,"./A/figures/Decision tree confusion matrix validation set")

    # Create randon forest model
    forest =  hf.random_forest_model(pneumonia)

    # Predict on the validation set
    y_pred = forest.predict(pneumonia.x_val)

    # Save plot evaluating the performance 
    hf.report_results(y_pred,pneumonia.y_val,"./A/figures/Random Forest confusion matrix validation set")

    # Report ROC Curve on the testing set for the different propose models
    models = [svm_rbf,forest,tree]
    labels = ["svm_rbf","forest","tree"]

    fig, ax = hf.models_roc(pneumonia.x_test,pneumonia.y_test,models,labels,name="")

    # PCA model on test set 
    x_test_scaled = scaler.transform(pneumonia.x_test)
    x_test_pca = pca.transform(x_test_scaled)

    # Predict probabilities for positive class PCA
    y_prob = svm_pca.predict_proba(x_test_pca)[:, 1]

    # Calculate ROC curve and AUC each model
    fpr_model, tpr_model, _ = roc_curve(pneumonia.y_test, y_prob)
    auc_model = roc_auc_score(pneumonia.y_test, y_prob)

    # Plot ROC curves for diffent models
    ax.plot(fpr_model, tpr_model, label=f'PCA-SVM (AUC = {auc_model:.2f})')
    ax.legend()

    # Save plot in the figures folder
    plt.savefig("./A/figures/ROC Curve all Models.png")

    # Plot confusion matrix for all the models on test set
    main_label = "Confusion matrix "

    for index, model in enumerate(models):
        print(labels[index])
        y_pred = model.predict(pneumonia.x_test)
        hf.report_results(y_pred,pneumonia.y_test,"./A/figures/"+main_label+labels[index])

    # Predict x test using PCA-SVM 
    y_pred = svm_pca.predict(x_test_pca)

    # Plot PCA confusion matrix on test set
    print("PCA-SVM")
    hf.report_results(y_pred,pneumonia.y_test,"./A/figures/PCA Confusion matrix")

