""" Functions to control tasks"""

# Importing modules
import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO
import numpy as np
import seaborn as sns

from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)


class MNISTDataManager:
    """Class for loading and managing the MNIST dataset."""

    def __init__(self, data_flag):
        """Initialize attributes"""
        self.x_train, self.y_train = None, None
        self.x_val, self.y_val = None, None
        self.x_test, self.y_test = None, None
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.data_flag = data_flag

    def load_data(self):
        """
        Load the MNIST dataset and split it into train, validation, and test sets
        """

        download = True
        info = INFO[self.data_flag]
        data_class = getattr(medmnist, info["python_class"])

        # Download and load the PneumoniaMNIST dataset
        self.train_dataset = data_class(split="train", download=download)
        self.validation_dataset = data_class(split="val", download=download)
        self.test_dataset = data_class(split="test", download=download)

    def split_dataset(self, flat: bool = False):
        """Split it into train, validation, and test sets.

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

    def barplot_categories(self, angle,name: str):
        """
        Create a bar plot showing the distribution of classes
        comparing training, validation and testing sets.

        Args:
            angle(int): rotation angle for xlabels
            name (str) : specify the file name
        """
        fig, ax = plt.subplots(1, 3, figsize=(30, 10))

        ys = [self.y_train, self.y_val, self.y_test]

        datasets_dict = [
            self.train_dataset.__dict__,
            self.validation_dataset.__dict__,
            self.test_dataset.__dict__,
        ]

        for i in range(3):
            # Calculating value counts
            unique_values, value_counts = np.unique(ys[i], return_counts=True)
            get_dict = datasets_dict[i]
            labels_dict = get_dict["info"]["label"]

            # Extract labels
            value_labels = [labels_dict[str(value)] for value in unique_values]

            # Plot bar
            bar_container = ax[i].bar(value_labels, value_counts)

            # Plot number of each bar
            ax[i].bar_label(bar_container, fmt="{:,.0f}", fontsize=12)
            ax[i].set_xticklabels(value_labels, rotation=angle)

        # Setting axis titles
        ax[0].set_title("Training set", fontsize=14)
        ax[1].set_title("Validation set", fontsize=14)
        ax[2].set_title("Testing set", fontsize=14)
        ax[0].set_ylabel("Number of samples")
        plt.tight_layout()


        # Save plot in the figures folder
        plt.savefig(f"{name}.png")

    def plot_images(self):
        """ Plot some images in the dataset """
        num_total = len(self.train_dataset.imgs)
        # class_names = self.y_train
        images = self.train_dataset.imgs

        # Select color mapping
        if self.train_dataset.__dict__["info"]["n_channels"] == 1:
            mapping = "grey"
        else:
            mapping = "viridis"

        plt.subplots(3, 3, figsize=(5, 5))

        for i, k in enumerate(np.random.randint(num_total, size=9)):  # Take a random sample of 9 images
            im = images[k]
            plt.subplot(3, 3, i + 1)
            plt.imshow(im, cmap=mapping)
            plt.axis("off")  # Turn off axis labels
        plt.tight_layout()
        plt.show()


def train_svm(x, y, kernel_type, **kwargs):
    """
    Train a Support Vector Machine model

    Args:
        x: Input features
        y: Target variable
        kernel_type: Type of kernel for SVM (e.g., 'linear', 'rbf', etc.)
        **kwargs: Additional arguments for the SVM model

    Returns:
        trained_model: Trained SVM model
    """

    # Create an SVM classifier
    svm_classifier = SVC(kernel=kernel_type, probability=True, **kwargs)

    # Train the classifier on the training data
    svm_classifier.fit(x, y)

    return svm_classifier


def plot_hyper_tuning(
    x_values, score_list, title: str, xlabel: str, name: str, log: bool = False
):
    """
    lineplot for hyperparamter tuning visualization

    Args:
        x_values (list): list of paramters tried
        score_list (list): list scores values
        title (str): plot title
        xlabel (str): x label for plot
        name(str): filename
        log(bool): select if x axis is logaritmic
    """
    plt.figure()
    plt.plot(x_values, score_list, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("F1-Score")
    plt.xticks(x_values)
    plt.grid(True)
    plt.title(title)
    if log:
        plt.xscale("log") #plot logaritmic scale

    # Define the path where you want to save the plot
    folder_path = "./A/figures"

    # Save plot in the figures folder
    plt.savefig(f"{folder_path}/{name}.png")


def iterate_hyper(dataset, iteractions, kernel_type) -> list:
    """
    Analyze the performance for different hyperparameter values in SVM

    Args:
        dataset (MNISTDataManager): instance of class MNISTDataManager
        iteractions (list): list of values to try
        kernel_type: Type of kernel for SVM


    Returns:
        scores(list): list of F1-scores per degree
    """

    scores = []

    for i in iteractions:
        if kernel_type == "poly":
            classifier = train_svm(
                dataset.x_train,
                dataset.y_train,
                degree=i,
                kernel_type="poly",
                gamma="scale",
            )
        else:
            classifier = train_svm(
                dataset.x_train, dataset.y_train, C=i, kernel_type="rbf", gamma="scale"
            )

        # Predict using the trained classifier
        y_pred = classifier.predict(dataset.x_val)

        # Calculate accuracy
        score = f1_score(dataset.y_val, y_pred)

        scores.append(score)

    return scores

def models_roc(x_eval,y_eval,models:list,labels:list,name:str=""):

    """
    Generate ROC curves for multiple models evaluated on given data.

    Args:
        x_eval (array-like): features on which the models are evaluated.
        y_eval (array-like): Ground truth labels corresponding to the evaluation data.
        models (list) : A list containing multiple models 
        labels (list) : list of labels for each model in the 'models' list.
        name(str): filename

    """    
    
    # creating the plot
    fig, ax = plt.subplots(1, figsize=(8, 6))

    # Plotting the diagonal line for reference (random classifier)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')

    for index,model in enumerate(models):

        # Predict probabilities for positive class
        y_prob = model.predict_proba(x_eval)[:, 1]

        # Calculate ROC curve and AUC each model
        fpr_model, tpr_model, _ = roc_curve(y_eval, y_prob)
        auc_model = roc_auc_score(y_eval, y_prob)

        # Plot ROC curves for diffent models
        ax.plot(fpr_model, tpr_model, label=f'{labels[index]} (AUC = {auc_model:.2f})')


    # Set plot labels and title
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve for Different Models')
    ax.legend()
    ax.grid(True)

    if name == "":
        # Define the path where you want to save the plot
        folder_path = "./A/figures"

        # Save plot in the figures folder
        plt.savefig(f"{folder_path}/{name}.png")

    return fig,ax

def report_results(y_pred,y_real,path):

    """
    Generate report for the Classifier.

    Args:
        y_pred (array-like): Predicted labels generated by the classifier.
        y_real (array-like): Ground truth (real) labels.
        path (str) : Path or file location to report the results.
    """

    # Calculate accuracy
    accuracy = f1_score(y_real, y_pred)
    print(f"F1-Score of the Classifier: {accuracy}")

    # Generate classification report
    report = classification_report(y_real, y_pred)
    print("Classification Report:")
    print(report)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_real, y_pred)

    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted 0', 'Predicted 1'], 
                yticklabels=['Actual 0', 'Actual 1'])

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # Save plot in the path
    plt.savefig(f"{path}.png")

def PCA_model(dataset,dimension):

    """
    Perform PCA dimensionality reduction followed by SVM training on the dataset.

    Args:
        dataset (MNISTDataManager): instance of class MNISTDataManager 
        dimension(int) : integer specifying the number of dimensions/components for PCA.

    Returns:
        svm_classifier : SVM model
        pca: pca instance
        scaler: standard scaler instance used to fit model
    """
        
    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on training data and transform both training and test data
    x_train_scaled = scaler.fit_transform(dataset.x_train)

    # Initialize PCA with the number of components
    pca = PCA(n_components=dimension)

    # Fit PCA to the data
    pca.fit(x_train_scaled)

    # Fit PCA on your data
    x_train_pca = pca.transform(x_train_scaled)

    #print("Transformed shape:", x_train_pca.shape)

    # Create an SVM classifier (you can choose a kernel and set parameters)
    svm_classifier = train_svm(dataset.x_train, dataset.y_train,C = 2,kernel_type= 'rbf',gamma='scale')

    # Train the classifier on the training data
    svm_classifier.fit(x_train_pca, dataset.y_train)

    return svm_classifier,pca,scaler

def PCA_SVM(dataset, dimensions:list):

    """
    Perform PCA dimensionality reduction for different dimensions

    Args:
        dataset (MNISTDataManager): instance of class MNISTDataManager 
        dimensions(list) : List of integers specifying the number of dimensions/components for PCA.

    Returns:
        scores(list): list of scores calculated
    """

    print("Original shape:", dataset.x_train.shape)

    scores = []

    for dimension in dimensions:

        #Train the model
        svm_classifier,pca,scaler = PCA_model(dataset,dimension)

        # Transform the validation set
        x_val_scaled = scaler.transform(dataset.x_val)
        x_val_pca = pca.transform(x_val_scaled)

        # Predict using the trained classifier
        y_pred = svm_classifier.predict(x_val_pca )

        # generate scores list
        score = f1_score(dataset.y_val, y_pred)
        scores.append(score)

    return scores

def tree_model(dataset):
    """
    Create a decision tree model

    Args:
        dataset (MNISTDataManager): instance of class MNISTDataManager 

    Returns:
        tree : decision tree model
    """
    

    tree_params={
        'criterion':'gini',
        'splitter':'best',
        'max_features': 100,
        'max_depth': 10,
        'min_samples_split':50

    }

    # Initialize the Decision Tree Classifier
    tree = DecisionTreeClassifier(**tree_params,random_state=42)

    # Train the classifier
    tree.fit(dataset.x_train, dataset.y_train)

    return tree


def random_forest_model(dataset):
    """
    Create a random forest model

    Args:
        dataset (MNISTDataManager): instance of class MNISTDataManager 

    Returns:
        forest : random forest model
    """

    # Define parameters for Random Forest
    rf_params = {
        'n_estimators': 100,  # Number of trees in the forest
        'criterion': 'gini',  # Split criterion: 'gini' or 'entropy'
        'max_depth': None,  # Maximum depth of the trees
        'min_samples_split': 2,  # Minimum samples required to split a node
        'min_samples_leaf': 1,  # Minimum samples required at each leaf node
        'max_features': 'sqrt',  # Number of features to consider for the best split
        'random_state': 42  # Random seed for reproducibility
    }


    rfm=RandomForestClassifier(**rf_params)

    #Train the model using the training sets
    rfm.fit(dataset.x_train,dataset.y_train)

    return rfm
