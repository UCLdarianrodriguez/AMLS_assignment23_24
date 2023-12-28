
""" Executing task A"""
#from A.helper_functions import MNISTDataManager
from A import helper_functions as hf

def pneumonia_task():

    """ Execute the whole code related to task A
        pneumonia classification
    """
    
    # Load pneumonia dataset
    pneumonia = hf.MNISTDataManager("pneumoniamnist")
    pneumonia.load_data()
    pneumonia.split_dataset(flat=True)
    pneumonia.barplot_categories("Pneumonia classes distribution")


