""" This is the main program """
from A import task as task_A
from B import task as task_B


def main():
    """ 
        Call the code from different tasks
    """
    print("RUNNING TASK A")
    task_A.pneumonia_task()

    print("RUNNING TASK B")
    task_B.multiclass_task()

if __name__ == "__main__":
    main()