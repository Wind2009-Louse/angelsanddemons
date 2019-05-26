from model import *
import sys

if __name__ == "__main__":
    #get_train_models("train_small.csv")
    if len(sys.argv) == 2:
        test_name = sys.argv[1]
    elif len(sys.argv) == 1:
        test_name = "test.csv"
        get_train_models()
        show_models_state()
        model_cut()
    else:
        error_msg = "Wrong parameter! You need to run as 'main.py [data_filename]'."
        a = input(error_msg)
        raise(error_msg)
    testdata_judge(test_name)