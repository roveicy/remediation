import os
import argparse
import joblib
import pandas as pd
import time
from sktime.classification.dictionary_based import MUSE
from sktime.datasets import load_from_tsfile_to_dataframe
from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_score, recall_score, f1_score
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested
from sagemaker_containers.beta.framework import content_types, encoders

if __name__ == "__main__":
    print("Training Started")
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args = parser.parse_args()

    filepath_train = os.path.join(args.train, 'train.ts')
    filepath_test = os.path.join(args.test, 'test.ts')

    X_train, y_train = load_from_tsfile_to_dataframe(filepath_train)
    X_test, y_test = load_from_tsfile_to_dataframe(filepath_test)
    
    clf = MUSE(n_jobs=-1)
    
    #Train
    clf.fit(X_train, y_train)
    print("Training Completed")

    #Predict
    print("Evaluating performance on test set")
    pred_time_start = time.time()
    y_pred = clf.predict(X_test)
    pred_time_stop = time.time()
    pred_time = pred_time_stop - pred_time_start
    
    #Model Evaluation for training set
    
    #Model Evaluation for training set

    #Prediction time
    print(f'Pred_time={pred_time}')

    #Accuracy = (TP+TN)/total -- how often is the classifier correct
    print(f'Accuracy={accuracy_score(y_test, y_pred)}')

    #Precision = TP / (TP+FP) - ratio of true positives to total predictive positives
    #Used when the occurence of false positives is unacceptable
    print(f'Precision={precision_score(y_test, y_pred, pos_label="1")}')

    #Recall =  (TP/(TP+FN)) - ratio of true positives to total actual positives in the data
    #Used when the occurence of false negatives is unacceptable
    print(f'Recall={recall_score(y_test, y_pred, pos_label="1")}')

    # F Score is the weighted average of the recall and precision
    print(f'F1={f1_score(y_test, y_pred, pos_label="1")}')

    # Save model
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))

    print("Training Completed")

    # Save model
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))

    print("Training Completed")


def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

def input_fn(input_data, request_content_type):
    if request_content_type == 'application/x-npy':
        np_array = encoders.decode(input_data, request_content_type)
        panel_data = pd.DataFrame({'dim_0': pd.Series(np_array)})
        return panel_data
    else:
        raise ValueError("{} not supported by script.".format(request_content_type))

def predict_fn(input_data, model):
    y_predict = model.predict(input_data)
    return y_predict
    

    