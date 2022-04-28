import argparse, os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import joblib

if __name__ == "__main__":

    # Pass in environment variables and hyperparameters
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--estimators", type=int, default=15)

    # sm_model_dir: model artifacts stored here after training
#     parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
#     parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--train", type=str, default='./')
    parser.add_argument("--model_dir", type=str)
    
    
    args, _ = parser.parse_known_args()
    estimators = args.estimators
    model_dir = args.model_dir
#     sm_model_dir = args.sm_model_dir
    training_dir = args.train

    # Read in data
    df = pd.read_csv(training_dir + "/train.csv", sep=",")

    # Preprocess data
    X = df.drop(["class", "class_cat"], axis=1)
    y = df["class_cat"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Build model
    regressor = RandomForestRegressor(n_estimators=estimators)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    # Save model
    joblib.dump(regressor, "./model.joblib")