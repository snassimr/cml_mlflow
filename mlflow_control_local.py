from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import mlflow
import mlflow.sklearn
mlflow.sklearn.autolog(disable = True)

def load_iris_data():
    data = load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 10)
    input_data = (X_train, X_test, y_train, y_test)
    return input_data

def train_predict_evaluate_dtree(input_data, params , mlf):
    X_train, X_test, y_train, y_test = input_data
    clf = DecisionTreeClassifier(random_state=42, max_leaf_nodes=params['leaf_nodes'], max_depth=params['max_depth'])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    test_f1_score = metrics.f1_score(y_test, y_pred, average='weighted')
    
    test_metrics = {'test_accuracy' : test_accuracy,
                    'test_f1_score' : test_f1_score}
    
    mlf.log_param_dict(params)
    mlf.log_metric_dict(test_metrics)

    return clf, test_metrics

import sys
SYS_CML_DIR       = "C:\\Users\\nmatatov\\OneDrive - NI\\cml_mlflow"
if (not(SYS_CML_DIR in sys.path)):
   sys.path.insert(0,SYS_CML_DIR)
   
from mlflow_utils import *
input_data = load_iris_data()
mlf = MLflower(APP_ACTIVE_RUN_ID)

params = {'leaf_nodes': 3, 'max_depth' :2}
model, test_metrics = train_predict_evaluate_dtree(input_data, params, mlf)