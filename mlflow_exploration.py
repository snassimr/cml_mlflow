
import mlflow
from   mlflow.tracking import MlflowClient
import mlflow.tracking                      # Management Functions

ml_tracking_uri = "file:///C:\\Users\\nmatatov\\Desktop\\ML\\MLFlow\\Experiments"
mlflow.tracking.set_tracking_uri(ml_tracking_uri)

client = MlflowClient()
ml_experiment_id = client.create_experiment("Exp1", artifact_location=None)

# client.delete_experiment('0')                                                          # MlflowException: Could not find experiment with ID 6
ml_experiments   = client.list_experiments(view_type = None)

mlflow_run = mlflow.start_run(experiment_id = '1')                                     # Active Run with system generated ID
mlflow.log_param("a", 1)
mlflow.log_metric("b", 2)
mlflow_run = mlflow.end_run()
print(client.list_run_infos(experiment_id = '1'))

client.delete_run(run_id = 'b814d4eea3e44d5aa3b70eac417c5f82')

mlflow_runName = client.get_tag('mlflow.runName')
print(mlflow.runName)

# The mlflow.tracking module provides a Python CRUD interface to MLflow experiments and runs. 
# This is a lower level API that directly translates to MLflow REST API calls. For a higher level API for managing an “active run”, use the mlflow module.

from  mlflow.tracking import MlflowClient
client = MlflowClient()
ml_experiments   = client.list_experiments(view_type = None)

# list_run_infos(experiment_id, run_view_type=1)
# ml_run_artifacts = list_artifacts(run_id, path = None)

# TO DO : Learn difrence between ml tracking from mlflow and mlflow.tracking client




from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import mlflow
import mlflow.sklearn
mlflow.sklearn.autolog()

def load_iris_data():
    data = load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 10)
    input_data = (X_train, X_test, y_train, y_test)
    return input_data

def train_predict_evaluate_dtree(input_data, params):
    with mlflow.start_run(run_name = "Decision Tree Classifier Experiments"):
        X_train, X_test, y_train, y_test = input_data
        clf = DecisionTreeClassifier(random_state=42, max_leaf_nodes=params['leaf_nodes'], max_depth=params['max_depth'])
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        test_accuracy = metrics.accuracy_score(y_test, y_pred)
        test_f1_score = metrics.f1_score(y_test, y_pred, average='weighted')
        test_metrics = (test_accuracy, test_f1_score)
        
        mlflow.log_metric('test_accuracy' , test_accuracy)
        mlflow.log_metric('test_f1_score', test_f1_score)
        mlflow.log_metric('test_f1_duplicated', test_f1_score)
        mlflow.log_metric('test_f1_duplicated', test_f1_score*10)
    
    return clf, test_metrics

input_data = load_iris_data()
params = {'leaf_nodes': 3, 'max_depth' :2}
model, test_metrics = train_predict_evaluate_dtree(input_data, params)

################################################ Run on another console
### Repalce with mlflow tracking server
import os
cmd_mlflow_ui_up = r'cd C:\Users\nmatatov & mlflow ui'
os.system(cmd_mlflow_ui_up)
# http://localhost:5000/


#  C:\Users\nmatatov\mlruns\0\67ddd043deba4777a4f18b831dbc09f9




################################################################################### MLflow wrapper
import mlflow
ml_tracking_uri = 'file:///C:/Users/nmatatov/mlruns'
mlflow.tracking.set_tracking_uri(ml_tracking_uri)
from  mlflow.tracking import MlflowClient
mlflow_client = MlflowClient()

#### Administrator commands
# mlflow_experiments = mlflow_client.list_experiments()
# mlflow_experiments
# create_experiment(name, artifact_location=None)

#### Intra DS administration
# mlflow.active_run()

#### mlflower
SYS_MLFLOW_EXPERIMENT_NAME = 'Default'
mlflow.set_experiment(SYS_MLFLOW_EXPERIMENT_NAME)

APP_EXPERIMENT_ID = mlflow.get_experiment_by_name(SYS_MLFLOW_EXPERIMENT_NAME).experiment_id
APP_RUN_NAME      = 'run_1'
APP_RUN_TAGS      = {'Tag1': 'Tag1_Value'}
APP_ACTIVE_RUN    = mlflow.start_run(run_name = APP_RUN_NAME, experiment_id = APP_EXPERIMENT_ID , tags = APP_RUN_TAGS)
APP_ACTIVE_RUN_ID = APP_ACTIVE_RUN.info.run_id

APP_EXPERIMENT_ID
APP_ACTIVE_RUN_ID


# https://www.mlflow.org/docs/latest/python_api/mlflow.html

### Automatic push to git
# git add --all my/awesome/stuff/
# git commit -m with APP_RUN_NAME
# git push

       
import os
cmd_git_commit = r'cd C:\\Users\\nmatatov\\OneDrive - NI\\cml_mlflow & git add --all & git commit -m with rrr & git push'
os.system(cmd_git_commit)


class Number:
    def __init__(self, value):
        self.value = value
    def next_number(self):
        self.value += 1
        return self.value
    
# number = Number(1)

# print(number.next_number())
# print(number.next_number())
# print(number.next_number())

################################################### Local run
# https://www.mlflow.org/docs/latest/python_api/mlflow.projects.html#mlflow.projects.run
# https://github.com/mlflow/mlflow-example
mlflow.projects.run(uri, experiment_name = None, experiment_id = None, run_id = None , entry_point = 'main', version = None, 
                    parameters  = None, 
                    docker_args = None, backend = 'local', backend_config = None, use_conda = True, storage_dir = None, synchronous=True)


