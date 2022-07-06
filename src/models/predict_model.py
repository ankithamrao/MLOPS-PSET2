import pickle
import json
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score,confusion_matrix
from sklearn.linear_model import ElasticNet, LogisticRegression
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging
import pandas as pd
import numpy as np

from src.config import Config



mlflow.set_tracking_uri("https://dagshub.com/ankithamrao/MLOPS-PSET2.mlflow")
tracking_uri = mlflow.get_tracking_uri()
print("Current tracking uri: {}".format(tracking_uri))

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

with mlflow.start_run():
    with open ("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)
    
    model_type = params['model_type']
    alpha = params['train']['alpha']
    l1_rate = params['train']['l1_rate']
    
    X_test = pd.read_csv(str(Config.FEATURES_PATH / "test_features.csv"))
    y_test = pd.read_csv(str(Config.FEATURES_PATH / "test_labels.csv"))
    
    model = pickle.load(open(str(Config.MODELS_PATH / "model.pickle"), "rb"))
    
    y_pred = model.predict(X_test)
    (rmse, mae, r2) = eval_metrics(y_test, y_pred)
    #acc = accuracy_score(y_test,y_pred)
    #cnf_mat = confusion_matrix(y_test,y_pred)
    
    if model_type == "ElasticNet":
        #pass
        print(f"ElasticNet model : alpha = {alpha}, l1_rate = {l1_rate}")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_rate)
    
    if model_type == "RandomForestRegressor":
        pass
        #print(f"RandomForestRegressor model")
    
    print(f"RMSE : {rmse}\nMAE : {mae}\nR2 : {r2}")
    #print(f"Accuracy score: {accuracy_score}")
    #print(f"Confision matrix : {cnf_mat}")
    
    #For dvc we just write this out as regular data and track it later
    with open(str(Config.METRICS_FILE_PATH), "w") as outfile:
        json.dump(dict(rmse=rmse, mae=mae,r2=r2), outfile)
    
    #with open(str(Config.PLOTS_FILE_PATH), "w") as outfile:
    #    json.dump(dict(accuracy_score=accuracy_score, confusion_matrix=cnf_mat), outfile)
    

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(model, "model")
    """
    ax0.set_ylabel('Target predicted')
    ax0.set_xlabel('True Target')
    ax0.set_title('Ridge regression \n without target transformation')
    ax0.text(100, 1750, r'$R^2$=%.2f, MAE=%.2f' % (
    r2_score(y_test, y_pred), median_absolute_error(y_test, y_pred)))
    """