from pathlib import Path

#Parameters and folders for the different experiments for dvc-example
#RANDOM_SEED = 42
#ALPHA = 0.3
#L1_RATIO = 0.03

class Config:
    MODEL_TYPE = "ElasticNet" #"RandomForestRegressor" #"LogisticRegression"

    #RANDOM_SEED = RANDOM_SEED
    #ALPHA = ALPHA
    #L1_RATIO = L1_RATIO
    
    #Intermediate results
    DVC_ASSETS_PATH = Path("./assets")
    
    #Raw data
    ORIGINAL_DATASET_FILE_PATH = Path("housing.csv")
    
    DATASET_PATH = Path(DVC_ASSETS_PATH / "data")
    FEATURES_PATH = Path(DVC_ASSETS_PATH / "features")
    MODELS_PATH = Path(DVC_ASSETS_PATH / "models")
    METRICS_FILE_PATH = Path(DVC_ASSETS_PATH / "metrics.json")
    PLOTS_FILE_PATH = Path(DVC_ASSETS_PATH / "plots.json")