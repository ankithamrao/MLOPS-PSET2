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
    #DVC_ASSETS_PATH = Path("./data/assets")
    
    #Raw data
    ORIGINAL_DATASET_FILE_PATH = Path("./data/raw_files/winequality-red.csv")
    DATASET_PATH = Path("./data/processed")
    FEATURES_PATH = Path("./features")
    MODELS_PATH = Path("./models")
    METRICS_FILE_PATH = Path("./reports/metrics.json")
    PLOTS_FILE_PATH = Path("./reports/plots.json")
    PARAMS_PATH = Path("./params.yaml")