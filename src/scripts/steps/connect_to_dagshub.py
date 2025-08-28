import os
import dagshub
from dotenv import load_dotenv

def connect_to_dagshub():
    # Load enviroment variables
    load_dotenv()

    dagshub.init(repo_owner=os.getenv("DAGSHUB_MLFLOW_TRACKING_USERNAME"), repo_name=os.getenv("DAGSHUB_REPOSITORY"), mlflow=True)