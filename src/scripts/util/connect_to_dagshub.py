import os
import dagshub
from dotenv import load_dotenv

# Load enviroment variables
load_dotenv()

dagshub.init(repo_owner=os.environ["DAGSHUB_MLFLOW_TRACKING_USERNAME"], repo_name=os.environ["DAGSHUB_REPOSITORY"], mlflow=True)