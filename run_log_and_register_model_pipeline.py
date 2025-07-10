from app.ml.steps import log_register_custom_model
from sys import version_info
from app.ml.mlflow_models import HFTransformerImageModelWrapper

def log_and_register_model_pipeline():
    # Set model parameter
    MODELS_DIR = "models"

    # Set python version
    PYTHON_VERSION = "{major}.{minor}".format(major=version_info.major, minor=version_info.minor)

    # Define artifacts
    artifacts = {
        "hf_model" : "./mlflow_model_configs/mlflow_model_config.json"
    }
    # conda enviroment
    conda_env = {
        "channels": ["defaults"],
        "dependencies": [
            f"python={PYTHON_VERSION}",
            "pip",
            {
                "pip": [
                    "aiofiles==24.1.0",
                    "click",
                    "fastapi==0.115.12",
                    "mlflow==2.21.2",
                    "mlflow_skinny==2.21.2",
                    "numpy==2.2.5",
                    "opencv_python==4.11.0.86",
                    "pandas==2.2.3",
                    "Pillow==11.2.1",
                    "scikit_learn==1.6.1",
                    "starlette==0.46.2",
                    "transformers==4.51.3",
                    "zenml==0.82.1"
                ]
            }
        ]
    }

    # instantiate the model wrapper
    model = HFTransformerImageModelWrapper()

    # Define registered model name
    registered_model_name = "trocr-base-handwritten-with-pre-and-post-processing"

    # Log Transformer
    log_register_custom_model(
            python_model=model,
            conda_env=conda_env,
            artifacts=artifacts,
            artifact_path=MODELS_DIR,
            registered_model_name=registered_model_name
        )

if __name__ == "__main__":
    log_and_register_model_pipeline()