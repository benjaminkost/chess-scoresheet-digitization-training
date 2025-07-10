from steps import log_register_custom_model
from sys import version_info
from mlflow_models.trocr_mlflow_model import HFTransformerImageModelWrapper

def log_and_register_model_pipeline():
    # Set model parameter
    MODELS_DIR = "../../models"

    # Set python version
    PYTHON_VERSION = "{major}.{minor}".format(major=version_info.major, minor=version_info.minor)

    # Define artifacts
    artifacts = {
        "hf_model" : "./src/scripts/mlflow_models/mlflow_model_config.json"
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