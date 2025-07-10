import logging
import os

import pandas as pd
from sklearn.model_selection import ParameterGrid

from app.ml.classes_for_steps import HuggingFaceImageDataIngestorStrategy
from app.ml.classes_for_steps import HuggingFacePreprocessingStrategy, ThresholdMethod
# Configure Logger:
# ANSI Escape Code for white letters
WHITE = "\033[37m"
RESET = "\033[0m"  # reset of color

# Logger configure
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console-Handler
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Formatter with ANSI Escape Code for white letters
formatter = logging.Formatter(f'{WHITE}%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s{RESET}')
handler.setFormatter(formatter)

# Handler for Logger added
logger.addHandler(handler)

# Define Hyperparameter-Grid
param_grid = {
    "kernelsize_gaussianBlur": [(3, 3), (5, 5), (7, 7)],
    "sigmaX": [0, 1.0, 2.0],
    "threshold_method": [ThresholdMethod.OTSU, ThresholdMethod.ADAPTIVE_THRESH_MEAN_C, ThresholdMethod.ADAPTIVE_THRESH_GAUSSIAN_C],
    "block_size": [9, 11, 13, 15, 17, 19, 21, 23],
    "c_value": [1, 2, 3],
    "horizontal_kernel_divisor": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "vertical_kernel_divisor": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "erosion_iterations": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "dilation_iterations": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

# Generating all possible combinations of the grid
param_combinations = ParameterGrid(param_grid)

best_params = None
max_length = -1

# Loading Dataset
owner = "BenjaminKost"
dataset_name = "unprocessed_hcs"
ingestor = HuggingFaceImageDataIngestorStrategy()
dataset = ingestor.ingest_data(owner=owner, dataset_name=dataset_name)

# Initialize tuning history list
# tuning_history = []

# Verzeichnis für Tuning-Historie sicherstellen
os.makedirs("tuning_history", exist_ok=True)

# Datei in der die Ergebnisse gespeichert werden
current_date = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
file_path = f"./tuning_history/tuning_history_{current_date}.csv"

# Schreibe die Header nur einmal am Anfang
with open(file_path, 'w') as f:
    pd.DataFrame(columns=[
        "kernelsize_gaussianBlur", "sigmaX", "threshold_method", "block_size",
        "c_value", "horizontal_kernel_divisor", "vertical_kernel_divisor",
        "erosion_iterations", "dilation_iterations", "result_length", "error_message"
    ]).to_csv(f, index=False)  # Initiale Header-Zeile schreiben


# Optimierung starten
for params in param_combinations:
    try:
        # Creating the strategy object with the current parameters
        strategy = HuggingFacePreprocessingStrategy(
            kernelsize_gaussianBlur=params["kernelsize_gaussianBlur"],
            sigmaX=params["sigmaX"],
            threshold_method=params["threshold_method"],
            block_size=params["block_size"],
            c_value=params["c_value"],
            horizontal_kernel_divisor=params["horizontal_kernel_divisor"],
            vertical_kernel_divisor=params["vertical_kernel_divisor"],
            erosion_iterations=params["erosion_iterations"],
            dilation_iterations=params["dilation_iterations"]
        )

        # Call fit method of the strategy
        strategy.fit(dataset)

        # Call transform method of the strategy to get the transformed dataset
        transformed_dataset = strategy.preprocess_dataset(dataset)
        current_length = len(transformed_dataset)

        # save results of tuning
        row = {
            "kernelsize_gaussianBlur": params["kernelsize_gaussianBlur"],
            "sigmaX": params["sigmaX"],
            "threshold_method": params["threshold_method"].name,
            "block_size": params["block_size"],
            "c_value": params["c_value"],
            "horizontal_kernel_divisor": params["horizontal_kernel_divisor"],
            "vertical_kernel_divisor": params["vertical_kernel_divisor"],
            "erosion_iterations": params["erosion_iterations"],
            "dilation_iterations": params["dilation_iterations"],
            "result_length": current_length,
            "error_message": ""
        }

        with open(file_path, 'a') as f:
            pd.DataFrame([row]).to_csv(f, index=False, header=False)


    # Save the best parameters and the corresponding length
        if current_length > max_length:
            max_length = current_length
            best_params = params



    except Exception as e:
        logger.error(f"Fehler bei Parametern {params}: {e}")
        row = {
        "kernelsize_gaussianBlur": params["kernelsize_gaussianBlur"],
        "sigmaX": params["sigmaX"],
        "threshold_method": params["threshold_method"].name,
        "block_size": params["block_size"],
        "c_value": params["c_value"],
        "horizontal_kernel_divisor": params["horizontal_kernel_divisor"],
        "vertical_kernel_divisor": params["vertical_kernel_divisor"],
        "erosion_iterations": params["erosion_iterations"],
        "dilation_iterations": params["dilation_iterations"],
        "result_length": "error",
        "error_message": str(e)
        }

        with open(file_path, 'a') as f:
            pd.DataFrame([row]).to_csv(f, index=False, header=False)

logger.info(f"Best Parameters: {best_params} with fully processed images count of: {max_length}")
