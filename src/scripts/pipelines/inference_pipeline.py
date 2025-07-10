import logging

import numpy as np
from PIL import Image

from app.ml.steps.predict_postprocessing_step import postprocessing_prediction
from app.ml.steps.predict_preprocessing_step import preprocessing_image
from app.ml.steps.predictor import predictor
from app.ml.steps.dynamic_importer import load_png_image
from app.ml.steps.model_loader import load_model

# configure logger
logging.basicConfig(
    level=logging.INFO,  # Log-Ebene (z. B. DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log-Format
)
logger = logging.getLogger(__name__)  # Logger mit Modulnamen beziehen

# @pipeline(enable_cache=False)
def inference_pipeline(image_file_path: str, model_name: str):
    """
    Give a prediction for a given image

    :return: prediction
    """

    # Load image that was uploaded
    logger.info(f"Loading image from {image_file_path}")
    np_img = load_png_image(image_file_path)

    # Convert np_array to PIL
    logger.info(f"Converting np_array to PIL")
    input_image = Image.fromarray(np_img)

    # Preprocess image
    logger.info(f"Preprocessing image")
    list_of_move_boxes = preprocessing_image(input_image)

    # Load the registered model
    logger.info(f"Loading model: {model_name}")
    model = load_model(model_name)

    # Run the prediction
    list_of_predictions = []
    logger.info(f"Running prediction")
    for move_box in list_of_move_boxes:
        # convert images to RGB
        move_box = move_box.convert("RGB")

        # convert PIL image to numpy array
        np_move_box_img = np.array(move_box)

        # predict text on image
        prediction_dict = predictor(model, np_move_box_img)
        prediction = prediction_dict["prediction"]

        # Add it to the list of predictions for the move boxes
        list_of_predictions.append(prediction)

    # Post-process prediction list
    logger.info(f"Post-processing prediction")
    pgn_str = postprocessing_prediction(list_of_predictions)

    return pgn_str
