import numpy as np
from PIL import Image
from zenml import step

# @step
def load_png_image(file_path: str):
    """
    Should load the image file and return the image as PIL.Image.

    :param file_path: directory for a image file that was uploaded by a user
    :return: image as PIL.Image
    """
    if not file_path:
        raise ValueError("No file path provided.")
    if not file_path.endswith(".png"):
        raise ValueError("File path must point to a .png file.")

    # Open image file from file_path
    image = Image.open(file_path)

    # Convert PIL.Image to numpy
    np_img = np.array(image)

    return np_img