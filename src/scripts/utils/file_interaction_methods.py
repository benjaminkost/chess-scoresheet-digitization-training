import logging
import os
from pathlib import Path

import PIL
from pdf2image import convert_from_path

from src.scripts.utils.config import get_root_dir_path

# Logger set-up
# ANSI Escape Code for white letters
WHITE = "\033[37m"
RESET = "\033[0m"  # Zum Zurücksetzen der Farbe

# Logger configure
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console-Handler
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Formatter with ANSI Escape Code for white letters
formatter = logging.Formatter(f'{WHITE}%(asctime)s - %(name)s - %(levelname)s - %(message)s{RESET}')
handler.setFormatter(formatter)

# Handler for Logger added
logger.addHandler(handler)

def convert_folder_of_pdf_files_to_png(relative_path_to_folder: Path):
    """
    Convert PDF files

    :param relative_path_to_folder:
    :return:
    """
    list_of_files_in_folder = os.listdir(relative_path_to_folder)

    PIL.Image.MAX_IMAGE_PIXELS = 933120000

    for filename in list_of_files_in_folder:
        if filename.split(".")[-1] == "pdf":
            path_of_file = relative_path_to_folder / filename
            logger.info(f"Processing file: {path_of_file}")
            convert_pdf_to_png(path_of_file)

def convert_pdf_to_png(file_path: Path) -> None:
    """
    Converts single PDF file into PNG files

    :param file_path: Absolute path to PDF file
    :return: Saves PNG files in same folder as PDF and deletes PDF file
    """
    try:
        pages = convert_from_path(str(file_path), 500)

        logger.info(f"PDF file '{file_path}' was converted to PNG files")

        for page in pages:
            new_file_path = f"{file_path.parent}/{file_path.name.split(".")[0]}.png"
            page.save(new_file_path, 'PNG')
            logger.info(f"PNG file '{new_file_path}' was saved")

        try:
            os.remove(file_path)
            logger.info(f"PDF file '{file_path.name}' was deleted")

        except Exception as ex:
            logger.error(f"PDF file '{file_path.name}' could not be deleted, because of: {ex}")
    except Exception as ex:
        logger.error(f"PDF could not be converted to PNG files, because: {ex}")

if __name__ == "__main__":
    relative_path_to_folder = get_root_dir_path() / "data" / "raw_data" / "hsssd_data" / "images"

    convert_folder_of_pdf_files_to_png(relative_path_to_folder)
