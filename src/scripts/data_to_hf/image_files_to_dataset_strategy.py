import logging
from abc import ABC, abstractmethod
from pathlib import Path

from datasets import Dataset, Image
import os
import pandas as pd

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

class ImageLabelDirToDatasetStrategy(ABC):
    @abstractmethod
    def get_dataset(self, path_to_image_dir: Path, path_to_label_file: Path) -> Dataset:
        """Abstract method to get a dataset from image and label files"""
        pass

class UnprocessedHcsImageLabelDirToDatasetStrategy(ImageLabelDirToDatasetStrategy):
    def create_dict_for_move_boxes_and_labels(self, image_name: str, list_labels: list) -> dict:
        """
        Method to create dictionary object of ground truth values for specific image
        prerequisite: the moves in the ground truth values have to be in the correct move order

        :param list_labels: list of labels
        :param image_name: name of the image file
        :return: dictionary object of the name (string) of the move box and the responding ground truth value
        """
        res = {}

        for label in list_labels:
            if image_name in label:
                # name of move box and the ground truth label are saved in a dict object
                res[label.split(" ")[0]] = label.split(" ")[1]

        return res

    def create_dict_with_multiple_images_with_move_boxes_and_labels(
            self, path_to_image_dir: Path, path_to_label_file: Path) -> dict:
        """
        Creates a dictionary for all image files with a dictionary inside for the move boxes with
        the corresponding label

        :param path_to_image_dir: path to image files
        :param path_to_label_file: path to the label/ ground truth file
        :return: dictionary containing all image files with a dictionary inside for the move boxes with
        the corresponding label
        """

        res = {}

        list_images = os.listdir(path_to_image_dir)

        file_ground_truth = open(path_to_label_file, "r")
        str_ground_truth = file_ground_truth.read()
        file_ground_truth.close()

        list_labels = str_ground_truth.split("\n")

        logger.info(f"Creating a dictionary inside a dictionary for main images and the sub image with the corresponding label."
                     f"With the path to the images: "
                     f"{path_to_image_dir} and the ground truth file: {path_to_label_file}")

        for image_compl_name in list_images:
            if ".png" in image_compl_name:
                image_name = image_compl_name.split(".")[0]
                temp_dict = self.create_dict_for_move_boxes_and_labels(image_name, list_labels)
                res[image_compl_name] = temp_dict

        return res

    def create_dataset_from_dict_with_move_boxes(self, path_to_image_dir: Path, img_label_dict) -> Dataset:
        """
        Creates a huggingface dataset object of an image list and for each image the list of labels
        for the move boxes

        :param path_to_image_dir: path to image files
        :param img_label_dict: dictionary object containing all image files with a dictionary inside for the move boxes with
        :return: a huggingface dataset object of an image list and for each image the list of labels
        for the move boxes
        """

        # List for Dataset
        data = []

        logger.info(f"Creating a dataset from a dictionary with the image and the move boxes with the corresponding label.")

        try:
            # Transform data
            for main_img, move_boxes in img_label_dict.items():
                temp_labels = []
                image_path = os.path.join(path_to_image_dir, main_img)
                if os.path.exists(image_path):
                    for move_box, label in move_boxes.items():
                        temp_labels.append(label)
                    data.append({"image": image_path, "labels": temp_labels})

                dataset = Dataset.from_list(data).cast_column("image", Image())
        except Exception as e:
            print(e)

        return dataset

    def dict_to_list(self, d):
        return list(d.items())

    def get_dataset(self, path_to_image_dir: Path, path_to_label_file: Path) -> Dataset:
        """
        Creates a huggingface dataset object of an image list and for each image the list of labels
        for the move boxes with path of the images and the path to the label file

        :param path_to_image_dir: path to image files
        :param path_to_label_file: path to the label/ ground truth file
        :return: Creates a huggingface dataset object of an image list and for each image the list of labels
        for the move boxes
        """

        ## Load Labels regarding there image names
        data_dict = self.create_dict_with_multiple_images_with_move_boxes_and_labels(path_to_image_dir, path_to_label_file)

        ## Create dataset object from dict
        dataset = self.create_dataset_from_dict_with_move_boxes(path_to_image_dir, data_dict)

        logger.info(f"Dataset created successfully.")

        return dataset

class ProcessedHcsImageLabelDirToDatasetStrategy(ImageLabelDirToDatasetStrategy):
    def create_dict_for_image_to_label(self, path_to_image_dir: Path, path_to_label_file: Path):
        """
        Creates a dictionary object consisting of the image name of a move box and the label

        :param path_to_image_dir: path to image files
        :param path_to_label_file: path to the label/ ground truth file
        :return: a dictionary object consisting of the image name of a move box and the label
        """

        res = {}

        # all image names
        list_images = os.listdir(path_to_image_dir)
        # Sort them alphabetically
        list_images.sort()

        # ground truth values as Pandas DataFrame
        list_ground_truth = pd.read_csv(path_to_label_file)
        # Sort them alphabetically
        list_ground_truth.sort_values(by=["image"])

        logger.info(f"Creating a dictionary inside a dictionary for main images and the sub image with the corresponding label."
                     f"With the path to the images: "
                     f"{path_to_image_dir} and the ground truth file: {path_to_label_file}")

        # Create dict object with image_name corresponding to label
        for image_name in list_images:
            for ground_truth_value in list_ground_truth:
                if ground_truth_value.count(image_name) > 1:
                    ValueError(f"Error: For {image_name} are multiple labels in the ground truth file!")
                elif image_name in ground_truth_value:
                    res[image_name] = ground_truth_value["label"]
                    break

        return res

    def create_dataset_from_dict_with_img_to_label(self, path_to_image_dir: Path, img_label_dict):
        # List for Dataset
        data = []

        logger.info(f"Creating a dataset from a dictionary with the main image and the sub image with the corresponding label.")

        # Transform data
        for img, label in img_label_dict.items():
            image_path = os.path.join(path_to_image_dir, img)
            if os.path.exists(image_path):
                data.append({"image": image_path, "label": label})

            dataset = Dataset.from_list(data).cast_column("image", Image())

        return dataset

    def dict_to_list(self, d):
        return list(d.items())

    def get_dataset(self, path_to_image_dir: Path, path_to_label_file: Path) -> Dataset:
        ## Make dict object with image and corresponding label
        processed_hcs_image_label_dict = self.create_dict_for_image_to_label(path_to_image_dir, path_to_label_file)

        ## Create dataset object from dict
        dataset_processed_hcs = self.create_dataset_from_dict_with_img_to_label(path_to_image_dir, processed_hcs_image_label_dict)

        logger.info(f"Dataset created successfully.")

        return dataset_processed_hcs

class HSSSDImageLabelDirToDatasetStrategy(ImageLabelDirToDatasetStrategy):
    def create_list_for_move_boxes_and_labels(self, image_name: str, df_labels: pd.DataFrame) -> list:
        """
        Method to create list object of ground truth values for specific image
        prerequisite: the moves in the ground truth values have to be in the correct move order

        :param df_labels: dataframe of labels
        :param image_name: name of the image file
        :return: list object of the name (string) of the move box and the responding ground truth value
        """
        res = []

        for index, label in df_labels.iterrows():
            if image_name in label["image"]:
                # name of move box and the ground truth label are saved in a dict object
                res.append(label["label"])

        return res

    def create_dict_with_multiple_images_with_move_boxes_and_labels(
            self, path_to_image_dir: Path, path_to_label_file: Path) -> dict:
        """
        Creates a dictionary for all image files with a dictionary inside for the move boxes with
        the corresponding label

        :param path_to_image_dir: path to image files
        :param path_to_label_file: path to the label/ ground truth file
        :return: dictionary containing all image files with a dictionary inside for the move boxes with
        the corresponding label
        """

        res = {}

        list_images = os.listdir(path_to_image_dir)

        df_labels = pd.read_csv(path_to_label_file)

        logger.info(f"Creating a dictionary inside a dictionary for main images and the sub image with the corresponding label."
                     f"With the path to the images: "
                     f"{path_to_image_dir} and the ground truth file: {path_to_label_file}")

        for image_name in list_images:
            if ".png" in image_name:
                image_name = image_name.split(".")[0]
                temp_dict = self.create_list_for_move_boxes_and_labels(image_name, df_labels)
                res[image_name] = temp_dict
                logger.info(f"Current size of dictionary object: {len(res)}")

        return res

    def create_dataset_from_dict_with_move_boxes(self, path_to_image_dir: Path, img_label_dict) -> Dataset:
        """
        Creates a huggingface dataset object of an image list and for each image the list of labels
        for the move boxes

        :param path_to_image_dir: path to image files
        :param img_label_dict: dictionary object containing all image files with a dictionary inside for the move boxes with
        :return: a huggingface dataset object of an image list and for each image the list of labels
        for the move boxes
        """

        # List for Dataset
        data = []

        logger.info(f"Creating a dataset from a dictionary with the image and the move boxes with the corresponding label.")

        try:
            # Transform data
            for main_img_name, list_of_labels in img_label_dict.items():
                image_path = os.path.join(path_to_image_dir, f"{main_img_name}.png")
                if os.path.exists(image_path):
                    data.append({"image": image_path, "labels": list_of_labels})
                else:
                    logger.error(f"{image_path} does not exist so it is not")

                dataset = Dataset.from_list(data).cast_column("image", Image())
        except Exception as e:
            print(e)

        return dataset

    def dict_to_list(self, d):
        return list(d.items())

    def get_dataset(self, path_to_image_dir: Path, path_to_label_file: Path) -> Dataset:
        """
        Creates a huggingface dataset object of an image list and for each image the list of labels
        for the move boxes with path of the images and the path to the label file

        :param path_to_image_dir: path to image files
        :param path_to_label_file: path to the label/ ground truth file
        :return: Creates a huggingface dataset object of an image list and for each image the list of labels
        for the move boxes
        """

        ## Load Labels regarding their image names
        logger.info("Starting to create dictionary object out of the images and labels...")

        data_dict = self.create_dict_with_multiple_images_with_move_boxes_and_labels(path_to_image_dir, path_to_label_file)

        logger.info(f"Dictionary object successfully created with length {len(data_dict)}")

        ## Create dataset object from dict
        logger.info("Starting to create dataset object...")

        dataset = self.create_dataset_from_dict_with_move_boxes(path_to_image_dir, data_dict)

        logger.info(f"Dataset created successfully.")

        return dataset