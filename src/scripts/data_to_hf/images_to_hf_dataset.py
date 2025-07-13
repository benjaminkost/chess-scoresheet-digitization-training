import logging
from abc import abstractmethod
from abc import ABC
from .image_files_to_dataset_strategy import ImageLabelDirToDatasetStrategy

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

class DataVersioning(ABC):
    @abstractmethod
    def upload_dataset(self, path_to_save_dataset: str, path_to_image_dir: str, path_to_label_file: str, owner: str, dataset_name: str):
        """
        Uploads a dataset to a versioning system. This abstract method is expected to be
        implemented by subclasses, providing the functionality to upload a given
        dataset along with associated image directory and label information,
        linking it to a specific owner and dataset name.

        :param path_to_save_dataset: 
        :param path_to_image_dir: 
        :param path_to_label_file: 
        :param owner: The owner or user who will be associated with the dataset.
        :param dataset_name: The name under which the dataset will be stored.
        :return: None
        """
        pass

class HuggingfaceDataVersioning(DataVersioning):

    def __init__(self, dataset_strategy: ImageLabelDirToDatasetStrategy):
        self.dataset_strategy = dataset_strategy

    def set_dataset_strategy(self, dataset_strategy: ImageLabelDirToDatasetStrategy):
        logger.info(f"Set dataset strategy to {dataset_strategy}")

        self.dataset_strategy = dataset_strategy

    def upload_dataset(self, path_to_save_dataset: str, path_to_image_dir: str, path_to_label_file: str, owner: str, dataset_name: str):
        # Get dataset
        dataset = self.dataset_strategy.get_dataset(path_to_image_dir, path_to_label_file)

        # Save dataset to disk
        dataset.save_to_disk(path_to_save_dataset)

        logger.info(f"Dataset successfully saved to disk under: {path_to_save_dataset}.")

        # upload dataset to Huggingface
        dataset.push_to_hub(owner+"/"+dataset_name)

        logger.info(f"Dataset successfully uploaded to Huggingface repo: {owner}/{dataset_name}.")
