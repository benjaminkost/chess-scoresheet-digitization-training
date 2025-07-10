import random
from abc import ABC, abstractmethod
from datasets import Dataset
import logging

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

class DataLoaderStrategy(ABC):
    @abstractmethod
    def load_batch(self, dataset: Dataset, batch_size=None, batch_start_index=None, shuffle=False):
        pass
    def reset_batch_start(self):
        pass

class HuggingFaceImageDatasetDataLoader(DataLoaderStrategy):
    def __init__(self, batch_size=50, data_set="train", _batch_start_index=0):
        """

        :param batch_size: size of batch
        :param data_set: name of the dataset to load, e.g. "train" or "test"
        :param _batch_start_index: start index of batch
        """
        self._batch_size=batch_size
        self._data_set=data_set
        self._batch_start_index=_batch_start_index

    def load_batch(self, dataset: Dataset, batch_size=None, batch_start_index=None, shuffle=False):
        """

        :param batch_size:
        :param batch_start_index:
        :param dataset: huggingface dataset
        :return: a batch of the list of dicts with the images and the corresponding labels
        """
        # Initialize if batchsize is not None
        if batch_size is not None:
            self._batch_size = batch_size

        if batch_start_index is not None:
            self._batch_start_index = batch_start_index

        res_list = []

        # Check if dataset partition exists
        if self._data_set not in dataset:
            raise ValueError(f"Dataset partition '{self._data_set}' not found in dataset")

        # Set start and end index of current batch
        batch_start_index = self._batch_start_index
        batch_end_index = self._batch_start_index + self._batch_size

        # Check that start and end index are in dataset
        if batch_start_index >= len(dataset[self._data_set]):
            self.reset_batch_start()
            batch_start_index = self._batch_start_index
            batch_end_index = self._batch_start_index + self._batch_size
        elif batch_end_index > len(dataset[self._data_set]):
            self._batch_start_index = len(dataset[self._data_set]) - 1

        # define dataset from split
        dataset_split = dataset[self._data_set]

        # Set Feature and Label column
        logger.debug(f"Structure of dataset: {dataset}, and Structure of Dataset {self._data_set} Dataset: "
                     f"\ntype: {type(dataset_split)}\nDataset Content: {dataset_split}")
        feature_column = list(dataset_split.features.keys())[0]
        logger.debug(f"Feature column selected: {feature_column}")
        label_column = list(dataset_split.features.keys())[1]
        logger.debug(f"Label column selected: {label_column}")

        # Create batch
        for i in range(batch_start_index, batch_end_index):
            dict_element = {
                feature_column: dataset_split[i][feature_column],
                label_column: dataset_split[i][label_column]
            }
            res_list.append(dict_element)

        # Make end of this batch to start of the next batch
        self._batch_start_index = batch_end_index

        # Shuffle list
        if shuffle:
            random.shuffle(res_list)

        return res_list

    def reset_batch_start(self):
        self._batch_start_index = 0

