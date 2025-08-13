import logging
from typing import Any

from datasets import Dataset

from src.scripts.scripts_for_steps.data_splitter_strategy import DataSplittingStrategy, SimpleDataSplittingStrategy

WHITE = "\033[37m"
RESET = "\033[0m"

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

def split_train_test(dataset, split, feature_column, target_column) -> tuple[Dataset, Dataset] | None:
    """
    Split the dataset into train and test sets.

    """
    data_splitter = SimpleDataSplittingStrategy()

    try:
        X_train, X_test, y_train, y_test = data_splitter.split_data(dataset, split, feature_column, target_column)

        logger.info(f'Created train and test dataset with train shape {X_train.shape} and shape {X_test.shape}')

        train_dict = convert_feature_label_lists_into_dict(X_train, y_train, feature_column, target_column)
        test_dict = convert_feature_label_lists_into_dict(X_test, y_test, feature_column, target_column)

        logger.info(f'Created train and test dict' )

        train_dataset = Dataset.from_dict(train_dict)
        test_dataset = Dataset.from_dict(test_dict)

        logger.info(f'Created train and test dataset')

        return train_dataset, test_dataset
    except Exception as e:
        logger.error(f"Dataset was not found, cannot split into train and test sets. Error: {e}")


def convert_feature_label_lists_into_dict(feature_list: list, label_list: list, feature_name: str, label_name: str):
    """
    Convert the feature list and the corresponding label list into a dictionary.

    :param feature_list: list of pixel values per image
    :param label_list: list of tokens for each label

    """
    res_dataset = {}

    for feature, label in zip(feature_list, label_list):
        res_dataset[feature_name] = feature
        res_dataset[label_name] = label

    return res_dataset