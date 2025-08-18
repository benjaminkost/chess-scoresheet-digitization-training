import logging

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

        logger.info(f'Created train and test dataset with train length {len(X_train)} and test length {len(X_test)}')

        train_dict = {
            "image": X_train,
            "label": y_train
        }

        test_dict = {
            "image": X_test,
            "label": y_test
        }

        logger.info(f'Created train and test dict' )

        train_dataset = Dataset.from_dict(train_dict)
        test_dataset = Dataset.from_dict(test_dict)

        logger.info(f'Created train and test dataset')

        return train_dataset, test_dataset
    except Exception as e:
        logger.error(f"Error: {e}")