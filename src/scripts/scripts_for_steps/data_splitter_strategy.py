import logging
from abc import ABC, abstractmethod
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Configure Logger:
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

# Abstract class for Data Splitting Strategy
# -----------------------------------------------
# This class defines a common interface for different data splitting strategies.
# Subclasses must implement the split_data method.
class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, dataset: Dataset, split: str, feature_column: str, target_column: str):
        """
        Abstract class to split the dataset into training and testing set

        :param feature_column: the column where the image for the corresponding text is inside
        :param split: huggingface by default creates the split "train"
        :param dataset: huggingface dataset with images and the corresponding text as target values
        :param target_column: the column where the text for the corresponding image is inside
        :return: X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        pass

# Concrete Strategy for simple Train-Test Split
# ---------------------------------------------
# This strategy implements a simple train-test split.
class SimpleDataSplittingStrategy(DataSplittingStrategy):
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initializes the SimpleTrainTestSplitStrategy with specific parameters.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: The seed used by the random number generator.
        """
        self.test_size=test_size
        self.random_state=random_state

    def split_data(self, dataset: Dataset, split: str, feature_column: str, target_column: str):
        """
        Implements a simple data_split

        :param feature_column: the column where the image for the corresponding text is inside
        :param split: huggingface by default creates the split "train"
        :param dataset: hugging face dataset with images and the corresponding text as target values
        :param target_column: the column where the text for the corresponding image is inside
        :return: X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """

        # Guard clauses
        if dataset is None:
            raise ValueError("Dataset can not be None")
        if not isinstance(split, str) or not split.strip():
            raise ValueError("Split kann not be an empty string or whitespace")
        if not isinstance(feature_column, str) or not isinstance(target_column, str):
            raise ValueError("Feature and Target columns must be Strings")

        try:
            # Check if the columns exist
            if split not in dataset:
                raise KeyError(f"Split '{split}' not found in Dataset")
            if feature_column not in dataset[split].features:
                raise KeyError(f"Feature-column '{feature_column}' not found in Dataset")
            if target_column not in dataset[split].features:
                raise KeyError(f"Target-column '{target_column}' not found in Dataset")

            # Extract data
            X = dataset[split][feature_column]
            y = dataset[split][target_column]

            # Check if the data is not empty
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Dataset does not contain empty data")

            # Execute Data splitting
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state
            )

            logger.info(f"Train-Test-Split done successfully: "
                        f"Training data: {len(X_train)}, Test data: {len(X_test)}")

            return X_train, X_test, y_train, y_test

        except KeyError as e:
            logger.error(f"Error accessing the Dataset split: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Train-Test-Splits: {str(e)}")
            raise

