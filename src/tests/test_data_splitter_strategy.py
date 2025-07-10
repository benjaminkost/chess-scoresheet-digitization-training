import unittest
from PIL import Image
from app.ml.classes_for_steps.data_splitter_strategy import SimpleDataSplittingStrategy
from app.ml.classes_for_steps.ingest_data_strategy import HuggingFaceImageDataIngestorStrategy

class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Give
        owner = "BenjaminKost"
        datasetName = "unprocessed_hcs"
        ingestor = HuggingFaceImageDataIngestorStrategy()

        cls.dataset = ingestor.ingest_data(owner=owner, dataset_name=datasetName)

    def test_SimpleDataSplittingStrategy_data_split_give_unprocessed_hcs_dataset_showed_return_train_test_sets(self):
        # When
        splitter = SimpleDataSplittingStrategy()
        X_train, X_test, y_train, y_test  = splitter.split_data(self.dataset, "train", "image", "labels")

        # Then
        self.assertIsInstance(X_train, list)
        self.assertIsInstance(X_train[0], Image.Image)
        self.assertIsInstance(y_train, list)
        self.assertIsInstance(y_train[0], list)
        self.assertEqual(len(X_train), 164)
        self.assertEqual(len(y_test), 42)

    def test_SimpleDataSplittingStrategy_data_split_give_wrong_split_name_return_error(self):
        # When
        splitter = SimpleDataSplittingStrategy()

        # Then
        self.assertRaises(KeyError, splitter.split_data, self.dataset, "wrong_split_name", "image", "labels")

    def test_SimpleDataSplittingStrategy_data_split_give_wrong_feature_column_name_return_error(self):
        # When
        splitter = SimpleDataSplittingStrategy()

        # Then
        self.assertRaises(KeyError, splitter.split_data, self.dataset, "train", "wrong_feature_column_name", "labels")

    def test_SimpleDataSplittingStrategy_data_split_give_wrong_labels_column_name_return_error(self):
        # When
        splitter = SimpleDataSplittingStrategy()

        # Then
        self.assertRaises(KeyError, splitter.split_data, self.dataset, "train", "image", "wrong_target_column_name")

    def test_SimpleDataSplittingStrategy_data_split_with_attributes_return_error(self):
        # When
        splitter = SimpleDataSplittingStrategy(0.5, 10)

        X_train, X_test, y_train, y_test  = splitter.split_data(self.dataset, "train", "image", "labels")

        # Then
        self.assertIsInstance(X_train, list)
        self.assertIsInstance(X_train[0], Image.Image)
        self.assertIsInstance(y_train, list)
        self.assertIsInstance(y_train[0], list)
        self.assertEqual(len(X_train), 103)
        self.assertEqual(len(y_test), 103)


if __name__ == '__main__':
    unittest.main()
