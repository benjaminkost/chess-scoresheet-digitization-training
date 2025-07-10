import unittest

from PIL import Image
from datasets import Dataset

from app.ml.classes_for_steps.data_module import HuggingFaceImageDataModule
from app.ml.classes_for_steps.ingest_data_strategy import HuggingFaceImageDataIngestorStrategy
from app.ml.classes_for_steps.preprocessing_strategy import HuggingFacePreprocessingStrategy
from app.ml.classes_for_steps.data_splitter_strategy import SimpleDataSplittingStrategy
from app.ml.classes_for_steps.data_loader_strategy import HuggingFaceImageDatasetDataLoader

class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Give
        owner = "BenjaminKost"
        dataset_name = "unprocessed_hcs"
        ingest_data_strategy = HuggingFaceImageDataIngestorStrategy()
        preprocessing_strategy = HuggingFacePreprocessingStrategy()
        data_splitter_strategy = SimpleDataSplittingStrategy()
        data_loader_strategy = HuggingFaceImageDatasetDataLoader()
        cls.data_module = HuggingFaceImageDataModule(ingest_data_strategy=ingest_data_strategy,
                                                     preprocessing_strategy=preprocessing_strategy,
                                                     data_splitter_strategy=data_splitter_strategy,
                                                     dataloader_strategy=data_loader_strategy)

        # When
        cls.dataset = cls.data_module.ingest_data(owner=owner, dataset_name=dataset_name)

    # Test Data ingest strategy
    def test_ingest_image_dataset_from_huggingface_BenjaminKost_unprocessed_hcs_Dataset_csv_ReturnsDS(self):
            # Then
            self.assertEqual(206, len(self.dataset["train"]))  # add assertion here

    def test_ingest_image_dataset_from_huggingface_BenjaminKost_unprocessed_hcs_Dataset_csv_and_check_datatypes_Returns_image(self):

        # Then
        self.assertIsInstance(self.dataset["train"][0]["image"], Image.Image)

    def test_ingest_image_dataset_from_huggingface_BenjaminKost_unprocessed_hcs_Dataset_csv_and_check_datatypes_Returns_str(self):

        # Then
        self.assertIsInstance(self.dataset["train"][0]["labels"], list)

    def test_ingest_image_dataset_from_huggingface_BenjaminKost_unprocessed_hcs_Dataset_csv_and_check_datatypes_of_column_image_Returns_list(self):

        # Then
        self.assertIsInstance(self.dataset["train"]["image"], list)

    def test_ingest_image_dataset_from_huggingface_BenjaminKost_unprocessed_hcs_Dataset_csv_and_check_datatypes_of_column_text_Returns_list(self):

        # Then
        self.assertIsInstance(self.dataset["train"]["labels"], list)

    # Test Preprocessing strategy
    def test_transform_returns_dataset_with_cut_out_move_boxes(self):
        # When
        res_dataset = self.data_module.preprocess_dataset(self.dataset)

        # Then
        self.assertIsInstance(res_dataset, Dataset)
        is_bigger_list = len(res_dataset) > len(self.dataset)
        self.assertTrue(is_bigger_list)

    # Test Data Splitter Strategy
    def test_SimpleDataSplittingStrategy_data_split_give_unprocessed_hcs_dataset_showed_return_train_test_sets(self):
        # When
        X_train, X_test, y_train, y_test  = self.data_module.split_data(self.dataset, "train", "image", "labels")

        # Then
        self.assertIsInstance(X_train, list)
        self.assertIsInstance(X_train[0], Image.Image)
        self.assertIsInstance(y_train, list)
        self.assertIsInstance(y_train[0], list)
        self.assertEqual(len(X_train), 164)
        self.assertEqual(len(y_test), 42)

    def test_SimpleDataSplittingStrategy_data_split_give_wrong_split_name_return_error(self):
        # Then
        self.assertRaises(KeyError, self.data_module.split_data, self.dataset, "wrong_split_name", "image", "labels")

    def test_SimpleDataSplittingStrategy_data_split_give_wrong_feature_column_name_return_error(self):

        # Then
        self.assertRaises(KeyError, self.data_module.split_data, self.dataset, "train", "wrong_feature_column_name", "labels")

    def test_SimpleDataSplittingStrategy_data_split_give_wrong_labels_column_name_return_error(self):

        # Then
        self.assertRaises(KeyError, self.data_module.split_data, self.dataset, "train", "image", "wrong_target_column_name")

    def test_SimpleDataSplittingStrategy_data_split_with_attributes_return_error(self):
        # When
        self.data_module.set_data_splitter_strategy(SimpleDataSplittingStrategy(0.5, 10))

        X_train, X_test, y_train, y_test  = self.data_module.split_data(self.dataset, "train", "image", "labels")

        # Then
        self.assertIsInstance(X_train, list)
        self.assertIsInstance(X_train[0], Image.Image)
        self.assertIsInstance(y_train, list)
        self.assertIsInstance(y_train[0], list)
        self.assertEqual(len(X_train), 103)
        self.assertEqual(len(y_test), 103)

    # Test Data Loader Strategy
    def test_load_batch(self):
        # When
        batch = self.data_module.load_batch(self.dataset)

        # Then
        self.assertEqual(50,len(batch))

    def test_load_batch_and_load_two_batches_returns_different_batches(self):

        # When
        self.data_module.reset_batch_start()
        first_batch =  self.data_module.load_batch(self.dataset)
        second_batch = self.data_module.load_batch(self.dataset)

        # Then
        self.assertEqual(50,len(first_batch))
        self.assertEqual(50,len(second_batch))
        for i in range(50):
            self.assertEqual(self.dataset["train"][i]["labels"], first_batch[i]["labels"])
        self.assertNotEqual(first_batch,second_batch)
        for i in range(50, 99):
            self.assertEqual(self.dataset["train"][i]["labels"], second_batch[i-50]["labels"])

    def test_reset_batch_start_returns_start_index_0(self):
        # Give
        first_element = self.dataset["train"][0]["labels"][0]

        # When
        self.data_module.load_batch(self.dataset)
        second_batch = self.data_module.load_batch(self.dataset)
        self.assertNotEqual(first_element, second_batch[0]["labels"][0])
        self.data_module.reset_batch_start()

        first_batch_second_call = self.data_module.load_batch(self.dataset)

        # Then
        self.assertEqual(first_element, first_batch_second_call[0]["labels"][0])


if __name__ == '__main__':
    unittest.main()
