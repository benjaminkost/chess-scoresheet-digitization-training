import unittest
from  src.scripts.scripts_for_steps.ingest_data_strategy import HuggingFaceImageDataIngestorStrategy
from PIL import Image


class ImageDataIngestorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Give
        owner = "BenjaminKost"
        dataset_name = "unprocessed_hcs"
        ingestor = HuggingFaceImageDataIngestorStrategy()

        # When
        cls.dataset = ingestor.ingest_data(owner=owner, dataset_name=dataset_name)

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


if __name__ == '__main__':
    unittest.main()
