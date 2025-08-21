import unittest

from src.scripts.scripts_for_steps.data_loader_strategy import HuggingFaceImageDatasetDataLoader
from src.scripts.scripts_for_steps.ingest_data_strategy import HuggingFaceImageDataIngestorStrategy

class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Give
        owner = "BenjaminKost"
        dataset_name = "unprocessed_hcs"
        ingest_data_strategy = HuggingFaceImageDataIngestorStrategy()

        # When
        cls.dataset = ingest_data_strategy.ingest_data(owner=owner, dataset_name=dataset_name)

    def test_load_batch(self):
        # Give
        sut = HuggingFaceImageDatasetDataLoader()

        # When
        batch = sut.load_batch(self.dataset)

        # Then
        self.assertEqual(50,len(batch))

    def test_load_batch_and_load_two_batches_returns_different_batches(self):
        # Give
        sut = HuggingFaceImageDatasetDataLoader()

        # When
        sut.reset_batch_start()
        first_batch = sut.load_batch(self.dataset)
        second_batch = sut.load_batch(self.dataset)

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
        sut = HuggingFaceImageDatasetDataLoader()

        # When
        sut.load_batch(self.dataset)
        self.assertNotEqual(0, sut._batch_start_index)
        sut.reset_batch_start()

        # Then
        self.assertEqual(0, sut._batch_start_index)

if __name__ == '__main__':
    unittest.main()
