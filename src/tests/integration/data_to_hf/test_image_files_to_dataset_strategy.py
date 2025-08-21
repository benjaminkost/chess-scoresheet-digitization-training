from pathlib import Path
import os
import unittest
from src.scripts.data_to_hf.image_files_to_dataset_strategy import UnprocessedHcsImageLabelDirToDatasetStrategy, ProcessedHcsImageLabelDirToDatasetStrategy

class UnprocessedHcsImageLabelDirToDatasetStrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        current_path = Path(__file__).resolve()
        cls.path_to_image_dir = current_path.parent.parent.parent.parent / "data" /"raw_data"/"unprocessed_hcs_data"/"images"
        cls.path_to_label_file = current_path.parent.parent.parent.parent /"data"/"raw_data"/"unprocessed_hcs_data"/"training_tags.txt"

    def test_create_dict_with_multiple_images_with_move_boxes_and_labels_containing_all_images_return_len_image_count(self):
        # Give
        strategy = UnprocessedHcsImageLabelDirToDatasetStrategy()

        # When
        list_images = [item for item in os.listdir(self.path_to_image_dir) if ".png" in item]

        res = (strategy.create_dict_with_multiple_images_with_move_boxes_and_labels(self.path_to_image_dir, self.path_to_label_file))

        # Then
        self.assertEqual(len(list_images), len(res))  # add assertion here

class ProcessedHcsImageLabelDirToDatasetStrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        current_path = Path(__file__).resolve()
        cls.path_to_image_dir = current_path.parent.parent.parent.parent / "data" / "raw_data" / "processed_hcs_data" / "images"
        cls.path_to_label_file = current_path.parent.parent.parent.parent / "data" / "raw_data" / "processed_hcs_data" / "training_tags.txt"

    def calculate_label_count(self):
        """
        Calculate the maximum amount of labels from the label file
        """
        label_file = open(self.path_to_label_file, "r")
        label_str = label_file.read()
        label_file.close()
        label_list = [item for item in label_str.split("\n") if len(item) > 0]

        return label_list

    def test_create_dict_for_image_to_label_return_len_image_count(self):
        # Give
        strategy = ProcessedHcsImageLabelDirToDatasetStrategy()

        # When
        actual_number_of_pairs = 13731 # Analysed in uat_image_files_to_dataset_strategy.ipynb
        res = strategy.create_dict_for_image_to_label(self.path_to_image_dir, self.path_to_label_file)

        # Then
        self.assertEqual(actual_number_of_pairs, len(res))  # add assertion here


if __name__ == '__main__':
    unittest.main()

