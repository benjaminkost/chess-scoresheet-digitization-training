import os
import unittest
from .image_files_to_dataset_strategy import UnprocessedHcsImageLabelDirToDatasetStrategy, ProcessedHcsImageLabelDirToDatasetStrategy


class UnprocessedHcsImageLabelDirToDatasetStrategyTests(unittest.TestCase):
    def test_create_dict_with_multiple_images_with_move_boxes_and_labels_containing_all_images_return_len_image_count(self):
        # Give
        path_to_image_dir = "../../chess-scoresheet-digitization-training/data/raw_data/unprocessed_hcs_data/images"
        path_to_label_file = "../../chess-scoresheet-digitization-training/data/raw_data/unprocessed_hcs_data/training_tags.txt"

        strategy = UnprocessedHcsImageLabelDirToDatasetStrategy()

        # When
        list_images = [item for item in os.listdir(path_to_image_dir) if ".png" in item]

        res = (strategy.create_dict_with_multiple_images_with_move_boxes_and_labels(path_to_image_dir, path_to_label_file))

        # Then
        self.assertEqual(len(list_images), len(res))  # add assertion here

class ProcessedHcsImageLabelDirToDatasetStrategyTests(unittest.TestCase):
    def calculate_label_count(self):
        """
        Calculate the maximum amount of labels from the label file
        """
        path_to_label_file = "../../chess-scoresheet-digitization-training/data/raw_data/processed_hcs_data/training_tags.txt"

        label_file = open(path_to_label_file, "r")
        label_str = label_file.read()
        label_file.close()
        label_list = [item for item in label_str.split("\n") if len(item) > 0]

        return label_list

    def test_create_dict_for_image_to_label_return_len_image_count(self):
        # Give
        path_to_image_dir = "../../chess-scoresheet-digitization-training/data/raw_data/processed_hcs_data/images"
        path_to_label_file = "../../chess-scoresheet-digitization-training/data/raw_data/processed_hcs_data/training_tags.txt"

        strategy = ProcessedHcsImageLabelDirToDatasetStrategy()

        # When
        actual_number_of_pairs = 13731 # Analysed in uat_image_files_to_dataset_strategy.ipynb
        res = strategy.create_dict_for_image_to_label(path_to_image_dir, path_to_label_file)

        # Then
        self.assertEqual(actual_number_of_pairs, len(res))  # add assertion here


if __name__ == '__main__':
    unittest.main()

