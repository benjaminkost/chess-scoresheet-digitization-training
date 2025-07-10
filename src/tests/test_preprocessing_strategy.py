import unittest

import numpy as np
from datasets import Dataset

from app.ml.classes_for_steps.ingest_data_strategy import HuggingFaceImageDataIngestorStrategy
from app.ml.classes_for_steps.preprocessing_strategy import HuggingFacePreprocessingStrategy


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Give
        owner = "BenjaminKost"
        dataset_name = "unprocessed_hcs"
        ingestor = HuggingFaceImageDataIngestorStrategy()
        cls.sut_preprocessing = HuggingFacePreprocessingStrategy()

        # When
        cls.dataset = ingestor.ingest_data(owner=owner, dataset_name=dataset_name)

    def test_convert_dataset_to_list_Return_list_of_dataset_returns_list(self):
        # When
        split_of_dataset = "train"
        list_of_dataset = (self.sut_preprocessing
                           .convert_dataset_to_list(
            self.dataset, split_of_dataset, "image", "labels"))

        # Then
        self.assertIsInstance(list_of_dataset, list)
        self.assertEqual(206, len(list_of_dataset))

    def test_process_image_dataset_rgb_to_grayscale_returns_list(self):
        # Give
        split_of_dataset = "train"
        list_of_dataset = (self.sut_preprocessing
        .convert_dataset_to_list(
            self.dataset, split_of_dataset, "image", "labels"))
        # When
        grayscaled_list = self.sut_preprocessing.process_image_dataset_rgb_to_grayscale(list_of_dataset)

        # Then
        self.assertIsInstance(grayscaled_list, list)
        for i in range(0, len(grayscaled_list)):
            sample_image = np.array(grayscaled_list[i]["image"])
            count_of_dimensions = len(sample_image.shape)
            self.assertEqual(2, count_of_dimensions)
        self.assertEqual(206, len(grayscaled_list))  # add assertion here

    def test_process_image_dataset_gray_scaled_to_binary_with_threshold_returns_list(self):
        # Give
        split_of_dataset = "train"
        list_of_dataset = (self.sut_preprocessing
        .convert_dataset_to_list(
            self.dataset, split_of_dataset, "image", "labels"))
        grayscaled_list = self.sut_preprocessing.process_image_dataset_rgb_to_grayscale(list_of_dataset)

        # When
        list_binary = self.sut_preprocessing.process_image_dataset_gray_scaled_to_binary_with_threshold(grayscaled_list)

        # Then
        self.assertIsInstance(list_binary, list)
        for i in range(0, len(list_binary)):
            sample_image = np.array(list_binary[i]["image"])
            is_image_binary = np.all(np.isin(sample_image, [0, 255]))
            self.assertTrue(is_image_binary)
        self.assertEqual(206, len(list_binary))  # add assertion here

    def test_process_image_dataset_binary_to_grid_lines_returns_list(self):
        # Give
        split_of_dataset = "train"
        list_of_dataset = (self.sut_preprocessing
        .convert_dataset_to_list(
            self.dataset, split_of_dataset, "image", "labels"))
        grayscaled_list = self.sut_preprocessing.process_image_dataset_rgb_to_grayscale(list_of_dataset)
        list_binary = self.sut_preprocessing.process_image_dataset_gray_scaled_to_binary_with_threshold(grayscaled_list)

        # When
        list_of_binary_image_with_grid_lines = self.sut_preprocessing.process_image_dataset_binary_to_grid_lines(list_binary)

        # Then
        self.assertIsInstance(list_of_binary_image_with_grid_lines, list)
        for i in range(0, len(list_of_binary_image_with_grid_lines)):
            sample_image = np.array(list_of_binary_image_with_grid_lines[i]["image"])
            is_image_binary = np.all(np.isin(sample_image, [0, 255]))
            self.assertTrue(is_image_binary)
        self.assertEqual(206, len(list_of_binary_image_with_grid_lines))  # add assertion here

    def test_generate_image_dataset_binary_grid_to_list_of_contours(self):
        # Give
        split_of_dataset = "train"
        list_of_dataset = (self.sut_preprocessing
        .convert_dataset_to_list(
            self.dataset, split_of_dataset, "image", "labels"))
        grayscaled_list = self.sut_preprocessing.process_image_dataset_rgb_to_grayscale(list_of_dataset)
        list_binary = self.sut_preprocessing.process_image_dataset_gray_scaled_to_binary_with_threshold(grayscaled_list)
        list_of_binary_image_with_grid_lines = self.sut_preprocessing.process_image_dataset_binary_to_grid_lines(list_binary)
        column_for_contours = "list_of_contours"
        column_for_labels = "labels"
        with_restriction = True
        count_of_needed_contours = 120

        # When
        list_of_contours = self.sut_preprocessing.generate_image_dataset_binary_grid_to_list_of_contours(
            list_of_binary_image_with_grid_lines, column_for_contours, column_for_labels, with_restriction, count_of_needed_contours)

        # Then
        self.assertIsInstance(list_of_contours, list)
        self.assertEqual(len(list_of_binary_image_with_grid_lines), len(list_of_contours))  # add assertion here
        for elem in list_of_contours:
            self.assertIsInstance(elem, dict)
            self.assertEqual(2, len(elem), "The length of elem is not 2")
            self.assertIn(column_for_contours, elem, "The key \"list_of_contours\" not in elem")
            self.assertIn(column_for_labels, elem, "The key \"labels\" not in elem")

    def test_generate_from_contour_list_and_image_list_cut_out_image_to_label_dataset(self):
        # Give
        split_of_dataset = "train"
        list_of_dataset = (self.sut_preprocessing
        .convert_dataset_to_list(
            self.dataset, split_of_dataset, "image", "labels"))
        list_of_gray_scaled = self.sut_preprocessing.process_image_dataset_rgb_to_grayscale(list_of_dataset)
        list_binary = self.sut_preprocessing.process_image_dataset_gray_scaled_to_binary_with_threshold(list_of_gray_scaled)
        list_of_binary_image_with_grid_lines = self.sut_preprocessing.process_image_dataset_binary_to_grid_lines(list_binary)
        column_for_contours = "list_of_contours"
        column_for_labels = "labels"
        with_restriction = True
        count_of_needed_contours = 120
        list_of_contour_list_per_image = self.sut_preprocessing.generate_image_dataset_binary_grid_to_list_of_contours(
            list_of_binary_image_with_grid_lines, column_for_contours, column_for_labels, with_restriction, count_of_needed_contours)

        # When
        column_for_contours = "list_of_contours"
        image_column = "image"
        label_column = "labels"
        dataset_move_boxes_with_labels = (self.sut_preprocessing
        .generate_from_contour_list_and_image_list_cut_out_image_to_label_dataset(
                list_of_contour_list_per_image, column_for_contours, list_of_gray_scaled, image_column, label_column))

        # Then
        np_img = np.array(dataset_move_boxes_with_labels[0]["image"])
        for elem in dataset_move_boxes_with_labels:
            self.assertIsInstance(dataset_move_boxes_with_labels, Dataset)
            self.assertEqual(2, len(elem), "The length of elem is not 2")
            self.assertIn(image_column, elem, f"The key \"{image_column}\" not in elem")
            self.assertIn("label", elem, f"The key \"label\" not in elem")

    def test_transform_returns_dataset_with_cut_out_move_boxes(self):
        # When
        res_dataset = self.sut_preprocessing.preprocess_dataset(self.dataset)

        # Then
        self.assertIsInstance(res_dataset, Dataset)
        is_bigger_list = len(res_dataset) > len(self.dataset)
        self.assertTrue(is_bigger_list)

if __name__ == '__main__':
    unittest.main()
