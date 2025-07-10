import unittest
import cv2
import numpy as np

from preprocessing_strategy import HuggingFacePreprocessingStrategy, ThresholdMethod
from ingest_data_strategy import HuggingFaceImageDataIngestorStrategy


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Give
        owner = "BenjaminKost"
        dataset_name = "unprocessed_hcs"
        ingestor = HuggingFaceImageDataIngestorStrategy()
        cls.sut_preprocessing = HuggingFacePreprocessingStrategy()
        cls.dataset = ingestor.ingest_data(owner=owner, dataset_name=dataset_name)

    # 1. Hyperparameter configuration
    def test_preprocess_image_dataset_with_first_config(self):
        # When
        improving_image_processed = 2016
        res_dataset = self.sut_preprocessing.preprocess_dataset(self.dataset)

        # Then
        self.assertGreaterEqual(len(res_dataset), improving_image_processed)

    # 1. Hyperparameter configuration
    def test_process_image_dataset_gray_scaled_to_binary_with_threshold_first_config(self):
        # Give
        list_dataset = self.sut_preprocessing.convert_dataset_to_list(self.dataset, "train", "image", "labels")
        list_gray_scaled = self.sut_preprocessing.process_image_dataset_rgb_to_grayscale(list_dataset)
        threshold_method = ThresholdMethod.OTSU

        # When
        improving_image_processed = 2400

        res_dataset = self.sut_preprocessing.process_image_dataset_gray_scaled_to_binary_with_threshold(list_gray_scaled)

        for i, data in enumerate(res_dataset):
            # Extrahiere das Bild
            img = np.array(data["image"])  # Konvertiere das Bild in ein numpy-Array, falls nötig

            # Zeigt das Bild an
            cv2.imshow(f"Image {i + 1}", img)
            print(f"Showing Image {i + 1}. Press 'Enter' to go to the next image or 'q' to quit.")

            # Warten auf Benutzereingabe
            key = cv2.waitKey(0)  # Wartet auf eine beliebige Taste
            if key == 13:  # Enter-Taste
                cv2.destroyAllWindows()  # Schließt das aktuelle Bild
            elif key == ord('q'):  # Mit 'q' kann der Benutzer abbrechen
                print("Exit requested. Closing...")
                cv2.destroyAllWindows()
                break

        # Then
        self.assertGreater(len(res_dataset), improving_image_processed)


    # 2-4. Hyperparameter configuration
    def test_preprocess_image_dataset_with_second_config(self):
        # Give
        threshold_method = ThresholdMethod.ADAPTIVE_THRESH_MEAN_C

        # When
        improving_image_processed = 2400

        res_dataset = self.sut_preprocessing.preprocess_dataset(self.dataset)

        # Then
        self.assertGreater(len(res_dataset), improving_image_processed)

    # 5-7. Hyperparameter configuration
    def test_preprocess_image_dataset_with_third_config(self):
        # Give
        threshold_method = ThresholdMethod.ADAPTIVE_THRESH_GAUSSIAN_C

        # When
        improving_image_processed = 2400

        res_dataset = self.sut_preprocessing.preprocess_dataset(self.dataset)

        for i, data in enumerate(res_dataset):
            # Extrahiere das Bild
            img = np.array(data["image"])  # Konvertiere das Bild in ein numpy-Array, falls nötig

            # Zeigt das Bild an
            cv2.imshow(f"Image {i + 1}", img)
            print(f"Showing Image {i + 1}. Press 'Enter' to go to the next image or 'q' to quit.")

            # Warten auf Benutzereingabe
            key = cv2.waitKey(0)  # Wartet auf eine beliebige Taste
            if key == 13:  # Enter-Taste
                cv2.destroyAllWindows()  # Schließt das aktuelle Bild
            elif key == ord('q'):  # Mit 'q' kann der Benutzer abbrechen
                print("Exit requested. Closing...")
                cv2.destroyAllWindows()
                break

        # Then
        self.assertGreater(len(res_dataset), improving_image_processed)

    # 5-7. Hyperparameter configuration
    def test_process_image_dataset_gray_scaled_to_binary_with_threshold_fifth_config(self):
        # Give
        list_dataset = self.sut_preprocessing.convert_dataset_to_list(self.dataset, "train", "image", "labels")
        list_gray_scaled = self.sut_preprocessing.process_image_dataset_rgb_to_grayscale(list_dataset)
        threshold_method = ThresholdMethod.ADAPTIVE_THRESH_MEAN_C

        # When
        improving_image_processed = 2400

        res_dataset = self.sut_preprocessing.process_image_dataset_gray_scaled_to_binary_with_threshold(list_gray_scaled)

        for i, data in enumerate(res_dataset):
            # Extrahiere das Bild
            img = np.array(data["image"])  # Konvertiere das Bild in ein numpy-Array, falls nötig

            # Zeigt das Bild an
            cv2.imshow(f"Image {i + 1}", img)
            print(f"Showing Image {i + 1}. Press 'Enter' to go to the next image or 'q' to quit.")

            # Warten auf Benutzereingabe
            key = cv2.waitKey(0)  # Wartet auf eine beliebige Taste
            if key == 13:  # Enter-Taste
                cv2.destroyAllWindows()  # Schließt das aktuelle Bild
            elif key == ord('q'):  # Mit 'q' kann der Benutzer abbrechen
                print("Exit requested. Closing...")
                cv2.destroyAllWindows()
                break

        # Then
        self.assertGreater(len(res_dataset), improving_image_processed)

    # 8. Hyperparameter configuration
    def test_process_image_dataset_gray_scaled_to_binary_with_threshold_eight_config(self):
        # Give
        list_dataset = self.sut_preprocessing.convert_dataset_to_list(self.dataset, "train", "image", "labels")
        list_gray_scaled = self.sut_preprocessing.process_image_dataset_rgb_to_grayscale(list_dataset)
        threshold_method = ThresholdMethod.ADAPTIVE_THRESH_MEAN_C

        # When
        improving_image_processed = 2400

        res_dataset = self.sut_preprocessing.process_image_dataset_gray_scaled_to_binary_with_threshold(list_gray_scaled)

        for i, data in enumerate(res_dataset):
            # Extrahiere das Bild
            img = np.array(data["image"])  # Konvertiere das Bild in ein numpy-Array, falls nötig

            # Zeigt das Bild an
            cv2.imshow(f"Image {i + 1}", img)
            print(f"Showing Image {i + 1}. Press 'Enter' to go to the next image or 'q' to quit.")

            # Warten auf Benutzereingabe
            key = cv2.waitKey(0)  # Wartet auf eine beliebige Taste
            if key == 13:  # Enter-Taste
                cv2.destroyAllWindows()  # Schließt das aktuelle Bild
            elif key == ord('q'):  # Mit 'q' kann der Benutzer abbrechen
                print("Exit requested. Closing...")
                cv2.destroyAllWindows()
                break

        # Then
        self.assertGreater(len(res_dataset), improving_image_processed)

    # 8. Hyperparameter configuration
    def test_process_image_dataset_gray_scaled_to_binary_with_threshold_nine_config(self):
        # Give
        list_dataset = self.sut_preprocessing.convert_dataset_to_list(self.dataset, "train", "image", "labels")
        list_gray_scaled = self.sut_preprocessing.process_image_dataset_rgb_to_grayscale(list_dataset)
        threshold_method = ThresholdMethod.ADAPTIVE_THRESH_MEAN_C

        # When
        improving_image_processed = 2400

        res_dataset = self.sut_preprocessing.process_image_dataset_gray_scaled_to_binary_with_threshold(list_gray_scaled)

        for i, data in enumerate(res_dataset):
            # Extrahiere das Bild
            img = np.array(data["image"])  # Konvertiere das Bild in ein numpy-Array, falls nötig

            # Zeigt das Bild an
            cv2.imshow(f"Image {i + 1}", img)
            print(f"Showing Image {i + 1}. Press 'Enter' to go to the next image or 'q' to quit.")

            # Warten auf Benutzereingabe
            key = cv2.waitKey(0)  # Wartet auf eine beliebige Taste
            if key == 13:  # Enter-Taste
                cv2.destroyAllWindows()  # Schließt das aktuelle Bild
            elif key == ord('q'):  # Mit 'q' kann der Benutzer abbrechen
                print("Exit requested. Closing...")
                cv2.destroyAllWindows()
                break

        # Then
        self.assertGreater(len(res_dataset), improving_image_processed)

    # 9. Hyperparameter configuration
    def test_process_image_dataset_gray_scaled_to_binary_with_threshold_nine_second_config(self):
        # Give
        list_dataset = self.sut_preprocessing.convert_dataset_to_list(self.dataset, "train", "image", "labels")
        list_gray_scaled = self.sut_preprocessing.process_image_dataset_rgb_to_grayscale(list_dataset)
        threshold_method = ThresholdMethod.ADAPTIVE_THRESH_MEAN_C

        # When
        improving_image_processed = 2400

        res_dataset = self.sut_preprocessing.process_image_dataset_gray_scaled_to_binary_with_threshold(list_gray_scaled)

        for i, data in enumerate(res_dataset):
            # Extrahiere das Bild
            img = np.array(data["image"])  # Konvertiere das Bild in ein numpy-Array, falls nötig

            # Zeigt das Bild an
            cv2.imshow(f"Image {i + 1}", img)
            print(f"Showing Image {i + 1}. Press 'Enter' to go to the next image or 'q' to quit.")

            # Warten auf Benutzereingabe
            key = cv2.waitKey(0)  # Wartet auf eine beliebige Taste
            if key == 13:  # Enter-Taste
                cv2.destroyAllWindows()  # Schließt das aktuelle Bild
            elif key == ord('q'):  # Mit 'q' kann der Benutzer abbrechen
                print("Exit requested. Closing...")
                cv2.destroyAllWindows()
                break

        # Then
        self.assertGreater(len(res_dataset), improving_image_processed)

    # 9. Hyperparameter configuration
    def test_preprocess_image_dataset_with_nine_config(self):
        # When
        improving_image_processed = 2016
        res_dataset = self.sut_preprocessing.preprocess_dataset(self.dataset)

        # Then
        self.assertGreaterEqual(len(res_dataset), improving_image_processed)


        # Then
        self.assertGreater(len(res_dataset), improving_image_processed)

if __name__ == '__main__':
    unittest.main()
