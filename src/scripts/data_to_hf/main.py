from image_files_to_dataset_strategy import HSSSDImageLabelDirToDatasetStrategy
from images_to_hf_dataset import HuggingfaceDataVersioning
from src.scripts.utils.config import get_root_dir_path

if __name__ == "__main__":
    """
    # Unprocessed HCS
    ## input parameter
    path_to_dataset = "./data/datasets/unprocessed_hcs"
    path_to_image_dir = "./data/raw_data/unprocessed_hcs_data/images"
    path_to_label_file = "./data/raw_data/unprocessed_hcs_data/training_tags.txt"
    owner = "BenjaminKost"
    dataset_name = "unprocessed_hcs"

    ## Execution
    strategy = UnprocessedHcsImageLabelDirToDatasetStrategy()
    HuggingfaceDataVersioning(strategy).upload_dataset(path_to_dataset, path_to_image_dir, path_to_label_file, owner, dataset_name)
    """
    """
    # Processed HCS
    ## input parameter
    path_to_dataset = "./data/datasets/processed_hcs"
    path_to_image_dir = "./data/raw_data/processed_hcs_data/images"
    path_to_label_file = "./data/raw_data/processed_hcs_data/training_tags.txt"
    owner = "BenjaminKost"
    dataset_name = "processed_hcs"

    ## Execution
    strategy = ProcessedHcsImageLabelDirToDatasetStrategy()
    HuggingfaceDataVersioning(strategy).upload_dataset(path_to_dataset, path_to_image_dir, path_to_label_file, owner, dataset_name)
    """

    # Processed HCS
    ## input parameter
    path_to_dataset = get_root_dir_path() / "data" / "datasets" / "hsssd_data"
    path_to_image_dir = get_root_dir_path() / "data" / "raw_data" / "hsssd_data" / "images"
    path_to_label_file = get_root_dir_path() / "data" / "raw_data" / "hsssd_data" / "ground_truth_labels.csv"
    owner = "BenjaminKost"
    dataset_name = "HSSSD"

    ## Execution
    strategy = HSSSDImageLabelDirToDatasetStrategy()
    HuggingfaceDataVersioning(strategy).upload_dataset(path_to_dataset, path_to_image_dir, path_to_label_file, owner, dataset_name)
