from image_files_to_dataset_strategy import UnprocessedHcsImageLabelDirToDatasetStrategy, ProcessedHcsImageLabelDirToDatasetStrategy
from images_to_hf_dataset import HuggingfaceDataVersioning

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