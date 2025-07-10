from PIL import Image

from app.ml.classes_for_steps.preprocessing_strategy import ThresholdMethod, HuggingFacePreprocessingStrategy


# @step
def preprocessing_image(image: Image) -> list:
    # load preprocessing strategy
    ## Best parameters tested with tuning: src/tuning/preprocessing/preprocessing_hyperparameter_tuning.py
    kernelsize_gaussianBlur = (5, 5)
    sigmaX = 0
    threshold_method = ThresholdMethod.OTSU
    maxValue_threshold = 255
    block_size = 9
    c_value = 1
    horizontal_kernel_divisor = 30
    vertical_kernel_divisor = 20
    erosion_iterations = 1
    dilation_iterations = 1

    preprocessing_strategy = HuggingFacePreprocessingStrategy(
        kernelsize_gaussianBlur=kernelsize_gaussianBlur,
        sigmaX=sigmaX,
        threshold_method=threshold_method,
        maxValue_threshold=maxValue_threshold,
        block_size=block_size,
        c_value=c_value,
        horizontal_kernel_divisor=horizontal_kernel_divisor,
        vertical_kernel_divisor=vertical_kernel_divisor,
        erosion_iterations=erosion_iterations,
        dilation_iterations=dilation_iterations
    )

    # Image into gray scale
    image_gray_scaled = image.convert("L")

    # Convert gray-scale image to binary image with otsu's method
    image_binary = preprocessing_strategy.process_image_gray_scaled_to_binary_with_threshold(image_gray_scaled)

    ## Generate image containing only grid lines
    image_only_grid_lines = preprocessing_strategy.process_binary_image_to_grid_lines(image_binary)

    ## Find contours in image with only grid lines
    list_of_contour_for_image = preprocessing_strategy.generate_binary_grid_image_to_list_of_contours(image_only_grid_lines)

    ## Cut out boxes with padding
    list_cut_out_move_boxes = []
    for contour in list_of_contour_for_image:
        cut_out_image = preprocessing_strategy.generate_from_four_contour_points_and_image_a_cut_out_image(contour, image)
        cut_out_image_rgb = cut_out_image.convert("RGB")
        list_cut_out_move_boxes.append(cut_out_image_rgb)

    return list_cut_out_move_boxes