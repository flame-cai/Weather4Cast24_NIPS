"""
Data preprocessing utilities for Weather4Cast challenge.
Includes functions for downsampling and data resampling.
"""

import h5py
import numpy as np
import cv2
from tqdm import tqdm
from skimage import filters


def downsample_h5(input_file, output_file, target_shape, batch_size=100):
    """
    Downsample HDF5 file to target shape.

    Args:
        input_file (str): Path to input HDF5 file
        output_file (str): Path to output HDF5 file
        target_shape (tuple): Target shape for downsampling (height, width)
        batch_size (int): Batch size for processing
    """
    with h5py.File(input_file, "r") as f:
        dataset = f["rates.crop"]
        total_frames = dataset.shape[0]

        with h5py.File(output_file, "w") as out_f:
            downsampled_data = out_f.create_dataset(
                "rates.crop",
                (total_frames, target_shape[0], target_shape[1]),
                dtype=np.float32,
            )

            for start in tqdm(
                range(0, total_frames, batch_size), desc="Processing Batches"
            ):
                end = min(start + batch_size, total_frames)
                batch_data = dataset[start:end]

                batch_downsampled = np.zeros(
                    (batch_data.shape[0], target_shape[0], target_shape[1]),
                    dtype=np.float32,
                )

                for i in range(batch_data.shape[0]):
                    batch_downsampled[i] = cv2.resize(
                        batch_data[i, 0], target_shape, interpolation=cv2.INTER_LINEAR
                    )

                downsampled_data[start:end] = batch_downsampled


def process_and_save_resampled(input_file, output_file, selected_channels=[3, 4, 5, 6]):
    """
    Process and save resampled data with Otsu thresholding.

    Args:
        input_file (str): Path to input HDF5 file
        output_file (str): Path to output HDF5 file
        selected_channels (list): List of channel indices to process
    """
    with h5py.File(input_file, "r") as infile:
        dataset = infile["REFL-BT"]
        img = dataset[:, selected_channels, :, :]
        img = np.mean(img, axis=1)
        num_images = dataset.shape[0]

        with h5py.File(output_file, "w") as outfile:
            binarized_dataset = outfile.create_dataset(
                "Binarized-REFL-BT", shape=(num_images, 252, 252), dtype=dataset.dtype
            )

            for i in tqdm(range(num_images), desc="Processing Images"):
                image = np.squeeze(np.mean(dataset[i : i + 1, 1:, :, :], axis=1))
                thresh = filters.threshold_otsu(image)
                mask = image < thresh
                masked_img = np.copy(image)
                masked_img[mask == 0] = np.nan
                binarized_dataset[i, :, :] = masked_img


def preprocess_all(base_path, year, file_types=["val", "train"]):
    """
    Preprocess all data for a given year.

    Args:
        base_path (str): Base path to data directory
        year (int): Year to process
        file_types (list): List of file types to process
    """
    target_shape = (252, 252)

    for file_type in file_types:
        # Process OPERA files
        opera_input = f"{base_path}/{year}/OPERA-CONTEXT/roxi_*.{file_type}{str(year)[-2:]}.rates.crop.h5"
        opera_output = f"{base_path}/{year}/OPERA-CONTEXT/roxi_*.{file_type}{str(year)[-2:]}.rates.crop.252.h5"
        downsample_h5(opera_input, opera_output, target_shape)

        # Process HRIT files
        hrit_input = f"{base_path}/{year}/HRIT/roxi_*.{file_type}.reflbt0.ns.h5"
        hrit_output = f"{base_path}/{year}/HRIT/roxi_*.{file_type}.binarized.252.h5"
        process_and_save_resampled(hrit_input, hrit_output)


if __name__ == "__main__":
    base_path = "../data"
    for year in [2019, 2020]:
        print(f"Processing year {year}")
        preprocess_all(base_path, year)
