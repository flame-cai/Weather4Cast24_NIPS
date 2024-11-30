import cv2
from skimage import filters
import numpy as np
import pandas as pd
import torch
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import csv
import os
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.extrapolation.semilagrangian import extrapolate
from src.models import ImprovedGenerator
from src.utils import crps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the output folder
output_folder = "../output_folder"
os.makedirs(output_folder, exist_ok=True)

# Load the trained generator model
generator = ImprovedGenerator(in_channels=1, out_channels=1)
generator.load_state_dict(
    torch.load("../improved_generator_5_200i.pth"), strict=False
)
generator.eval()


def pad_image(image, pad_size=10):
    return cv2.copyMakeBorder(
        image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE
    )


def create_edge_mask(shape, edge_width=10):
    mask = np.ones(shape)
    mask[:edge_width, :] = np.linspace(0, 1, edge_width)[:, np.newaxis]
    mask[-edge_width:, :] = np.linspace(1, 0, edge_width)[:, np.newaxis]
    mask[:, :edge_width] *= np.linspace(0, 1, edge_width)
    mask[:, -edge_width:] *= np.linspace(1, 0, edge_width)
    return mask


def interpolate_frames(frame, coords, flow, edge_mask):
    frames = []
    previous_frame = frame.copy()
    for f in range(1, 17):  # Predict 16 frames
        pixel_map = coords + (f / 4) * flow
        inter_frame = cv2.remap(frame, pixel_map, None, cv2.INTER_LINEAR)
        inter_frame = inter_frame * edge_mask + previous_frame * (1 - edge_mask)
        zero_mask = inter_frame < 75
        inter_frame[zero_mask] = previous_frame[zero_mask]
        frames.append(inter_frame)
        previous_frame = inter_frame
    return frames


def save_image(image, title, vmin, vmax):
    plt.figure(figsize=(10, 10))
    img_plot = plt.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.colorbar(img_plot, fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(output_folder, f"{title}.png"))
    plt.close()


def compute_stats(image):
    return {"min": np.nanmin(image), "max": np.nanmax(image), "mean": np.nanmean(image)}


def apply_otsu_threshold(image):
    thresh = filters.threshold_otsu(image)
    masked_img = np.copy(image)
    mask = image < thresh
    masked_img[mask == 0] = np.nan
    return masked_img.astype(np.float32)


def resample_image(image, target_shape=(1512, 1512)):
    return cv2.resize(image, target_shape, interpolation=cv2.INTER_LINEAR)


def process_case(img, slot_start, slot_end, x_start, x_end, y_start, y_end):
    # Prepare for optical flow
    pad_size = 10
    img_shape = img.shape[1:]
    padded_shape = (img_shape[0] + 2 * pad_size, img_shape[1] + 2 * pad_size)
    y_coords, x_coords = np.mgrid[0 : padded_shape[0], 0 : padded_shape[1]]
    coords = np.float32(np.dstack([x_coords, y_coords]))

    # Convert the image to 32-bit float single-channel format
    img_float32 = img.astype(np.float32)
    # img_padded = pad_image(img_float32[slot_start], pad_size)
    img_padded = pad_image(img_float32[slot_start:slot_end])

    # Calculate optical flow
    motion_field = dense_lucaskanade(
        img_padded, dense=True, fd_method="blob", interp_method="rbfinterp2d"
    )
    predicted_frames = extrapolate(
        img_padded[-1],
        motion_field,
        timesteps=16,
        interp_order=3,
        return_displacement=False,
    )

    # Process each interpolated frame through pix2pix
    pix2pix_outputs = []
    for frame in predicted_frames:
        # Apply Otsu's thresholding
        # frame[frame==np.nan]=0
        frame = np.nan_to_num(
            frame,
            nan=np.nanmax(frame),
            posinf=np.nanmax(frame),
            neginf=np.nanmax(frame),
        )
        print(np.nanmin(frame), np.nanmax(frame))
        frame[frame == 0] = np.nanmax(frame)
        thresholded_frame = apply_otsu_threshold(frame)

        # Normalize and prepare the frame for pix2pix input
        input_image = thresholded_frame / 150 - 1  # Scale to [-1, 1]
        input_image = np.nan_to_num(input_image, nan=0.0, posinf=0.0, neginf=0.0)
        input_image = torch.tensor(input_image, dtype=torch.float32)

        # Pad the input image to 256x256
        padding = (0, 256 - input_image.shape[1], 0, 256 - input_image.shape[0])
        input_image = torch.nn.functional.pad(input_image, padding)

        # Generate prediction using pix2pix
        with torch.no_grad():
            input_sequence = input_image.unsqueeze(0).unsqueeze(0)
            pix2pix_output = ((generator(input_sequence)) + 1) * 5
            pix2pix_output = pix2pix_output.squeeze().cpu().numpy()
            # print(np.unique(pix2pix_output))
            # pix2pix_output[pix2pix_output > 4.9] = 0
            pix2pix_output[pix2pix_output < 0.001] = 0
            pix2pix_output = pix2pix_output[:252, :252]

        # Resample pix2pix output to 1512x1512
        resampled_output = resample_image(pix2pix_output)
        # print(resampled_output.sum())
        pix2pix_outputs.append(resampled_output)

    # Extract 32x32 box and compute average
    box_averages = [
        output[y_start:y_end, x_start:x_end].mean() for output in pix2pix_outputs
    ]
    print(f"Final average: {(np.mean(box_averages))*4*10}")
    return (np.mean(box_averages)) * 4 * 10


def process_file(year, file_number):
    # Set the output folder
    output_folder = f"../Submissions_t/{year}/"
    os.makedirs(output_folder, exist_ok=True)

    # Use zero padding for file number
    padded_file_number = f"{file_number:04d}"

    # Determine the correct year suffix for the input filename
    year_suffix = str(year)[-2:]  # Get last two digits of the year

    # Load and preprocess HRIT data
    input_data = f"../{year}/HRIT/roxi_{padded_file_number}.cum1test{year_suffix}.reflbt0.ns.h5"
    with h5py.File(input_data, "r") as hrit_file:
        img = hrit_file["REFL-BT"][:]
        selected_channels_indices = [3, 4, 5, 6]
        img = img[:, selected_channels_indices, :, :]
        img = np.mean(img, axis=1)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    # Read input CSV
    input_csv = (
        f"/..t/roxi_{padded_file_number}.cum1test_dictionary.csv"
    )
    df = pd.read_csv(input_csv)

    # Process each case and store results
    results = []
    for _, row in df.iterrows():
        if row["year"] == year:
            case_id = row["Case-id"]
            slot_start = row["slot-start"]
            slot_end = row["slot-end"]
            x_start = row["x-top-left"]
            x_end = row["x-bottom-right"]
            y_start = row["y-top-left"]
            y_end = row["y-bottom-right"]

            average_value = process_case(
                img, slot_start, slot_end, x_start, x_end, y_start, y_end
            )
            results.append([case_id, np.round(average_value, decimals=2), 1])

    # Create output DataFrame and save to CSV
    output_csv = os.path.join(
        output_folder, f"roxi_{padded_file_number}.test.cum4h.csv"
    )
    with open(output_csv, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(results)

    print(
        f"Processing complete for roxi_{padded_file_number} {year}. Results written to {output_csv}"
    )


# Main execution
years = [2019, 2020]
file_numbers = [8, 9, 10]

for year in years:
    for file_number in file_numbers:
        process_file(year, file_number)

print("All processing complete.")

