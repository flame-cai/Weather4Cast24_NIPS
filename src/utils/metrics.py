import numpy as np
import h5py 

def crps(forecast, observation):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) between two single-channel images.
    
    :param forecast: numpy array representing the forecast image
    :param observation: numpy array representing the observed image
    :return: CRPS value
    """
    # Ensure the inputs are numpy arrays
    forecast = np.asarray(forecast)
    observation = np.asarray(observation)
    
    # Check if the shapes match
    if forecast.shape != observation.shape:
        raise ValueError("Forecast and observation arrays must have the same shape")
    
    # Flatten the arrays
    forecast = forecast.flatten()
    observation = observation.flatten()
    
    # Sort the forecast array
    forecast_sorted = np.sort(forecast)
    
    # Calculate the empirical CDF of the forecast
    n = len(forecast)
    forecast_cdf = np.arange(1, n + 1) / n
    
    # Calculate the Heaviside step function for the observation
    heaviside = np.where(forecast_sorted >= observation[:, np.newaxis], 1.0, 0.0)
    
    # Calculate CRPS
    crps = np.mean(np.square(forecast_cdf - heaviside))
    
    return crps



input_data = "../data/2019/HRIT/boxi_0015.train.binarized.252.h5'
hrit_file = h5py.File(input_data, 'r')
img = hrit_file['Binarized-REFL-BT'][:]  # Shape: (20308, 252, 252)
# selected_channels_indices = [3, 4, 5, 6]
# img = img[50:60, selected_channels_indices, :, :]
# img = np.mean(img, axis=1)
print(img.shape)
img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
img=img[50:60,:,:]
print(img.shape)

target_data = '../data/2019/OPERA-CONTEXT/boxi_0015.train19.rates.crop.252.h5'
opera_file = h5py.File(target_data, 'r')
target_data = opera_file['rates.crop'][:]
target_data = np.nan_to_num(target_data, nan=0.0, posinf=0.0, neginf=0.0)
target_data=target_data[50:60,:,:]
print(target_data.shape)

print(crps(np.squeeze(target_data[-1]),np.squeeze(target_data[-1])))
