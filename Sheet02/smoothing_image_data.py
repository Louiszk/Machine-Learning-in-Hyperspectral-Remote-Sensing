import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import rasterio

def interpolate(veg_im, veg_wl, new_wl):
    bands, rows, cols = veg_im.shape
    pixel_band_table = veg_im.reshape(bands, rows * cols).T  # (bands, rows, columns) -> (pixels, bands)
    
    # Interpolating
    interpolated_pixel_band_table = np.array([np.interp(new_wl, veg_wl, spectrum) for spectrum in pixel_band_table])
    
    return interpolated_pixel_band_table

def savgol_smoothing(pixel_band_table, n, p):

    smoothed_pixel_band_table = savgol_filter(pixel_band_table, window_length=n, polyorder=p, axis=1)
    return smoothed_pixel_band_table

def interpolate_back(smoothed_pixel_band_table, new_wl, original_wl):
    interpolated_back_pixel_band_table = np.array([np.interp(original_wl, new_wl, spectrum) for spectrum in smoothed_pixel_band_table])
    return interpolated_back_pixel_band_table

if __name__ == "__main__":
    path = "Sheet02/data/image/"

    with rasterio.open(path + "Suwannee_0609-1331_ref.dat") as dataset:
        veg_im = dataset.read() 
        profile = dataset.profile

    #print(profile)
    #print(veg_im[0, :5, :5])
    #print(veg_im.shape) # 360 bands 1200x320
    veg_im = veg_im[: , :100, :100] # subset 100x100

    veg_wl = pd.read_csv(path + "Suwannee_0609-1331_wl.txt", sep=" ", header=None)
    veg_wl = veg_wl.values[0] * 1000 # nm
    print(veg_wl.shape)

    new_wl = np.arange(min(veg_wl), max(veg_wl) + 1, 1)  # Regular interval of 1 nm
    #print(new_wl)
    
    interpolated_pixel_band_table = interpolate(veg_im, veg_wl, new_wl)
    
    #window sizes should be odd
    n_values = [21, 81, 131, 201, 21, 81, 141, 41, 121, 101, 181]
    p_values = [2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5]

    smoothed_images = []

    for n, p in zip(n_values, p_values):
        # Savitzky-Golay
        smoothed_pixel_band_table = savgol_smoothing(interpolated_pixel_band_table, n, p)
        
        # interpolate back to original wavelengths
        smoothed_original_wavelengths = interpolate_back(smoothed_pixel_band_table, new_wl, veg_wl)
        
        # Reshape back
        smoothed_image = smoothed_original_wavelengths.T.reshape(veg_im.shape)
        
        smoothed_images.append(smoothed_image)
    
    #Plotting
    fig, axs = plt.subplots(2, 6, figsize=(18, 8))

    axs[0, 0].imshow(veg_im[0], cmap='gray')
    axs[0, 0].set_title('Original Band')
    axs[0, 0].axis('off')


    for i in range(len(smoothed_images)):
        row = (i + 1) // 6   
        col = (i + 1) % 6    
        axs[row, col].imshow(smoothed_images[i][0], cmap='gray')
        axs[row, col].set_title(f'Smoothed n={n_values[i]}, p={p_values[i]}')
        axs[row, col].axis('off') 


    plt.tight_layout()
    plt.show()
