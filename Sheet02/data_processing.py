import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def show_plot(dat, wl):
    plt.figure(figsize=(10, 6))
    for i in range(4): 
        plt.plot(wl, dat[i], label=f'Spectra {i}')
    
    plt.xlabel('Wavelength / nm')
    plt.ylabel('Pixel value')
    plt.legend()
    plt.title('Spectral Data')
    plt.show(block=False)
    plt.pause(8)

def read_data(show_initial_plot = False):
    path = "Sheet02/data/"
    dat = pd.read_csv(path + "allplotsaggregatedspectra.txt", sep=" ")
    wl = pd.read_csv(path + "bands.txt", sep=",")
    traits = pd.read_csv(path + "N_percDW_all.txt", sep="\t")

    #print(list(dat.columns), len(list(dat.columns)), len(dat.iloc[0]))
    #print(list(traits.columns), len(list(traits.columns)), len(traits.iloc[0]))
    #print(list(wl.columns), len(list(wl.columns)), len(wl.iloc[0]))
    #print(dat.head())
    #print(traits.head())
    #print(wl.head())
    n = traits.iloc[:, 1:].mean(axis=1)
    #print(n)

    #First plotting
    if show_initial_plot:
        plt.figure(figsize=(10, 6))
        for i in range(4): 
            plt.plot(wl['WVL'], dat.iloc[i], label=f'spectra_{i}')
            
        bad_bands = wl[wl['IsBad'] == 1]['WVL']
        plt.scatter(bad_bands, np.zeros_like(bad_bands), color='red', marker='o', s=20)

        # First plot to answer Q2.1
        plt.xlabel('Wavelength / nm')
        plt.ylabel('Pixel value')
        plt.legend()
        plt.title('Spectral Data with Bad Bands Underlined')
        plt.show(block=False)

    # Further Processing
    # good bands
    good_bands = wl['IsBad'] == 0
    #print(len(good_bands), dat.shape[1])
    #print(good_bands, good_bands.values)
    wl_filtered = wl[good_bands].drop_duplicates(subset='WVL')
    dat_filtered = dat.iloc[:, wl_filtered.index.values]

    # Plotting again (gaps are connected)
    if show_initial_plot:
        show_plot(dat_filtered.values, wl_filtered['WVL'].values)
    
    return dat_filtered, wl_filtered, n

def resampling(dat, wl):
    wvl = wl['WVL'].values
    data = dat.values
    wl_out = np.arange(wvl[0], wvl[-1] + 1, 1)
    
    # Interpolating data
    dat_resampled = np.array([np.interp(wl_out, wvl, spectrum) for spectrum in data])
    
    show_plot(dat_resampled, wl_out)
    return dat_resampled, wl_out

def calculate_ndre(data, wvl):
    
    idx_720 = np.where(wvl == 720)[0][0]
    idx_790 = np.where(wvl == 790)[0][0]
    
    reflect_720 = data[:, idx_720]
    reflect_790 = data[:, idx_790]
    
    return (reflect_790 - reflect_720) / (reflect_790 + reflect_720)

def calculate_reip(data, wvl):

    idx_670 = np.where(wvl == 670)[0][0]
    idx_700 = np.where(wvl == 700)[0][0]
    idx_740 = np.where(wvl == 740)[0][0]
    idx_780 = np.where(wvl == 780)[0][0]
    
    reflect_670 = data[:, idx_670]
    reflect_700 = data[:, idx_700]
    reflect_740 = data[:, idx_740]
    reflect_780 = data[:, idx_780]
    
    reip = 700 + 40 * (((reflect_670 + reflect_780) / 2 - reflect_700) / (reflect_740 - reflect_700))
    return reip

def plot_distributions(arr1, arr2, size_one_bins = False):
    fig, axs = plt.subplots(2, figsize=(8,6))

    axs[0].hist(arr1, bins=np.arange(min(arr1), max(arr1) + 2, 1) if size_one_bins else 30, color='blue')
    axs[0].set_title('Histogram first distribution')
    axs[0].set_xlabel('Values')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(arr2, bins=np.arange(min(arr2), max(arr2) + 2, 1) if size_one_bins else 30, color='green')
    axs[1].set_title('Histogram second distribution')
    axs[1].set_xlabel('Values')
    axs[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def plot_nitrogen(n, ndre_values, reip_values):
    cor_ndre = np.corrcoef(n, ndre_values)[0, 1] ** 2
    cor_reip = np.corrcoef(n, reip_values)[0, 1] ** 2

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].scatter(n, ndre_values)
    axs[0].set_title(f'NDRE Leaf Nitrogen\n R2 = {cor_ndre:.4f}')
    axs[0].set_xlabel('Leaf Nitrogen (%)')
    axs[0].set_ylabel('NDRE')
    axs[0].grid(True)

    axs[1].scatter(n, reip_values)
    axs[1].set_title(f'REIP Leaf Nitrogen\n R2 = {cor_reip:.4f}')
    axs[1].set_xlabel('Leaf Nitrogen (%)')
    axs[1].set_ylabel('REIP')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_savgol(data, wvl, derivative=0):
    p = 3 
    n = 131
    data_smoothed = np.apply_along_axis(lambda m: savgol_filter(m, n, p, deriv = derivative), axis=1, arr=data)

    plt.figure(figsize=(12, 6))
    if not derivative:
        plt.plot(wvl, data[0, :], label='Original Spectrum')
    plt.plot(wvl, data_smoothed[0, :], color='red', label='Smoothed Spectrum (p=3, n=131)' if not derivative else 'Derivative')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Pixel Value')
    plt.title('Savitzky-Golay')
    plt.legend()
    plt.show()

    return data_smoothed

def plot_variations_savgol(data, wvl):
    polys = [2, 4, 8] 
    #window sizes should be odd
    windows = [51, 101, 151, 201]

    plt.figure(figsize=(12, 10))

    for i, p in enumerate(polys):
        for j, n in enumerate(windows):
            if n < data.shape[1]:
                smoothed = np.apply_along_axis(lambda m: savgol_filter(m, n, p), axis=1, arr=data)
                # Plotting
                plt.subplot(len(polys), len(windows), i * len(windows) + j + 1)
                plt.plot(wvl, data[0, :], alpha=0.3)
                plt.plot(wvl, smoothed[0, :], color='red')
                plt.title(f'p={p}, n={n}')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Importing and screening the data
    dat, wl, n = read_data(show_initial_plot=True)

    #Resampling the spectra to 1 nm resolution
    data, wvl = resampling(dat, wl) # now two numpy arrays

    #Narrow band vegetation indices
    ndre = calculate_ndre(data, wvl)
    reip = calculate_reip(data, wvl)
    print(min(ndre), max(ndre), min(reip), max(reip)) # Q2.3
    plot_distributions(ndre, reip) # Q2.3
    plot_nitrogen(n, ndre, reip) # Q2.4

    # Spectral smoothing
    data_smoothed = plot_savgol(data, wvl) 
    plot_variations_savgol(data, wvl) # Q2.6
    ndre2 = calculate_ndre(data_smoothed, wvl)
    reip2 = calculate_reip(data_smoothed, wvl)
    plot_nitrogen(n, ndre2, reip2) # Q2.7
    deriv_savgol = plot_savgol(data, wvl, derivative=1)
    mxd1 = wvl[np.argmax(deriv_savgol, axis=1)]
    print(min(reip2), max(reip2), min(mxd1), max(mxd1)) # Q2.8
    plot_distributions(reip2, mxd1, size_one_bins=True) # Q2.8
