import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import matplotlib.pyplot as plt

app_name = "HIS17-two-classes" 



prominence = 0.0002

# Read SHAP values from file
if os.path.exists('./results/'+app_name+'/shap_values_df.csv'):
    df = pd.read_csv('./results/'+app_name+'/shap_values_df.csv')
else:
    print("File not found")


def classify_peaks(density_values, xs, prominence):
    peaks, _ = find_peaks(density_values, prominence=prominence)
    peaks_x = xs[peaks]
    if len(peaks_x) > 1 and any(peaks_x < 0) and any(peaks_x > 0):
        return "Two Peaks", peaks_x
    return "Single Peak", peaks_x

for feature in df.columns:

    log2_x = np.sign(df[feature]) * (np.abs(df[feature] - df[feature].mean()))  # Log2 scale preserving sign
    log2_y = np.log2(np.abs(df[feature] / df[feature].mean()) + 1)
    log2_x += np.random.normal(0, 1e-6, size=log2_x.shape)  # Adding tiny noise
    volcano_df = pd.DataFrame({'Log2_X': log2_x, 'Log2_Y': log2_y})
    density = gaussian_kde(log2_x)
    xs = np.linspace(min(log2_x), max(log2_x), 1000)
    density_values = density(xs)


    # plot density estimation
    plt.plot(xs, density_values, color='black')

    # Identify peaks in the density estimation
    peaks, _ = find_peaks(density_values, prominence=prominence)

    # Mark the peaks on the plot
    plt.plot(xs[peaks], density_values[peaks], 'x', color='red')

    # Annotate the peaks with their corresponding log2_x values
    for peak in peaks:
        plt.annotate(f'{xs[peak]:.2f}', xy=(xs[peak], density_values[peak]), xytext=(xs[peak] + 0.05, density_values[peak] + 0.01))

    plt.xlabel('Log2(X - Mean(X) + 1)')
    plt.ylabel('Density')
    plt.title(f'Volcano plot for feature: {feature}')
    plt.show()

    peak_classification, peaks_x = classify_peaks(density_values, xs, prominence)

    print(peak_classification)