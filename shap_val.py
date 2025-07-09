import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Step 1: Read SHAP values from CSV
file_path = './results/HIS17-two-classes/shap_val.csv'
df = pd.read_csv(file_path)

# Assuming the SHAP values are in a column called 'shap_values'. Adjust accordingly.
shap_val = df['Length of Stay'].values  # Adjust column name

# Step 2: Perform Kernel Density Estimation (KDE)
density = gaussian_kde(shap_val)

# Step 3: Create a range of 1000 evenly spaced points between the min and max SHAP values
xs = np.linspace(min(shap_val), max(shap_val), 1000)

# Step 4: Evaluate the density function at these points
density_values = density(xs)

# Step 5: Find peaks in the KDE with adjustable prominence
peaks, _ = find_peaks(density_values, prominence=0.000001)  # Adjust prominence

# Step 6: Plot KDE and mark the peaks
plt.figure(figsize=(10, 6))
plt.plot(xs, density_values, label='KDE of SHAP values')
plt.plot(xs[peaks], density_values[peaks], "x", label='Peaks', color='red')  # Mark peaks with red 'x'
plt.xlim(-7000, 20000)  # Adjust the x-axis range

# Labeling the plot
plt.title('Kernel Density Estimate and Peak Detection in SHAP Values')
plt.xlabel('SHAP Value')
plt.ylabel('Density')
plt.legend()

# Show the plot
plt.show()

# Step 7: Print the peak SHAP values
print("Peak SHAP Values:")
for peak in peaks:
    print(f"SHAP value at peak: {xs[peak]}, Density: {density_values[peak]}")
