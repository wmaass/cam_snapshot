import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Data provided
data_dict = {
    'Length of Stay': [694.42980629],
    'Hospital Service Area': [23.76307419],
    'Hospital County': [-1.84296256],
    'Facility Name': [69.8761504],
    'Age Group': [3.69287662],
    'Zip Code': [-9.14553155],
    'Gender': [-5.16888285],
    'Ethnicity': [0.70433379],
    'Type of Admission': [15.26170646],
    'Patient Disposition': [-30.32696728],
    'CCS Diagnosis Description': [40.44922215],
    'CCS Procedure Description': [-274.39565663],
    'APR DRG Description': [-58.64574847],
    'APR MDC Code': [189.8865308],
    'APR MDC Description': [-74.50978778],
    'APR Severity of Illness Code': [-44.53915761],
    'APR Severity of Illness Description': [-29.98645376],
    'APR Risk of Mortality': [-9.022282],
    'APR Medical Surgical Description': [42.85098544],
    'Payment Typology 1': [-4.00755333],
    'Payment Typology 2': [-21.71527118],
    'Payment Typology 3': [1.32459113],
    'Emergency Department Indicator': [-3.63738409],
    'Operating Certificate Number': [-128.06628936],
    'Permanent Facility Id': [2.75780488],
    'CCS Diagnosis Code': [-2.33958183],
    'CCS Procedure Code': [100.09544208],
    'APR DRG Code': [22.62980062]
}

# Create DataFrame
df = pd.DataFrame(data_dict)

# Calculate the mean of each column
means = df.mean()

# Convert the means to a DataFrame
df_means = pd.DataFrame(means, columns=['Mean'])

# Apply logarithmic transformation to the means
df_means['LogMean'] = np.sign(df_means['Mean']) * np.log1p(np.abs(df_means['Mean']))

# Scale the log-transformed values
scaler = StandardScaler()
scaled_log_means = scaler.fit_transform(df_means['LogMean'].values.reshape(-1, 1))

# Apply K-Means clustering
n_clusters = 5  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_means['SHAP_clusters'] = kmeans.fit_predict(scaled_log_means)

# Print the resulting DataFrame
print(df_means)

# Plot the results with vertical annotations
plt.figure(figsize=(10, 2))
plt.scatter(df_means['LogMean'], np.zeros_like(df_means['LogMean']), c=df_means['SHAP_clusters'], cmap='viridis')
for i, txt in enumerate(df_means.index):
    plt.annotate(txt, (df_means['LogMean'][i], 0), fontsize=9, rotation=90, verticalalignment='bottom')
plt.title(f'K-Means Clustering (Log-Transformed and Scaled Data, {n_clusters} clusters)')
plt.xlabel('Log-Transformed Mean Value')
plt.yticks([])
plt.show()
