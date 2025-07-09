import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Data provided
data_dict = {
    'Feature': [
        'Length of Stay', 'Hospital Service Area', 'Hospital County', 'Facility Name', 'Age Group',
        'Zip Code', 'Gender', 'Ethnicity', 'Type of Admission', 'Patient Disposition',
        'CCS Diagnosis Description', 'CCS Procedure Description', 'APR DRG Description', 'APR MDC Code',
        'APR MDC Description', 'APR Severity of Illness Code', 'APR Severity of Illness Description',
        'APR Risk of Mortality', 'APR Medical Surgical Description', 'Payment Typology 1',
        'Payment Typology 2', 'Payment Typology 3', 'Emergency Department Indicator', 
        'Operating Certificate Number', 'Permanent Facility Id', 'CCS Diagnosis Code', 
        'CCS Procedure Code', 'APR DRG Code'
    ],
    'Value': [
        694.42980629, 23.76307419, -1.84296256, 69.8761504, 3.69287662, -9.14553155, -5.16888285, 
        0.70433379, 15.26170646, -30.32696728, 40.44922215, -274.39565663, -58.64574847, 
        189.8865308, -74.50978778, -44.53915761, -29.98645376, -9.022282, 42.85098544, 
        -4.00755333, -21.71527118, 1.32459113, -3.63738409, -128.06628936, 2.75780488, 
        -2.33958183, 100.09544208, 22.62980062
    ]
}

# Create DataFrame
df = pd.DataFrame(data_dict)

# Apply logarithmic transformation to the absolute values (adding 1 to avoid log(0))
df['LogValue'] = np.sign(df['Value']) * np.log1p(np.abs(df['Value']))

# Convert the log-transformed values to a 2D array for clustering
log_data_values = df['LogValue'].values.reshape(-1, 1)

# Apply DBSCAN with an appropriate eps for log-transformed data
eps = 1  # Adjust this parameter based on your data distribution
min_samples = 1  # Minimum number of points to form a cluster
dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(log_data_values)
labels = dbscan.labels_

# Add cluster labels to the DataFrame
df['Cluster'] = labels

# Plot the results with vertical annotations
plt.figure(figsize=(10, 2))
plt.scatter(df['LogValue'], np.zeros_like(df['LogValue']), c=df['Cluster'], cmap='viridis')
for i, txt in enumerate(df['Feature']):
    plt.annotate(txt, (df['LogValue'][i], 0), fontsize=9, rotation=90, verticalalignment='bottom')
plt.title('DBSCAN Clustering on Log-Transformed Data')
plt.xlabel('Log-Transformed Value')
plt.yticks([])
plt.show()

# Print the resulting clusters
print(df)
