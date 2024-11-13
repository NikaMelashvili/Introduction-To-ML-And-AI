import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Step 1: Create dataset as a NumPy array
X = np.array([[185, 72],
              [170, 56],
              [168, 60],
              [179, 68],
              [182, 72],
              [188, 77]])

# Step 2: Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

# Step 3: Extract cluster centers and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Step 4: Display initial data along with cluster labels in a DataFrame
df = pd.DataFrame(X, columns=['Height', 'Weight'])
df['Cluster_Label'] = labels

# Step 5: Create a DataFrame for the centroids
centroids_df = pd.DataFrame(centroids, columns=['Height', 'Weight'])

# Step 6: Print the DataFrames separately
print("\nData with Cluster Labels:")
print(df)

# Exporting DataFrame to CSV

# Export DataFrame to a CSV file
df.to_csv('C:/data/df_centroids.csv', index=False)
print("\nDataFrame has been exported to 'df_centroids.csv'.")

df_first_cluster = df[df['Cluster_Label'] == 0]
df_first_cluster.to_csv('C:/data/df_second_cluster.csv', index=False)

print("\nCentroid Coordinates (Height and Weight):")
print(centroids_df)

# Step 7: Plot the data points with their respective clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', label='Clustered Points')

# Plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=100, label='Centroids')

# Step 8: Enhance the plot with labels and a legend
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('K-Means Clustering of Height and Weight')
plt.legend()

# Step 9: Display the final plot
plt.show()

# Step 10: Reading Data from CSV

# Read the CSV file into a new DataFrame
df_from_csv = pd.read_csv('C:/data/df_centroids.csv')
print("\nDataFrame read from 'data.csv':")
print(df_from_csv)
