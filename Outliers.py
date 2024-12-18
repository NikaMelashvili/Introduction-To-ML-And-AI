import pandas as pd
from sklearn.cluster import DBSCAN

data = {
    'x_coordinate': [1.2, 2.1, 3.8, 8.5, 8.7, 10.0, 1.5, 3.0, 9.0],
    'y_coordinate': [2.3, 3.2, 4.1, 8.0, 8.5, 10.2, 2.6, 4.0, 8.9]
}

df = pd.DataFrame(data)
dbscan = DBSCAN(eps=1.0, min_samples=2)
df['cluster'] = dbscan.fit_predict(df[['x_coordinate', 'y_coordinate']])
outliers = df[df['cluster'] == -1]
print("Outlier is:")
print(outliers)