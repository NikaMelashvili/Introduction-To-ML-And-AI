# Import necessary libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the iris dataset
dataset = pd.read_csv("C:/data/iris.csv")

# Define features and target variable
X = dataset.drop('Species', axis=1)  # Features
y = dataset['Species']  # Target variable
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a K-NN model
k = 3  # Choosing the number of neighbors
model = KNeighborsClassifier(n_neighbors=k)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')