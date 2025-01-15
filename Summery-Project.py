import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

data = pd.read_csv("C:/data/Loans_Data.csv")

target = data['not.fully.paid']
features = data.drop('not.fully.paid', axis=1)

categorical_cols = features.select_dtypes(include='object').columns.tolist()
numeric_cols = features.select_dtypes(exclude='object').columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ]
)

features = preprocessor.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

best = 0
knnBestScore = 0

for i in range(1, 11):
    k = i
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    if accuracy > best:
        best = accuracy
        knnBestScore = k

    print(f"{i}) model accuracy: {accuracy:.2f}")

print(f"Best k: {knnBestScore}, Best accuracy: {best:.2f}")

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy for random forest:", metrics.accuracy_score(y_test, y_pred))

if best > metrics.accuracy_score(y_test, y_pred):
    print(f"KNN is better with {best:.2f} accuracy")
else:
    print(f"Random Forest is better with {metrics.accuracy_score(y_test, y_pred):.2f} accuracy")

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

best_pca = 0
knnBestScore_pca = 0

for i in range(1, 11):
    k = i
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_pca, y_train)
    predictions = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, predictions)

    if accuracy > best_pca:
        best_pca = accuracy
        knnBestScore_pca = k

    print(f"{i}) PCA model accuracy: {accuracy:.2f}")

print(f"Best k with PCA: {knnBestScore_pca}, Best PCA accuracy: {best_pca:.2f}")

kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(features)

data['cluster'] = clusters
data_with_clusters = pd.concat([data, pd.DataFrame(features, columns=preprocessor.get_feature_names_out())], axis=1)

print(data_with_clusters.head())
