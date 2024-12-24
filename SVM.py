import pandas as pd

prostate_cancer_df = pd.read_csv("C:/data/Prostate_Cancer.csv")

prostate_cancer_df.drop(["id"], axis=1, inplace=True)

prostate_cancer_df["diagnosis_result"] = [1 if element == "M" else 0 for element in prostate_cancer_df["diagnosis_result"]]

prostate_cancer_df["diagnosis_result"].value_counts()

x = prostate_cancer_df.drop(['diagnosis_result'], axis=1)
y = prostate_cancer_df["diagnosis_result"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, train_size=0.75, random_state=42)

clf_names = []
clf_scores = []

# Support Vector Machine (SVM) Classification
from sklearn.svm import SVC
svm = SVC(random_state=42)
svm.fit(x_train,y_train) #Fitting
print("SVM Classification Test Accuracy : {}".format(svm.score(x_test,y_test)))
clf_names.append("SVM")
clf_scores.append(svm.score(x_test,y_test))

print(clf_scores)
