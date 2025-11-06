import pandas as pd

df = pd.read_csv(r"D:\Dataset\emails.csv")
df = df.drop(columns=['Email No.'])
X = df.drop(columns=["Prediction"])
y = df["Prediction"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
result1=pd.DataFrame({
    'Actual':y_test.values,
    'Predicted':knn_pred
})
print(result1)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("\n===== KNN RESULTS =====")
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
print("\nKNN Classification Report:\n", classification_report(y_test, knn_pred))
from sklearn.svm import SVC
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
result2=pd.DataFrame({
    'Actual':y_test.values,
    'Predicted':svm_pred
})
print(result2)
print("\n===== SVM RESULTS =====")
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
print("\nSVM Classification Report:\n", classification_report(y_test, svm_pred))
