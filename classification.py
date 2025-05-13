from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\nNaïve Bayes:")

nb = GaussianNB()

nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_nb)}")

print(confusion_matrix(y_test, y_pred_nb))

print(classification_report(y_test, y_pred_nb))