import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

training_data = pd.read_csv('storepurchasedata.csv')

training_data.describe()
X = training_data.iloc[:, :-1].values
y = training_data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# minkowski is for euclidean distance
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

# Model training
classifier.fit(X_train, y_train)

y_prediction = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_prediction)

print(accuracy_score(y_test, y_prediction))

print(classification_report(y_test, y_prediction))

new_prediction = classifier.predict(sc.transform(np.array([[40, 20000]])))

new_prediction_proba = classifier.predict_proba(sc.transform(np.array([[40, 20000]])))[:, 1]

new_pred = classifier.predict(sc.transform(np.array([[42, 50000]])))

new_pred_proba = classifier.predict_proba(sc.transform(np.array([[42, 50000]])))[:, 1]

# Picking the Model and Standard Scaler


model_file = "classifier.pickle"

pickle.dump(classifier, open(model_file, 'wb'))

scalar_file = "sc.pickle"

pickle.dump(sc, open(scalar_file, 'wb'))
