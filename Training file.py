import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

data = pd.read_csv('Fraud.csv')
downsampled_data = pd.read_csv('Fraud.csv')
upsampled_data = pd.read_csv('Fraud.csv')

features = data.drop(['isFraud', 'nameDest', 'nameOrig','isFlaggedFraud','step'], axis=1)
etype = le.fit_transform(features['type'])
features.drop('type', axis=1, inplace=True)  # drop this column and add the encoded version
features['type'] = etype
target = data['isFraud']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Create your model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model accuracy using various metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the model accuracy metrics
print('Before Down-sampling')
print('Accuracy:', accuracy * 100, "%")
print('Precision:', precision * 100, "%")
print('Recall:', recall * 100, "%")
print('F1-score:', f1 * 100, "%")
print()

### Down-sampling your Data to Avoid Over Fitting
df_majority_notfraud = data[data["isFraud"] == 0]
df_minority_fraud = data[data["isFraud"] == 1]

df_majority_downsampled = resample(df_majority_notfraud,
                                   replace=False,  # sample without replacement
                                   n_samples=8213,  # to match minority class
                                   random_state=123)  # reproducible results

# Combine minority class with downsampled majority class
downsampled_data = pd.concat([df_majority_downsampled, df_minority_fraud])

features1 = downsampled_data.drop(['isFraud','nameDest','nameOrig','isFlaggedFraud','step'],axis=1)
etype1 = le.fit_transform(features1['type'])
features1.drop('type',axis=1,inplace=True) # drop this column and add the encoded version
features1['type'] = etype1
target1 = downsampled_data['isFraud']

X_train1, X_test1, y_train1, y_test1 = train_test_split(features1, target1, test_size=0.3, random_state=42)
# Create your model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train1, y_train1)

# Make predictions on the testing data
y_pred1 = model.predict(X_test1)

# Evaluate the model accuracy using various metrics
accuracy = accuracy_score(y_test1, y_pred1)
precision = precision_score(y_test1, y_pred1)
recall = recall_score(y_test1, y_pred1)
f1 = f1_score(y_test1, y_pred1)

# Print the model accuracy metrics
print('After Down-sampling')
print('Accuracy:', accuracy *100, "%")
print('Precision:', precision *100, "%")
print('Recall:', recall *100, "%")
print('F1-score:', f1 *100, "%")

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# generate confusion matrix
cm1 = confusion_matrix(y_test1, y_pred1)

# plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm1, cmap='Blues')
ax.grid(False)
ax.set_xlabel('Predicted labels', fontsize=12, color='black')
ax.set_ylabel('True labels', fontsize=12, color='black')
ax.set_xticks(range(2))
ax.set_yticks(range(2))
ax.set_xticklabels(['Not Fraud', 'Fraud'], fontsize=10, color='black')
ax.set_yticklabels(['Not Fraud', 'Fraud'], fontsize=10, color='black')
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)

# add annotations
thresh = cm1.max() / 2
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm1[i, j]}', ha='center', va='center', color='white' if cm1[i, j] > thresh else 'black')


plt.show()

# save the model to disk
joblib.dump(model, 'fraud.sav')
