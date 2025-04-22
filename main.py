from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

from fnn import MulticlassNeuralNetwork
from lg import MulticlassLogisticRegression
from knn import knn
from naive_bayes import MulticlassNaiveBayes
  
# fetch dataset 
steel_industry_energy_consumption = fetch_ucirepo(id=851) 
  
# data (as pandas dataframes) 
X = steel_industry_energy_consumption.data.features 
y = steel_industry_energy_consumption.data.targets 

# Data Preprocessing
X = X.drop(columns=['WeekStatus', 'Day_of_week'])
le = LabelEncoder()
y_encoded = le.fit_transform(y.values.ravel())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets 80-20
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.20, random_state=42, stratify=y)


accuracies = []
f1_scores = []

# Train the model
lr = MulticlassLogisticRegression(lr=0.01, epochs=1000)
lr.fit(X_train, y_train)

# Predict and evaluate
lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_f1_score = f1_score(y_test, lr_pred, average='weighted')
f1_scores.append(lr_f1_score)
accuracies.append(lr_accuracy)
print("Logistic Regression Accuracy:", lr_accuracy)
print("Logistic Regression F1 Score:", lr_f1_score)

# Train the kNN model
knn = knn(k=3)
knn.fit(X_train, y_train)

# Predict and evaluate
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_f1_score = f1_score(y_test, knn_pred, average='weighted')
f1_scores.append(knn_f1_score)
accuracies.append(knn_accuracy)
print("kNN Accuracy:", knn_accuracy)
print("kNN F1 Score:", knn_f1_score)



# Train Naive Bayes
nb = MulticlassNaiveBayes()
nb.fit(X_train, y_train)

# Predict and evaluate
nb_pred = nb.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
nb_f1_score = f1_score(y_test, nb_pred, average='weighted')
f1_scores.append(nb_f1_score)
accuracies.append(nb_accuracy)
print("Multiclass Naive Bayes Accuracy:", nb_accuracy)
print("Multiclass Naive Bayes F1 Score:", nb_f1_score)


# Train the nn model
nn = MulticlassNeuralNetwork(input_size=X_train.shape[1], hidden_size=16, output_size=3, epochs=1000)
nn.fit(X_train, y_train)

# Predict and evaluate
nn_pred = nn.predict(X_test)
nn_accuracy = accuracy_score(y_test, nn_pred)
nn_f1_score = f1_score(y_test, nn_pred, average='weighted')
f1_scores.append(nn_f1_score)
accuracies.append(nn_accuracy)
print("Neural Network Accuracy:", nn_accuracy)
print("Neural Network F1 Score:", nn_f1_score)


model_names = ['Logistic Regression', 'KNN', 'Naive Bayes', 'Neural Network']


# Accuracy plot
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies)
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Accuracy is between 0 and 1
plt.grid(axis='y')

# Save the plot as a PNG file
plt.savefig('model_accuracy_comparison.png')

# Show the plot
plt.show()



# F1 Plot
plt.figure(figsize=(10, 6))
plt.bar(model_names, f1_scores)
plt.title('Model F1 Comparison')
plt.xlabel('Model')
plt.ylabel('F1 score')
plt.grid(axis='y')

# Save the plot as a PNG file
plt.savefig('model_f1_comparison.png')

# Show the plot
plt.show()