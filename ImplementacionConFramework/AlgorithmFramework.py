#Logistic Regresion
#Marcela Ibarra A01231973

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import learning_curve

#load data
columns = ["class","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids",
"Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"]
df = pd.read_csv('wine.data',names=columns)
print("Loading wine.data")
#clean data
df_clean = df.drop(["Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids",
"Nonflavanoid phenols","Proanthocyanins","Hue","OD280/OD315 of diluted wines","Proline"], axis=1)
x = df_clean[["Alcohol","Color intensity"]].to_numpy()
y = df_clean["class"]
print("Cleaning wine.data")
print("shape y: ",y.shape)
print("shape x: ",x.shape)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y,random_state=1)
print("shape y_train: ",y_train.shape)
print("shape X_train: ",X_train.shape)
print("shape y_test: ",y_train.shape)
print("shape x_test: ",X_train.shape)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(fit_intercept=True)
print("Training the model")
model.fit(X_train,y_train)

print("Testing the model")
y_model = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy_score: ",accuracy_score(y_test,y_model))

a = [[12,8.3],[13,5.32],[15,4.5]]
print("For new predictions: ", a)

new_pred = model.predict(a)
print("The results of new predictions: ",new_pred)