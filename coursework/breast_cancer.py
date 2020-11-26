#Load models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split, 


def logistic_regression(datatouse):
    X = data.iloc[:,1:].values
    Y = data.iloc[1:,0].values
    #Scale X values to remove mean and improve accuracy
    standard_scaler = StandardScaler()
    X_scaled = standard_scaler.fit_transform(X)
    #Defining training and testing variables
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=0)
    #Training the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    #Printing results
    print("Logistic Regression:\n")
    print("Model training accuracy: ", model.score(X_train, y_train))
    print("Model testing accuracy: ", model.score(X_test, y_test))
    print("Model accuracy: ", model.score(y_test_pred, y_test))