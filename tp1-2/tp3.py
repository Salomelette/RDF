from tools import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


data=np.load("15_scenes_Xy.npz", "rb")
X = data["X"]
y = data["y"]

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3)

svm = SVC()
parameters = {'kernel':('linear', 'rbf'), 'C':range(10e-15,10)}
clf = GridSearchCV(svm, parameters, cv=5)
clf.fit(X_train,y_train)


