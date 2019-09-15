# LINEAR KERNEL
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
def svm(features_df):

    X = np.array(features_df.feature.tolist())
    y = np.array(features_df.class_label.tolist())
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
    svc = SVC(kernel='linear', C=1)
    svc = svc.fit(X_train, y_train)
    acc = svc.score(X_test, y_test)
    print("Accuracy",acc*100)