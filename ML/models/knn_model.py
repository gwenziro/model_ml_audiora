import pickle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Example: Load Kaggle data
X_train = np.load('ML/uploads/X_train.npy')  # Feature data
y_train = np.load('ML/uploads/y_train.npy')  # Age labels

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

with open('ML/models/knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)
