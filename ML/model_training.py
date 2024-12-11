import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from config import label_mapping

def train_knn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Definisikan parameter grid
    param_grid = {
    'n_neighbors': list(range(13, 31, 2)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'cosine']
    }

    # Inisialisasi model KNN
    knn = KNeighborsClassifier()

    # Grid Search dengan Cross-Validation
    grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Model terbaik
    knn_best = grid_search.best_estimator_

    # Evaluasi model pada data testing
    y_pred_test = knn_best.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print(f"Akurasi pada data testing: {accuracy_test * 100:.2f}%")

    # Evaluasi model pada data training
    y_pred_train = knn_best.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print(f"Akurasi pada data training: {accuracy_train * 100:.2f}%")

    print(f"Best Parameters: {grid_search.best_params_}")

    # Validasi silang dengan 5-fold
    scores = cross_val_score(knn_best, X_train, y_train, cv=5)
    print(f"Cross-Validation Accuracy: {np.mean(scores)*100:.2f}%")

    # Classification Report untuk data testing
    target_names = [k for k, v in sorted(label_mapping.items(), key=lambda item: item[1])]
    print("Classification Report (Data Testing):")
    print(classification_report(y_test, y_pred_test, target_names=target_names))

    # Classification Report untuk data training (opsional)
    print("Classification Report (Data Training):")
    print(classification_report(y_train, y_pred_train, target_names=target_names))

    # Menyimpan model
    joblib.dump(knn_best, 'knn_model.pkl')
    print("Model KNN telah disimpan.")

    return knn_best