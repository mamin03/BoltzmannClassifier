import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import pairwise_distances

class BoltzmannClassifier(BaseEstimator, ClassifierMixin):
    """
    Boltzmann Classifier with energy based on distance to nearest neighbors in each class.

    Parameters
    ----------
    k : float, default=1.0
        Scaling factor analogous to Boltzmann's constant.

    T : float, default=1.0
        Temperature parameter controlling the softness of the probability distribution.

    n_neighbors : int, default=5
        Number of nearest neighbors to use for energy computation.
    """
    def __init__(self, k=1.0, T=1.0, n_neighbors=5):
        self.k = k
        self.T = T
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
        Store training samples by class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self : object
        """
        self.classes_ = np.unique(y)
        self.class_samples_ = {
            cls: X[y == cls] for cls in self.classes_
        }
        return self

    def _energy(self, x, class_samples):
        """
        Compute the energy of sample x based on its distance to n nearest neighbors
        in the given class.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
        class_samples : ndarray of shape (n_samples_in_class, n_features)

        Returns
        -------
        float
            Energy value
        """
        distances = np.linalg.norm(class_samples - x, axis=1)
        nearest = np.sort(distances)[:self.n_neighbors]
        return np.mean(nearest)

    def predict_proba(self, X):
        """
        Compute class probabilities for each input sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        proba = []
        for x in X:
            energies = [
                np.exp(-self._energy(x, self.class_samples_[cls]) / (self.k * self.T))
                for cls in self.classes_
            ]
            total = np.sum(energies)
            proba.append([e / total for e in energies])
        return np.array(proba)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        probas = self.predict_proba(X)
        indices = np.argmax(probas, axis=1)
        return self.classes_[indices]

class BoltzmannRegressor:
    def __init__(self, n_neighbors=5, temperature=1.0):
        self.n_neighbors = n_neighbors
        self.temperature = temperature

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y).ravel()  # Ensure 1D numpy array

    def predict(self, X):
        X = np.asarray(X)
        preds = []
        for x in X:
            # Compute distances to *all* training samples
            #print(x)
            dists = pairwise_distances([x], self.X_train)[0]
            
            # Get indices of closest samples by POSITION
            neighbor_pos = np.argsort(dists)[:self.n_neighbors]

            # Fetch distances and y-values using positional indexing
            neighbor_dists = dists[neighbor_pos]
            neighbor_targets = self.y_train[neighbor_pos]

            # Apply Boltzmann weighting
            weights = np.exp(-neighbor_dists / self.temperature)
            pred = np.sum(weights * neighbor_targets) / np.sum(weights)
            preds.append(pred)
        return np.array(preds)

# Example
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42)

# Use the custom BoltzmannClassifier
clf = BoltzmannClassifier(k=1.0, T=1.0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

# Evaluate precision, recall, F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")
