import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class BoltzmannClassifier(BaseEstimator, ClassifierMixin):
    """
    Boltzmann Classifier

    A simple, interpretable probabilistic classifier inspired by the Boltzmann distribution.
    Class probabilities are computed from energy-based distances between the input features
    and class centroids.

    Parameters
    ----------
    k : float, default=1.0
        Scaling factor analogous to Boltzmann's constant.

    T : float, default=1.0
        Temperature parameter controlling the softness of the probability distribution.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during `fit`.

    class_means_ : dict
        Mapping from class label to mean feature vector (centroid).
    """
    def __init__(self, k=1.0, T=1.0):
        self.k = k
        self.T = T

    def fit(self, X, y):
        """
        Compute the mean vector for each class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.

        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.classes_ = np.unique(y)
        self.class_means_ = {
            cls: X[y == cls].mean(axis=0)
            for cls in self.classes_
        }
        return self

    def _energy(self, x, mean_vector):
        """
        Compute the energy of sample x relative to a class mean using L1 norm.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
        mean_vector : ndarray of shape (n_features,)

        Returns
        -------
        float
            Energy value.
        """
        return np.sum(np.abs(x - mean_vector))

    def predict_proba(self, X):
        """
        Compute class probabilities for each input sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Probability distribution over classes for each sample.
        """
        proba = []
        for x in X:
            energies = [
                np.exp(-self._energy(x, self.class_means_[cls]) / (self.k * self.T))
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
            Predicted class labels.
        """
        probas = self.predict_proba(X)
        indices = np.argmax(probas, axis=1)
        return self.classes_[indices]
