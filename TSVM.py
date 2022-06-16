import numpy as np
from sklearn.svm import SVC
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels


class TSVM(BaseEstimator, ClassifierMixin):
    """ The reason I decided to create my own estimator using scikit-learn is that
    I want to use GridSearchCV to easily search for best parameters(C, Cl, Cu).
    Parameters
    ----------
    C: the parameter for initial svm
    Cl: the weight of data with labels
    Cu: the weight of data without labels
    Attributes
    ----------
    X1 : ndarray, shape (n1_samples, m_features)
        The input passed during :meth:`fit`.
    Y1 : ndarray, shape (n1_samples,)
        The labels passed during :meth:`fit`.
    X2 : ndarray, shape (n2_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    def __init__(self, kernel="linear", C=1.0, Cl=1.0, Cu=0.001):
        self.kernel = kernel
        self.dataset = None
        self.C, self.Cl, self.Cu = C, Cl, Cu
        self.clf = SVC(C=C, kernel=self.kernel, random_state=0)

    def fit(self, X1, Y1, X2):
        """
        Train TSVM by X1, Y1, X2
        Parameters
        ----------
        X1: Input data with labels
                np.array, shape:[n1, m], n1: numbers of samples with labels, m: numbers of features
        Y1: labels of X1
                np.array, shape:[n1, ], n1: numbers of samples with labels
        X2: Input data without labels
                np.array, shape:[n2, m], n2: numbers of samples without labels, m: numbers of features
        """
        # Store the classes seen during fit
        # self.classes_ = unique_labels(Y1)
        N = len(X1) + len(X2)
        sample_weight = np.ones(N)
        sample_weight[len(X1):] = self.Cu

        self.clf.fit(X1, Y1)
        Y2 = self.clf.predict(X2)
        X2_id = np.arange(len(X2))
        X3 = np.vstack([X1, X2])
        Y3 = np.hstack([Y1, Y2])

        while self.Cu < self.Cl:
            self.clf.fit(X3, Y3, sample_weight=sample_weight)
            while True:
                Y2_d = self.clf.decision_function(X2)  # linear: w^Tx + b
                # Y2 = Y2.reshape(-1)
                epsilon = 1 - Y2 * Y2_d  # calculate function margin
                positive_set, positive_id = epsilon[Y2 > 0], X2_id[Y2 > 0]
                negative_set, negative_id = epsilon[Y2 < 0], X2_id[Y2 < 0]
                positive_max_id = positive_id[np.argmax(positive_set)]
                negative_max_id = negative_id[np.argmax(negative_set)]
                a, b = epsilon[positive_max_id], epsilon[negative_max_id]
                if a > 0 and b > 0 and a + b > 2.0:
                    Y2[positive_max_id] = Y2[positive_max_id] * -1
                    Y2[negative_max_id] = Y2[negative_max_id] * -1
                    Y3 = np.hstack([Y1, Y2])
                    self.clf.fit(X3, Y3, sample_weight=sample_weight)
                else:
                    break
            self.Cu = min(2 * self.Cu, self.Cl)
            sample_weight[len(X1):] = self.Cu
        return self

    def predict(self, X):
        """
        Feed X and predict Y by TSVM
        """
        return self.clf.predict(X)

    def score(self, X, Y):
        """
        Calculate accuracy of TSVM by X, Y
        Parameters
        ----------
        X: Input data
                np.array, shape:[n, m], n: numbers of samples, m: numbers of features
        Y: labels of X
                np.array, shape:[n, ], n: numbers of samples
        Returns
        -------
        Accuracy of TSVM
                float
        """
        return self.clf.score(X, Y)

    def predict_proba(self, X):
        """
        Feed X and predict Y by TSVM
        """
        return self.clf.predict_proba(X)

    def save(self, path='./TSVM.model'):
        """
        Save TSVM to model_path
        """
        joblib.dump(self.clf, path)

    def decision_function(self, X):
        """
        Feed X and predict Y by TSVM
        """
        return self.clf.decision_function(X)

    def classification_report(self, X):
        return self.clf.classification_report(X)
