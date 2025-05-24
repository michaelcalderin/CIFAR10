from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import numpy as np

# Combines training and test set that was provided by the dataset loading
class DatasetCombiner(BaseEstimator, TransformerMixin):
    """
    Combines training and test set into one.
    """

    def __init__(self):
        return
    
    def fit(self, X, y=None):
        """
        Not needed (does nothing).

        # Parameters
        **X**: tuple in form *(X_train, X_test, y_train, y_test)*
        - *X_train* and *X_test* represent features with array shape *(n, 32, 32, 3)*
        - *y_train* and *y_test* represent labels with array shape *(n, 1)* \n
        Note: n is the number of samples in that array (e.g. X_train and X_test will have different values of n) \n
        **y**: (optional) not used and set to *None* by default

        # Return
        **self**: the object
        """
        return self
    
    def transform(self, X):
        """
        Combines training and test set into one.

        # Parameters
        **X**: tuple in form *(X_train, X_test, y_train, y_test)*
        - *X_train* and *X_test* represent features with array shape *(n, 32, 32, 3)*
        - *y_train* and *y_test* represent labels with array shape *(n, 1)* \n
        Note: n is the number of samples in that array (e.g. X_train and X_test will have different values of n) \n
        **y**: (optional) not used and set to *None* by default

        # Return
        **(X_total, y_total)**: combined features (X_total) and combined labels (y_total)
        """
        X_train, X_test, y_train, y_test = X
        X_total = np.concatenate((X_train, X_test), axis=0)
        y_total = np.concatenate((y_train, y_test), axis=0)
        return (X_total, y_total)
    
    def fit_transform(self, X, y=None):
        """
        Combines training and test set into one.

        # Parameters
        **X**: tuple in form *(X_train, X_test, y_train, y_test)*
        - *X_train* and *X_test* represent features with array shape *(n, 32, 32, 3)*
        - *y_train* and *y_test* represent labels with array shape *(n, 1)* \n
        Note: n is the number of samples in that array (e.g. X_train and X_test will have different values of n) \n
        **y**: (optional) not used and set to *None* by default

        # Return
        **(X_total, y_total)**: combined features (X_total) and combined labels (y_total)
        """
        self.fit(X, y)
        return self.transform(X)
    

# Reshapes combined dataset into (60000, 3072) shape for features and (60000,) for labels
class Reshaper(BaseEstimator, TransformerMixin):
    """
    Reshapes X_total and y_total.
    """

    def __init__(self):
        return
    
    def fit(self, X, y=None):
        """
        Not needed (does nothing).

        # Parameters
        **X**: tuple in form *(X_total, y_total)*
        - *X_total* represents features with array shape *(n, 32, 32, 3)*
        - *y_total* represents labels with array shape *(n, 1)* \n
        Note: n is the number of samples \n
        **y**: (optional) not used and set to *None* by default

        # Return
        **self**: the object
        """
        return self
    
    def transform(self, X):
        """
        Reshapes X_total and y_total.

        # Parameters
        **X**: tuple in form *(X_total, y_total)*
        - *X_total* represents features with array shape *(n, 32, 32, 3)*
        - *y_total* represents labels with array shape *(n, 1)* \n
        Note: n is the number of samples \n
        **y**: (optional) not used and set to *None* by default

        # Return
        **(X_total, y_total)**: *X_total* reshaped into *(n, 3072)* and *y_total* reshaped into *(n,)*
        """
        X_total, y_total = X
        X_total = X_total.reshape(-1, 32*32*3)
        y_total = y_total.flatten()
        return (X_total, y_total)
    
    def fit_transform(self, X, y=None):
        """
        Reshapes X_total and y_total.

        # Parameters
        **X**: tuple in form *(X_total, y_total)*
        - *X_total* represents features with array shape *(n, 32, 32, 3)*
        - *y_total* represents labels with array shape *(n, 1)* \n
        Note: n is the number of samples \n
        **y**: (optional) not used and set to *None* by default

        # Return
        **(X_total, y_total)**: *X_total* reshaped into *(n, 3072)* and *y_total* reshaped into *(n,)*
        """
        self.fit(X, y)
        return self.transform(X)
    

# Splits dataset into training and testing set, stratifying by the labels
class Splitter(BaseEstimator, TransformerMixin):
    """
    Splits X_total and y_total into X_train, X_test, y_train, y_test (80% training, 20% testing). Stratifies by y_total.
    """

    def __init__(self):
        return
    
    def fit(self, X, y=None):
        """
        Not needed (does nothing).

        # Parameters
        **X**: tuple in form *(X_total, y_total)*
        - *X_total* represents features with array shape *(n, 3072)*
        - *y_total* represents labels with array shape *(n,)* \n
        Note: n is the number of samples \n
        **y**: (optional) not used and set to *None* by default

        # Return
        **self**: the object
        """
        return self
    
    def transform(self, X):
        """
        Splits X_total and y_total into X_train, X_test, y_train, y_test. Stratifies by y_total.

        # Parameters
        **X**: tuple in form *(X_total, y_total)*
        - *X_total* represents features with array shape *(n, 3072)*
        - *y_total* represents labels with array shape *(n,)* \n
        Note: n is the number of samples \n
        **y**: (optional) not used and set to *None* by default

        # Return
        **(X_train, X_test, y_train, y_test)**: training/testing set features and their labels (as arrays)
        """
        X_total, y_total = X
        X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.2, random_state=42, stratify=y_total)
        return (X_train, X_test, y_train, y_test)
    
    def fit_transform(self, X, y=None):
        """
        Splits X_total and y_total into X_train, X_test, y_train, y_test. Stratifies by y_total.

        # Parameters
        **X**: tuple in form *(X_total, y_total)*
        - *X_total* represents features with array shape *(n, 3072)*
        - *y_total* represents labels with array shape *(n,)* \n
        Note: n is the number of samples \n
        **y**: (optional) not used and set to *None* by default

        # Return
        **(X_train, X_test, y_train, y_test)**: training/testing set features and their labels (as arrays)
        """
        self.fit(X, y)
        return self.transform(X)


# Scales features to [0,1] by dividing by 255
class Scaler(BaseEstimator, TransformerMixin):
    """
    Scales features to [0,1] by dividing by 255
    """

    def __init__(self):
        return
    
    def fit(self, X, y=None):
        """
        Not needed (does nothing).

        # Parameters
        **X**: tuple in form *(X_train, X_test, y_train, y_test)*
        - *X_train* and *X_test* represent features with array shape *(n, 3072)*
        - *y_train* and *y_test* represent labels with array shape *(n,)* \n
        Note: n is the number of samples in that array (e.g. X_train and X_test will have different values of n) \n
        **y**: (optional) not used and set to *None* by default

        # Return
        **self**: the object
        """
        return self
    
    def transform(self, X):
        """
        Scales features to [0,1] by dividing by 255

        # Parameters
        **X**: tuple in form *(X_train, X_test, y_train, y_test)*
        - *X_train* and *X_test* represent features with array shape *(n, 3072)*
        - *y_train* and *y_test* represent labels with array shape *(n,)* \n
        Note: n is the number of samples in that array (e.g. X_train and X_test will have different values of n) \n
        **y**: (optional) not used and set to *None* by default

        # Return
        **(X_train, X_test, y_train, y_test)**: training/testing set features and their labels (as arrays)
        """
        X_train, X_test, y_train, y_test = X
        X_train = X_train / 255
        X_test = X_test / 255
        return (X_train, X_test, y_train, y_test)
    
    def fit_transform(self, X, y=None):
        """
        Scales features to [0,1] by dividing by 255

        # Parameters
        **X**: tuple in form *(X_train, X_test, y_train, y_test)*
        - *X_train* and *X_test* represent features with array shape *(n, 3072)*
        - *y_train* and *y_test* represent labels with array shape *(n,)* \n
        Note: n is the number of samples in that array (e.g. X_train and X_test will have different values of n) \n
        **y**: (optional) not used and set to *None* by default

        # Return
        **(X_train, X_test, y_train, y_test)**: training/testing set features and their labels (as arrays)
        """
        self.fit(X, y)
        return self.transform(X)