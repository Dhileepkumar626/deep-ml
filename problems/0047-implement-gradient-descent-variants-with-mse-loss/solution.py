import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    m,n = X.shape
    for _ in range(n_epochs):
        if method == 'batch':
            err = X @ weights - y
            gd = (2/m) * (X.T@err)
            weights -=  learning_rate * gd
        elif method == 'stochastic':
            for i in range(m):
                xb = X[i:i+1]
                xy = y[i:i+1]
                k = len(xb)

                err = xb @ weights - xy
                gd = (2/k) * (xb.T @ err)
                weights -= learning_rate * gd
        elif method == 'mini_batch':
            for i in range(0,m, batch_size):
                j = i+batch_size

                xb = X[i:j]
                xy = y[i:j]
                k = len(xb)
                err = xb @ weights - xy
                gd = (2/k) * (xb.T @ err)
                weights -= learning_rate * gd
        else:
            raise ValueError(
                "method must be 'batch', 'stochastic', or 'mini_batch'"
            )
    return weights
    pass
