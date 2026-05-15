import csv
import math
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        for row in csv.DictReader(f):
            x.append(float(row['x']))
            y.append(float(row['y']))
    return np.array(x), np.array(y)

def kernel(point, xmat, tau):
    m      = xmat.shape[0]
    weights = np.asmatrix(np.eye(m))
    for j in range(m):
        diff          = point - xmat[j]
        weights[j, j] = np.exp((diff * diff.T).item() / (-2.0 * tau**2))
    return weights

def local_weight(point, xmat, ymat, tau):
    W     = kernel(point, xmat, tau)
    theta = (xmat.T * (W * xmat)).I * (xmat.T * (W * ymat.T))
    return (point * theta).item()

def locally_weighted_regression(xmat, ymat, tau):
    m           = xmat.shape[0]
    predictions = np.zeros(m)
    for i in range(m):
        predictions[i] = local_weight(xmat[i], xmat, ymat, tau)
    return predictions

if __name__ == "__main__":
    x, y  = load_data('training_data_set_for_ex_9.csv')
    xmat  = np.asmatrix(np.column_stack((np.ones(len(x)), x)))
    ymat  = np.asmatrix(y)
    tau   = 0.1

    predictions = locally_weighted_regression(xmat, ymat, tau)

    idx  = x.argsort()
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Data Points')
    plt.plot(x[idx], predictions[idx], color='red', linewidth=2, label=f'LWR Fit (tau={tau})')
    plt.title('Locally Weighted Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

# OUTPUT: Plot showing blue data points with red LWR curve fitting them closely
