import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def linearHypothesis(X, Theta):
    h = np.dot(X, Theta)
    return h


def linearCost(X, Theta, Y,m):
    temp = linearHypothesis(X, Theta) - Y
    squares = np.square(temp)
    J = np.sum(squares) / (2 * m)
    MSE = np.sum(squares) / m
    return J, MSE


def linearGradientDescent(initialTheta, X, Y,alpha,numOfIterations,m):
    costInIterations = []
    MSEInIterations = []
    Theta = initialTheta
    for i in range(numOfIterations):
        cost, mse = linearCost(X, Theta, Y,m)
        costInIterations.append(cost)
        MSEInIterations.append(mse)
        print('Cost in iteration ', i, ' = ', costInIterations[i])
        print('MSE in iteration ', i, ' = ', MSEInIterations[i])
        num1 = linearHypothesis(X, Theta)
        s = np.multiply(num1 - Y, X)
        tTheta = np.zeros((len(X[0]), 1))
        for j in range(len(X[0])):
            for k in range(len(X)):
                tTheta[j][0] += s[k][j]
        num = alpha / m
        tempTheta = (Theta - (num * tTheta))
        Theta = tempTheta
    print(Theta)
    return Theta, costInIterations


def runLinearMultiValues():
    dataFile = pd.read_csv("house_data.csv", index_col=0)
    size = 21613
    numOfIterations = 1000
    x = dataFile[["grade", "bathrooms", "lat", "sqft_living", "view"]]
    y = dataFile[["price"]]
    Y = np.array(y[:size])
    XNormalized = (x - x.mean()) / x.std()
    X = XNormalized
    X = np.c_[np.ones(len(X)), X]

    Theta = np.zeros((len(X[0]), 1))
    m = len(Y)

    Th, costIterations = linearGradientDescent(Theta, X, Y, 0.001, numOfIterations, m)
    plt.plot(np.arange(1000), costIterations)
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.show()
    print("---------------------------------------------")
    Th, costIterations = linearGradientDescent(Theta, X, Y, 0.01, numOfIterations, m)
    plt.plot(np.arange(1000), costIterations)
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.show()
    print("---------------------------------------------")
    Th, costIterations = linearGradientDescent(Theta, X, Y, 0.1, numOfIterations, m)
    plt.plot(np.arange(1000), costIterations)
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.show()
    print("---------------------------------------------")

