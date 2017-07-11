import random
import pickle
import scipy
import numpy
import time
import math
import csv

reg_lambda = 0.01
learning_rate = 0.01
load_model_from_file = True


def read_train(filename):
    X, y = [], []

    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader, None)
        for row in reader:
            y.append(row[0])
            X.append(row[1:])

    y = list(map(int, y))

    for i, row in enumerate(X):
        X[i] = list(map(int, row))
    X = numpy.array(X)

    tmp = numpy.zeros((X.shape[0], 10))
    for i, ans in enumerate(y):
        tmp[i][ans] = 1
    y = tmp

    print("reading finished")
    return X, y


def read_test(filename):
    X = []

    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader, None)
        for row in reader:
            X.append(row)

    for i, row in enumerate(X):
        X[i] = list(map(int, row))
    X = numpy.array(X)

    print("reading finished")
    return X


def unison_shuffle(a, b):
    rnd_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rnd_state)
    numpy.random.shuffle(b)
    return a, b

def sigmoid (x):
    return 1/(1 + numpy.exp(-x))


def derivatives_sigmoid(x):
    return x * (1 - x)


def cost_function(model, X, y):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    m = X.shape[0]

    a1 = X
    z2 = a1.dot(W1) + numpy.tile(b1, [m, 1])
    a2 = sigmoid(z2)
    z3 = a2.dot(W2) + numpy.tile(b2, [m, 1])
    a3 = sigmoid(z3)
    hypothesis = a3

    return numpy.sum(numpy.square(hypothesis - y))


def predict(model, X):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    m = X.shape[0]

    a1 = X
    z2 = a1.dot(W1) + numpy.tile(b1, [m, 1])
    a2 = sigmoid(z2)
    z3 = a2.dot(W2) + numpy.tile(b2, [m, 1])
    a3 = sigmoid(z3)
    hypothesis = a3

    return numpy.argmax(hypothesis, axis=1)


def build_model(X, y, epochs=5000, print_cost=False):
    numpy.random.seed(int(time.time()))

    k = 10
    m, n = X.shape
    h = int(m / 2 / (n + k))

    if load_model_from_file:
        model = pickle.load(open("nn.b", "rb"))
    else:
        W1 = numpy.random.randn(n, h) / numpy.sqrt(n)
        b1 = numpy.zeros((1, h))
        W2 = numpy.random.randn(h, k) / numpy.sqrt(h)
        b2 = numpy.zeros((1, k))

        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    for i in range(epochs):
        print("iteration={}".format(i))
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

        a1 = X # m*n
        z2 = a1.dot(W1) + numpy.tile(b1, [m, 1]) # m*h
        a2 = sigmoid(z2) # m*h
        z3 = a2.dot(W2) + numpy.tile(b2, [m, 1]) # m*k
        a3 = sigmoid(z3) # m*k
        hypothesis = a3 # m*k

        delta3 = hypothesis - y # m*k б
        delta2 = numpy.dot(delta3, W2.T) * derivatives_sigmoid(a2)  #m*k*k*h *

        dW2 = 1 / m * numpy.dot(a2.T, delta3) # h*k triangle
        db2 = 1 / m * numpy.sum(delta3, axis=0).reshape(1, k)

        dW1 = 1 / m * numpy.dot(a1.T, delta2)
        db1 = 1 / m * numpy.sum(delta2, axis=0).reshape(1, h)

        dW2 = dW2 + reg_lambda / m * W2
        dW1 = dW1 + reg_lambda / m * W1

        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if print_cost and i % 100 == 0:
            print("iteration={} cost={}".format(i, cost_function(model, X, y)))

    return model


def cost_function(model, X, y):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    m = X.shape[0]

    a1 = X  # m*n
    z2 = a1.dot(W1) + numpy.tile(b1, [m, 1])  # m*h
    a2 = sigmoid(z2)  # m*h
    z3 = a2.dot(W2) + numpy.tile(b2, [m, 1])  # m*k
    a3 = sigmoid(z3)  # m*k
    hypothesis = a3  # m*k

    error = numpy.sum(numpy.square(hypothesis - y))  # m*k
    return error


def build_model_with_gradient_descent(model, X, y):
    numpy.random.seed(int(time.time()))

    k = 10
    m, n = train_X.shape
    h = int(m / 2 / (n + k))

    W1 = numpy.random.randn(n, h) / numpy.sqrt(n)
    b1 = numpy.zeros((1, h))
    W2 = numpy.random.randn(h, k) / numpy.sqrt(h)
    b2 = numpy.zeros((1, k))

    unrolled_parameters = numpy.concatenate((W1.ravel(),  b1.ravel(), W2.ravel(), b2.ravel()), axis=0)
    scipy.optimize.fmin_cg(cost_function, unrolled_parameters, fprime=cost_function_derivative, args=(X, y))


def test():
    model = pickle.load(open("nn.b", "rb"))
    test_data = read_test("test.csv")
    prediction = predict(model, test_data)

    output = open("results.csv", "w")
    output.write("ImageId, Label\n")
    for i in range(len(prediction)):
        output.write("{},{}\n".format(i + 1, prediction[i]))
    output.close()



def main():
    X, y = read_train("train.csv")
    unison_shuffle(X, y)

    train_size = int(X.shape[0] * 0.7)
    train_X, test_X = X[:train_size], X[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]
    test_y = numpy.argmax(test_y, axis=1)

    model = build_model(X, y, epochs=100, print_cost=True)
    hypothesis = predict(model, test_X)

    accuracy = numpy.sum(numpy.equal(hypothesis, test_y)) / test_y.shape[0]
    print("Accuracy={}".format(accuracy))

    pickle.dump(model, open("nn.b", "wb"))


if __name__ == "__main__":
    main()