import numpy as np

base_path = '/Users/rudra/Tech/coursera_ml_specialization/advanced_learning_algorithms/week1_lab_assignment/home/jovyan/work/'

def load_data():
    X = np.load(f"{base_path}/data/X.npy")
    y = np.load(f"{base_path}/data/y.npy")
    X = X[0:1000]
    y = y[0:1000]
    return X, y

def load_weights():
    w1 = np.load(f"{base_path}/data/w1.npy")
    b1 = np.load(f"{base_path}/data/b1.npy")
    w2 = np.load(f"{base_path}/data/w2.npy")
    b2 = np.load(f"{base_path}/data/b2.npy")
    return w1, b1, w2, b2

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
