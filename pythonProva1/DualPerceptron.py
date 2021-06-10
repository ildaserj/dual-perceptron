# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settinimg

import numpy as np
import dataset
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



# path = r'C:\Users\serja\PycharmProjects\pythonProva1\dataset\qsar_oral_toxicity.csv'

########################  DATASET

def dataset_(dcName, delimiter):

     if dcName == 'qsar_oral_toxicity':
        # delimiter ';'
        df = pd.read_csv(r'dataset\qsar_oral_toxicity.csv', delimiter)
        cleanup_nums = {"positive": 1, "negative": -1}
        df.replace(cleanup_nums, inplace=True)
        df = shuffle(df)
        return df

     elif dcName == 'biodeg':
        # delimiter ';'
        df = pd.read_csv(r'dataset\biodeg.csv', delimiter)
        cleanup_nums = {"RB": 1, "NRB": -1}
        df.replace(cleanup_nums, inplace=True)
        df = shuffle(df)
        return df

     elif dcName == 'winequality-red':
        # delimiter ';'
        df = pd.read_csv(r'dataset\winequality-red.csv', delimiter)
        df.iloc[:, len(df.columns) - 1].replace({range(0, 6): -1, range(6, 10): 1}, inplace=True)
        df = shuffle(df)
        return df
     else:
        print('Error: Dataset not found')


def splitDataSet(df):
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]

    X = pd.DataFrame(StandardScaler().fit_transform(X.values))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
    return X_train, X_val, X_test, y_train, y_val, y_test


def kernel_(x, y, kernel):
    op = 0
    if kernel == 'linear_kernel':
        op = np.dot(x, y)
    elif kernel == 'poly_kernel':
        op = (np.dot(x, y)+1) ** 5
    elif kernel == 'RBF_kernel':
        gamma = 1.0
        op = np.exp(-gamma * np.linalg.norm(x - y) ** 2)
    return op


def accurrency(y_true, y_pred):
    accurency = np.sum(y_true == y_pred)/float(len(y_true)) * 100
    count = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            count += 1
    accuracy = count / float(len(y_true)) * 100
    #print("y test len: ", len(y_true), "y_true==y_pred: ", accuracy)
    return accuracy

def accuracy_matrix (dim):
    accuracy = np.zeros(dim)
    for i in range(dim):
        accuracy[i] = -1
    return accuracy



class DualPerceptron:
    def __init__(self, nameDataset, delimiter, kernel):
        self.nameDataset = nameDataset
        self.dataset = dataset_(nameDataset, delimiter)
        self.X_train, self.X_val, self.x_test, self.Y_train, self.Y_val, self.y_test = splitDataSet(self.dataset)
        self.X_train = self.X_train.to_numpy()
        self.Y_train = self.Y_train.to_numpy()
        self.y_test = self.y_test.to_numpy()
        self.X_val = self.X_val.to_numpy()
        self.Y_val = self.Y_val.to_numpy()
        self.epoch = 5
        self.alfa = np.zeros(len(self.X_train))
        self.bias = 0
        self.kernel = kernel
        #self.activation_func = self.unit_step_func
        self.R = None
        self.gramMat = None
        self.predictACC = None


    def fit(self):

        self.R = self.calculate_R()

        ### gramMatrix
        print('gram matrix init')
        x_count = len(self.X_train)
        gramM = np.zeros((x_count, x_count))  ### np.zeros(dimX, dimX)
        for i in range(len(self.X_train)):
            for j in range(len(self.X_train)):
                gramM[i][j] = kernel_(self.X_train[i], self.X_train[j], self.kernel)
        print("gram matrix finish")
        self.gramMat = gramM
        ###

        print("inizio apprendimento")
        accuracy = accuracy_matrix(self.epoch)
        old_alfa = self.alfa
        old_bias = self.bias
        epoch_ = 0
        for _ in range(self.epoch):
            epoch_ += 1
            for i in range(len(self.X_train)):
                correct = 0
                sum = 0
                n_err = 0

                for j in range(len(self.X_train)):
                    sum += (self.alfa[j] * self.Y_train[j] * self.gramMat[j][i])
                    if (self.Y_train[i] * (sum + self.bias)) <= 0:
                        self.alfa[i] += 1
                        self.bias += (self.Y_train[i] * (self.R ** 2))
                        n_err += 1

                    else:
                        correct += 1

            print("error val_test epoch", epoch_, "length:", "len", len(self.X_val),  "--->", self.test())
            accuracy[_] = self.test()
            x = correct/float(len(self.X_train)) * 100
            print("# # # correct:", x, "# # # error fit:", (1 - x / 100), len(self.X_train))
            if n_err == 0 or accuracy[_] < accuracy[_-1]:
                self.alfa = old_alfa
                self.bias = old_bias
                break

        print(accuracy)
        print("fine apprendimento")
        print("epoca", epoch_)
        return self.alfa, self.bias


    def test(self):

        test = np.zeros(len(self.X_val))
        error = 0
        for i in range(len(self.X_val)):
            s = 0
            for j in range(len(self.X_val)):
                s += (self.alfa[j] * self.Y_val[j] * kernel_(self.X_train[j], self.X_val[i], self.kernel))
                if (s + self.bias) >= 0:
                    test[i] = 1
                else:
                    test[i] = -1
        accuracy = accurrency(self.Y_val, test)
        error = 1 - accuracy / 100
        print(error, accuracy)
        return accuracy


    def predict(self):
        print("inizio predizione")
        self.x_test = self.x_test.to_numpy()
        prediction = np.zeros(len(self.x_test))

        for i in range(len(self.x_test)):
            sum = 0
            for j in range(len(self.X_train)):
                sum += (self.alfa[j] * self.Y_train[j] * kernel_(self.X_train[j], self.x_test[i], self.kernel))
            if(sum + self.bias) >= 0:
                prediction[i] = 1
            else:
                prediction[i] = -1

        predictedAccuracy = accurrency(self.y_test, prediction)
        print("fine predizione")
        print("Dual perceptron accuracy for", self.nameDataset, "dataset with", self.kernel, "is:", predictedAccuracy)
        print("#############################################")
        #print("y test: ", self.y_test)
        #print("prediction:", prediction)
        self.predictACC = predictedAccuracy
        return self.alfa, self.bias

    def getPredictAccuracy(self):
        return self.predictACC

    def unit_step_func(self, X):
        return np.where(X >= 0, 1, -1)

    def calculate_R(self):
        maxN = 0
        for i in range(len(self.X_train)):
            if np.linalg.norm(self.X_train[i]) > maxN:
                maxN = np.linalg.norm(self.X_train, 1)
        return maxN




