import DualPerceptron
import numpy as np
import matplotlib.pyplot as plt


def plot(ds1Lin: DualPerceptron, ds1Poly: DualPerceptron, ds1RBF: DualPerceptron,
         ds2Lin: DualPerceptron, ds2Poly: DualPerceptron, ds2RBF: DualPerceptron,
         ds3Lin: DualPerceptron, ds3Poly: DualPerceptron, ds3RBF: DualPerceptron):

    avr_biodeg = (ds1Lin.getPredictAccuracy(), ds2Lin.getPredictAccuracy(), ds3Lin.getPredictAccuracy())
    avr_qsar = (ds1Poly.getPredictAccuracy(), ds2Poly.getPredictAccuracy(), ds3Poly.getPredictAccuracy())
    avr_wine = (ds1RBF.getPredictAccuracy(), ds2RBF.getPredictAccuracy(), ds3RBF.getPredictAccuracy())

    print(avr_biodeg, avr_qsar, avr_wine)
    index = np.arange(3)
    plt.bar(index, avr_biodeg, width=0.30, alpha=0.50, color='blue',
            label='linear kernel')
    plt.bar(index+0.30, avr_qsar, width=0.30, alpha=0.40, color='red',
            label='polynomial kernel')
    plt.bar(index+0.60, avr_wine, width=0.30, alpha=0.40, color='green',
            label='RBF kernel')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Kernel Accuracy')
    plt.xticks(index+0.15, ('Biodeg', 'QSAR', 'Red Wine'))
    plt.legend()
    plt.show()


