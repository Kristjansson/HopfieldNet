from keras.datasets import mnist
import random
import numpy as np


class HopfieldNet:
    @staticmethod
    def biasFunc(updateVal):
        return 1 if updateVal > 0 else -1

    def __init__(self, size):
        self.size = size
        self.weights = np.zeros(self.size, self.size)
        self.neurons = np.zeros(self.size, 1)

    def input(self, input):
        self.neurons = input

    #check that an input call has been made first
    def singleIteration(self, neuronNum):
        self.neurons[neuronNum] = self.biasFunc(np.dot(self.weights[:, neuronNum], self.neurons))

    def fullIteration(self):
        temp = random.shuffle(np.linspace(0, self.size-1, num=self.size))
        for idx in temp:
            self.singleIteration(idx)

    def learnPattern(self, pattern):
        for outerIndex in range(0, self.size):
            for innerIndex in range(0, self.size):
                if outerIndex != innerIndex:
                    val = (2*pattern[outerIndex]-1)*(2*pattern[innerIndex]-1)
                    self.weights[outerIndex, innerIndex] += val
                    self.weights[innerIndex, outerIndex] += val
                else:
                    self.weights[outerIndex, innerIndex] = 0


#Main
(X_train, y_train), (X_test, y_test) = mnist.load_data()