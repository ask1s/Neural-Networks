import numpy as np

# for training
x = np.array(([2,9], [1,5],[3,6]), dtype=float)  # [sleeping, studing]
y = np.array(([92], [86], [89]), dtype=float)  # test result

# for testing
X = np.array(([3,10], [3,4],[4,8], [2,8], [3,6],[2,5]), dtype=float)  # [sleeping, studing]
Y = np.array(([92], [86], [90], [90], [88], [86]), dtype=float)

# scailing 0>x,y>1
x = x/np.amax(x, axis=0)
y = y/100
X = X/np.amax(X, axis=0)
Y = Y/100

class NN(object):
    def __init__(self):
        #inputs, outputs, neurons
        self.inputSize = 2
        self.outSize = 1
        self.hiddenSize = 3
        self.deriv = False

        #weights
        self.w1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.w2 = np.random.randn(self.hiddenSize, self.outSize)

    def moveForward(self, x):
        self.z1 = np.dot(x, self.w1)
        self.z2 = self.activation(self.z1)
        self.z3 = np.dot(self.z2, self.w2)

        out = self.activation(self.z3)
        return out

    def activation(self, num):
        if (self.deriv):
            return num*(1-num)

        return 1/(1+np.exp(-num))

    def backward(self, x, y, out):
        self.deriv = True
        self.backError = y - out
        self.errorDelta = self.backError * self.activation(out)

        self.newZ2 = self.errorDelta.dot(self.w2.T)
        self.newZ2Delta = self.newZ2 * self.activation(self.newZ2)

        self.w1 += x.T.dot(self.newZ2Delta)
        self.w2 += self.z2.T.dot(self.errorDelta)

        self.deriv = False

    def train(self, x, y):
        out = self.moveForward(x)
        self.backward(x,y,out)

    def predict(self, X):
        out = self.moveForward(X)
        return out

NS = NN()

for i in range(1000):
    NS.train(x,y)
print("------------------Training------------------")
print("")
print("Input: " + str(x))
print("")
print("True output: " + str(y))
print("")
res1 = NS.moveForward(x)
print("Predicted output: " + str(res1))

print("------------------Predicting------------------")
print("")
print("Input: " + str(X))
print("")
print("True output: " + str(Y))
print("")
res2 = NS.predict(X)
print("Predicted output: " + str(res2))
print("")
print("Error: " + str(np.abs(Y - res2)))