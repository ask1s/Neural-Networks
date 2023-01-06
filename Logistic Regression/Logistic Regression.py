import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# TASK 1: load dataset as csv file
read = pd.read_csv("dataset.csv")


# TASK 2: process NaN values - any way you want it
read = read.replace("?", np.NaN)
read = read.fillna(read.mean())
for row in read.columns:
    if read[row].dtypes == 'object':
        read[row] = read[row].fillna(read[row].value_counts().index[0])

read = read.drop([read.columns[10],read.columns[13]], axis=1)

le = LabelEncoder()
lr = LogisticRegression()
for row in read.columns:
    if read[row].dtype=='object':
        read[row] = le.fit_transform(read[row])

read = read.values
input = read[:,0:13]
output = read[:, 13]

minmaxsca = MinMaxScaler(feature_range=(0,1))
rescaled = minmaxsca.fit_transform(input)

# TASK 3: create training and testing set analyze if stratified sampling is necessary
X_train, X_test, Y_train, Y_test = train_test_split(rescaled, output, test_size=0.33)


# TASK 4: define neural net
model = Sequential()
model.add(Dense(8, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# TASK 5: train neural net

model.fit(X_train, Y_train, epochs=200, batch_size=5)
lr.fit(X_train, Y_train)

# TASK 6: evaluate neural net, print confusion matrix
_, accuracy = model.evaluate(X_test, Y_test)
aut = lr.predict(X_test)
print(_)
print('Accuracy: %.2f' % (accuracy*100))
print('Accuracy of logistic regression: %.2f' % (lr.score(X_test,Y_test)))
print(confusion_matrix(Y_test, aut))